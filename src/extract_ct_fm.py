#!/usr/bin/env python
"""Extract one CT-FM embedding per 3D CT scan from MosMedData.

Model: project-lighter/ct_fm_feature_extractor (SegResEncoder, SimCLR-pretrained).

Preprocessing follows the CT-FM *pretraining* pipeline
(github.com/project-lighter/CT-FM experiments/fm/{base,frameworks/intrasample_simclr}.yaml):
    LoadImage -> EnsureChannelFirst -> Orientation(SPL) -> Spacing([3,1,1] mm, bilinear)
    -> CropForeground (on raw HU, i.e. keep bbox of HU > 0)
    -> ScaleIntensityRange(-1024..2048 -> 0..1, clip=True)
The HF model card's inference example omits Spacing and crops after scaling; use
`--card-preprocessing` to reproduce that exactly instead.
Clipping at -1024 HU maps MosMed's -2048 out-of-FOV padding to 0.

Embedding: last encoder feature map -> global average pool -> 512-d vector (as in the model card).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForeground,
    DivisiblePad,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    Orientation,
    ScaleIntensityRange,
    Spacing,
)

MODEL_ID = "project-lighter/ct_fm_feature_extractor"
CLASSES = ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]
HU_MIN, HU_MAX = -1024, 2048
DEFAULT_DATA = Path("/workspace/mosmed_full_data/mosmed_data_unzipped/studies")


def find_studies(studies_dir: Path) -> list[tuple[int, str, Path]]:
    """Return (study_id, class_name, path) for every study, sorted by id. Masks are not under studies/."""
    items = []
    for cls in CLASSES:
        for p in sorted((studies_dir / cls).glob("study_*.nii*")):
            m = re.match(r"study_(\d+)\.nii(\.gz)?$", p.name)
            if m:
                items.append((int(m.group(1)), cls, p))
    if not items:
        raise FileNotFoundError(f"No study_*.nii(.gz) files under {studies_dir}/CT-*")
    return sorted(items)


def build_preprocess(card: bool, spacing: tuple[float, float, float]) -> Compose:
    load = [LoadImage(image_only=True), EnsureChannelFirst(), EnsureType(dtype=torch.float32)]
    scale = ScaleIntensityRange(a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True)
    if card:  # exact HF model-card pipeline
        tail = [Orientation(axcodes="SPL"), scale, CropForeground(allow_smaller=True)]
    else:  # CT-FM pretraining pipeline
        tail = [
            Orientation(axcodes="SPL"),
            Spacing(pixdim=spacing, mode="bilinear"),
            CropForeground(allow_smaller=True),  # default select_fn: HU > 0 -> body bbox
            scale,
        ]
    # Encoder downsamples 4x by 2; pad to a multiple of 16 with 0 (= -1024 HU after scaling).
    return Compose(load + tail + [DivisiblePad(k=16, value=0.0)])


def load_model(device: torch.device) -> torch.nn.Module:
    from lighter_zoo import SegResEncoder

    model = SegResEncoder.from_pretrained(MODEL_ID)
    return model.eval().to(device)


@torch.no_grad()
def embed(model: torch.nn.Module, x: torch.Tensor, device: torch.device, fp16: bool) -> torch.Tensor:
    x = x.as_tensor() if hasattr(x, "as_tensor") else x  # drop MetaTensor metadata
    x = x.unsqueeze(0).to(device, non_blocking=True)  # (1, 1, S, P, L)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        feat = model(x)[-1]
    return torch.nn.functional.adaptive_avg_pool3d(feat.float(), 1).flatten()


def prep_tag(args: argparse.Namespace) -> str:
    return "card" if args.card_preprocessing else f"pretrain_spacing{tuple(args.spacing)}"


def load_checkpoint(out: Path, args: argparse.Namespace) -> list[dict]:
    """Return rows already stored in `out` (empty if none). Refuses to mix preprocessing settings."""
    if not out.exists():
        return []
    d = np.load(out)
    if str(d["model_id"]) != MODEL_ID or str(d["preprocessing"]) != prep_tag(args):
        raise SystemExit(f"{out} was made with {d['model_id']}/{d['preprocessing']}, not {MODEL_ID}/{prep_tag(args)}. "
                         "Use a different --out or pass --overwrite.")
    return [dict(emb=e, label=int(l), sid=int(s), path=str(p), shape=list(sh))
            for e, l, s, p, sh in zip(d["embeddings"], d["labels"], d["study_ids"], d["paths"], d["input_shapes"])]


def save(out: Path, rows: list[dict], args: argparse.Namespace) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        embeddings=np.stack([r["emb"] for r in rows]).astype(np.float32),
        labels=np.array([r["label"] for r in rows], dtype=np.int64),
        study_ids=np.array([r["sid"] for r in rows], dtype=np.int64),
        paths=np.array([r["path"] for r in rows]),
        input_shapes=np.array([r["shape"] for r in rows], dtype=np.int64),
        class_names=np.array(CLASSES),
        model_id=MODEL_ID,
        preprocessing=prep_tag(args),
    )
    tmp.replace(out)  # atomic: a crash mid-write never corrupts the existing checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="the studies/ directory")
    ap.add_argument("--out", type=Path, default=Path("/workspace/outputs/ct_fm_embeddings.npz"))
    ap.add_argument("--limit", type=int, default=None, help="process only the first N studies (by id)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--min-free-gb", type=float, default=6.0, help="auto: use GPU only if this much VRAM is free")
    ap.add_argument("--no-fp16", action="store_true", help="disable FP16 autocast on CUDA")
    ap.add_argument("--spacing", type=float, nargs=3, default=[3.0, 1.0, 1.0], help="target S,P,L spacing in mm")
    ap.add_argument("--card-preprocessing", action="store_true", help="use the HF model-card pipeline (no resampling)")
    ap.add_argument("--save-every", type=int, default=50, help="checkpoint to --out every N new scans")
    ap.add_argument("--overwrite", action="store_true", help="ignore an existing --out instead of resuming from it")
    args = ap.parse_args()

    if args.device == "auto":
        free_gb = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0.0
        device = torch.device("cuda" if free_gb >= args.min_free_gb else "cpu")
        print(f"[device] auto: {free_gb:.2f} GiB free VRAM -> {device}")
    else:
        device = torch.device(args.device)
    fp16 = device.type == "cuda" and not args.no_fp16

    studies = find_studies(args.data_dir)
    counts = {c: sum(s[1] == c for s in studies) for c in CLASSES}
    print(f"[data] {len(studies)} studies {counts}")
    if args.limit:
        studies = studies[: args.limit]

    preprocess = build_preprocess(args.card_preprocessing, tuple(args.spacing))
    model = load_model(device)
    print(f"[model] {MODEL_ID}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, fp16={fp16}")

    rows = [] if args.overwrite else load_checkpoint(args.out, args)
    done = {r["sid"] for r in rows}
    todo = [s for s in studies if s[0] not in done]
    print(f"[resume] {len(done)} already in {args.out}; {len(todo)} to process", flush=True)

    times, failed, new = [], [], 0
    for i, (sid, cls, path) in enumerate(todo, 1):
        try:
            t0 = time.perf_counter()
            x = preprocess(str(path))
            t1 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            e = embed(model, x, device, fp16)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            e = e.cpu().numpy()
            if not np.isfinite(e).all():
                raise RuntimeError("non-finite embedding")
        except Exception as ex:  # log and keep going; a rerun retries anything not in the checkpoint
            failed.append(sid)
            print(f"[{i}/{len(todo)}] study_{sid:04d} {cls} FAILED: {type(ex).__name__}: {ex}", flush=True)
            continue
        rows.append(dict(emb=e, label=CLASSES.index(cls), sid=sid, path=str(path), shape=list(x.shape[1:])))
        times.append(t2 - t0)
        new += 1
        eta_h = np.mean(times) * (len(todo) - i) / 3600
        mem = f" peakVRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GiB" if device.type == "cuda" else ""
        print(f"[{i}/{len(todo)}] study_{sid:04d} {cls} input={tuple(x.shape[1:])} "
              f"x[min,max]=[{float(x.min()):.3f},{float(x.max()):.3f}] |e|={np.linalg.norm(e):.1f} "
              f"prep={t1-t0:.1f}s model={t2-t1:.1f}s{mem} total={len(rows)} ETA={eta_h:.2f}h", flush=True)
        if new % args.save_every == 0:
            save(args.out, rows, args)
            print(f"[checkpoint] {len(rows)} embeddings -> {args.out}", flush=True)

    if rows:
        rows.sort(key=lambda r: r["sid"])
        save(args.out, rows, args)
    mean_t = f"{np.mean(times):.1f}s/scan" if times else "n/a"
    print(f"[done] {len(rows)} embeddings in {args.out} ({new} new, {mean_t}); failed={failed}", flush=True)


if __name__ == "__main__":
    main()
