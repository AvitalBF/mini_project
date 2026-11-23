#!/usr/bin/env python
"""Extract one SPECTRE embedding per 3D CT scan from MosMedData.

Model: SPECTRE-Large (`spectre-fm` package, weights from HF `cclaess/SPECTRE`): a ViT-L crop encoder
(SSL + vision-language pretrained) followed by a ViT feature combiner over the crop grid.

Preprocessing follows SPECTRE's official whole-scan linear-probing pipeline
(github.com/cclaess/SPECTRE experiments/cvpr26_fm_for_ct_diag_task_1/extract_feat_LP.py):
    LoadImage -> EnsureChannelFirst -> ScaleIntensityRange(-1000..1000 HU -> 0..1, clip=True)
    -> Orientation(RAS) -> Spacing(0.5, 0.5, 1.0 mm, bilinear)
    -> ResizeWithPadOrCrop(512, 512, 256)   # 256 mm cube around the volume centre, padded with 0 (= air)
    -> grid of 4x4x4 = 64 crops of 128x128x64 -> backbone -> feature combiner
Clipping at -1000 HU maps MosMed's -2048 out-of-FOV padding to 0, before any interpolation.

Embedding: the feature combiner's CLS token (outputs[:, 0]), as in the official LP baseline.
Output .npz has the same keys as extract_ct_fm.py, so it works with evaluate_embeddings.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    Orientation,
    ResizeWithPadOrCrop,
    ScaleIntensityRange,
    Spacing,
)

from extract_ct_fm import CLASSES, DEFAULT_DATA, find_studies

MODEL_ID = "spectre-large (cclaess/SPECTRE)"
HU_MIN, HU_MAX = -1000, 1000
SPACING = (0.5, 0.5, 1.0)
ROI = (512, 512, 256)
CROP = (128, 128, 64)


def build_preprocess(device: torch.device) -> Compose:
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        EnsureType(dtype=torch.float32, device=device),  # resample on the GPU: ~190M voxels at 0.5 mm
        ScaleIntensityRange(a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=SPACING, mode="bilinear"),
        ResizeWithPadOrCrop(spatial_size=ROI),
    ])


def to_crops(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """(1, H, W, D) -> (N, 1, 128, 128, 64) crops in SPECTRE's grid order (depth fastest)."""
    from spectre import grid_patch

    x = x.as_tensor() if hasattr(x, "as_tensor") else x
    return grid_patch(x, CROP)


def load_model(device: torch.device) -> torch.nn.Module:
    """spectre-large with its published weights.

    `from_pretrained(pretrained=True)` breaks with huggingface_hub>=2 (it refuses pickle .pt files),
    so build the identical preset architecture and load the same two files ourselves, with the
    tensor-only unpickler (weights_only=True, no unsafe fallback) and strict key matching.
    """
    from huggingface_hub import hf_hub_download
    from spectre import SpectreImageFeatureExtractor, get_preset

    preset = get_preset("spectre-large")
    model = SpectreImageFeatureExtractor.from_pretrained("spectre-large", pretrained=False)
    for module, url in ((model.backbone, preset.backbone_weights), (model.feature_combiner, preset.feature_combiner_weights)):
        filename = url.split("/resolve/main/")[1].split("?")[0]
        sd = torch.load(hf_hub_download("cclaess/SPECTRE", filename), map_location="cpu", weights_only=True)
        module.load_state_dict(sd, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def embed(model, crops, grid, device, amp_dtype, max_crops) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(crops, grid_size=grid, max_crops_per_forward=max_crops)  # (T', F')
    return out[0].float()  # combiner CLS token


def prep_tag(args: argparse.Namespace) -> str:
    return f"spectre_LP_hu{HU_MIN}_{HU_MAX}_ras_spacing{SPACING}_roi{ROI}_crop{CROP}_cls"


def load_checkpoint(out: Path, args: argparse.Namespace) -> list[dict]:
    if not out.exists():
        return []
    d = np.load(out)
    if str(d["model_id"]) != MODEL_ID or str(d["preprocessing"]) != prep_tag(args):
        raise SystemExit(f"{out} was made with {d['model_id']}/{d['preprocessing']}; use another --out or --overwrite.")
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
    ap.add_argument("--out", type=Path, default=Path("/workspace/outputs/spectre_embeddings.npz"))
    ap.add_argument("--limit", type=int, default=None, help="process only the first N studies (by id)")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16", help="autocast dtype on CUDA")
    ap.add_argument("--max-crops-per-forward", type=int, default=64, help="lower this if you run out of VRAM")
    ap.add_argument("--save-every", type=int, default=50, help="checkpoint to --out every N new scans")
    ap.add_argument("--overwrite", action="store_true", help="ignore an existing --out instead of resuming from it")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        print(f"[device] {torch.cuda.get_device_name()} free={torch.cuda.mem_get_info()[0]/2**30:.1f}GiB")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp] if device.type == "cuda" else None

    studies = find_studies(args.data_dir)
    print(f"[data] {len(studies)} studies {dict((c, sum(s[1] == c for s in studies)) for c in CLASSES)}")
    if args.limit:
        studies = studies[: args.limit]

    preprocess = build_preprocess(device)
    preprocess_cpu = build_preprocess(torch.device("cpu"))  # fallback when GPU resampling runs out of memory
    model = load_model(device)
    print(f"[model] {MODEL_ID}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, amp={amp_dtype}", flush=True)

    rows = [] if args.overwrite else load_checkpoint(args.out, args)
    done = {r["sid"] for r in rows}
    todo = [s for s in studies if s[0] not in done]
    print(f"[resume] {len(done)} already in {args.out}; {len(todo)} to process", flush=True)

    times, failed, new = [], [], 0
    for i, (sid, cls, path) in enumerate(todo, 1):
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                x = preprocess(str(path))
            except RuntimeError as ex:  # MONAI wraps CUDA OOM; large-FOV scans can need >20 GiB at 0.5 mm
                if device.type != "cuda":
                    raise
                print(f"    study_{sid:04d}: GPU preprocessing failed ({type(ex).__name__}), retrying on CPU", flush=True)
                torch.cuda.empty_cache()
                x = preprocess_cpu(str(path))
            crops, grid = to_crops(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            e = embed(model, crops, grid, device, amp_dtype, args.max_crops_per_forward)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            e = e.cpu().numpy()
            if not np.isfinite(e).all():
                raise RuntimeError("non-finite embedding")
        except Exception as ex:  # log and keep going; a rerun retries anything not in the checkpoint
            failed.append(sid)
            print(f"[{i}/{len(todo)}] study_{sid:04d} {cls} FAILED: {type(ex).__name__}: {ex}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        rows.append(dict(emb=e, label=CLASSES.index(cls), sid=sid, path=str(path), shape=list(x.shape[1:])))
        times.append(t2 - t0)
        new += 1
        eta_h = np.mean(times) * (len(todo) - i) / 3600
        mem = f" peakVRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GiB" if device.type == "cuda" else ""
        print(f"[{i}/{len(todo)}] study_{sid:04d} {cls} crops={tuple(crops.shape)} grid={grid} "
              f"x[min,max,mean]=[{float(x.min()):.2f},{float(x.max()):.2f},{float(x.mean()):.3f}] "
              f"emb={e.shape} |e|={np.linalg.norm(e):.1f} prep={t1-t0:.1f}s model={t2-t1:.1f}s{mem} "
              f"total={len(rows)} ETA={eta_h:.2f}h", flush=True)
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
