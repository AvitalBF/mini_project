#!/usr/bin/env python
"""Extract one TAP-CT embedding per 3D CT scan from MosMedData.

Model: TAP-CT-B-3D (`fomofo/tap-ct-b-3d`), a DINOv2-style 3D ViT-Base (patch 4x8x8, 4 register tokens),
pretrained task-agnostically on CT volumes. Loaded with `trust_remote_code=True` from a pinned, reviewed
revision (REVISION below); the weights are safetensors.

Preprocessing follows the model card exactly (HF image processor `TAPCTProcessor` from the same revision):
    SimpleITK ReadImage -> DICOMOrient("LPS") -> array (D, H, W) = (S, P, L)
    -> trilinear resize of H, W to 224x224 (slices kept at native count / spacing)
    -> pad z with -1024 HU to a multiple of 4
    -> clip HU to [-1008, 822] -> z-score with mean -86.8086, std 322.6347
Clipping maps MosMed's -2048 out-of-FOV padding to -1008 HU (air).

Attention runs through torch SDPA (identical math to the model's non-xformers fallback, verified cos=1.0 in fp32)
so the full N x N attention matrix is never materialised; inference is fp32 by default.
The whole preprocessed volume goes through the model in one forward pass (the model card's
"Default usage"; 3D position embeddings are interpolated to the volume's patch grid).
Embedding (--pool):
    cls       pooler_output = final [CLS] token, 768-d (default; what the model card exposes)
    cls_mean  [CLS] concatenated with the mean of the patch tokens, 1536-d (DINOv2 linear-probe style)
Output .npz has the same keys as extract_ct_fm.py / extract_spectre.py, so it works with evaluate_embeddings.py.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")  # xformers isn't installed; silence the model's import probe

import numpy as np
import SimpleITK as sitk
import torch

from extract_ct_fm import CLASSES, DEFAULT_DATA, find_studies

REPO = "fomofo/tap-ct-b-3d"
REVISION = "7694ac2563707791f8fbb098695ec6571cfdbf4a"  # remote code reviewed at this commit
MODEL_ID = f"{REPO}@{REVISION[:7]}"


def load_volume(path: Path) -> np.ndarray:
    """(D, H, W) int16/float array in raw HU, LPS-oriented, exactly as in the model card."""
    img = sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS")
    return sitk.GetArrayFromImage(img)


def _sdpa_forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
    """Same math as the model's non-xformers Attention.forward (softmax(q k^T * head_dim^-0.5) v, no dropout
    at eval), but via torch SDPA, which never materialises the N x N matrix (~14k tokens for a 72-slice scan)."""
    if attn_bias is not None:
        raise AssertionError("attn_bias (nested tensors) is not supported")
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
    return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))


def load_model_and_processor(device: torch.device, sdpa: bool = True):
    import types

    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    if sdpa:
        n = 0
        for m in model.modules():
            if type(m).__name__ in ("Attention", "MemEffAttention") and hasattr(m, "qkv"):
                m.forward = types.MethodType(_sdpa_forward, m)
                n += 1
        assert n == len(model.model.blocks), f"patched {n} attention modules, expected {len(model.model.blocks)}"
    return model, proc


@torch.no_grad()
def embed(model, x: torch.Tensor, amp_dtype, pool: str) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(x)  # last_hidden_state (1, N, 768) patch tokens; pooler_output (1, 768) CLS
    cls = out.pooler_output[0].float()
    if pool == "cls":
        return cls
    return torch.cat([cls, out.last_hidden_state[0].float().mean(0)])


def prep_tag(args: argparse.Namespace) -> str:
    return f"tapct_card_lps_resize224_padz4_clip-1008_822_zscore_wholevolume_{args.pool}"


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
    ap.add_argument("--out", type=Path, default=Path("/workspace/outputs/tap_ct_embeddings.npz"))
    ap.add_argument("--limit", type=int, default=None, help="process only the first N studies (by id)")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="none",
                    help="autocast dtype on CUDA (default fp32: ~0.2 s/scan anyway, and bf16 drifts to cos~0.98 vs fp32)")
    ap.add_argument("--pool", choices=["cls", "cls_mean"], default="cls")
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

    model, proc = load_model_and_processor(device)
    print(f"[model] {MODEL_ID}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"amp={amp_dtype}, pool={args.pool}", flush=True)

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
            vol = torch.from_numpy(load_volume(path).astype(np.float32))[None, None].to(device)  # (1,1,D,H,W)
            x = proc(vol)["pixel_values"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            e = embed(model, x, amp_dtype, args.pool)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            e = e.cpu().numpy()
            if not np.isfinite(e).all():
                raise RuntimeError("non-finite embedding")
        except Exception as ex:  # log and keep going; a rerun retries anything not in the checkpoint
            failed.append(sid)
            print(f"[{i}/{len(todo)}] study_{sid:04d} {cls} FAILED: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        rows.append(dict(emb=e, label=CLASSES.index(cls), sid=sid, path=str(path), shape=list(x.shape[2:])))
        times.append(t2 - t0)
        new += 1
        eta_h = np.mean(times) * (len(todo) - i) / 3600
        mem = f" peakVRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GiB" if device.type == "cuda" else ""
        print(f"[{i}/{len(todo)}] study_{sid:04d} {cls} input={tuple(x.shape[2:])} "
              f"x[min,max,mean]=[{float(x.min()):.2f},{float(x.max()):.2f},{float(x.mean()):.2f}] "
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
