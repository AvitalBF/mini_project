#!/usr/bin/env python
"""Fuse SAMv2 anatomical landmark matching with SPECTRE regional pooling, one record per 3D CT
scan from MosMedData.

Idea: SAMv2 (samv2_model.py) gives a dense, anatomically-meaningful embedding at every voxel.
SPECTRE (extract_spectre.py) gives a strong whole-scan embedding, built internally from a 4x4x4
grid of 64 independently-encoded crops (each 128x128x64 voxels = 64x64x64 mm) that get combined
into one global vector. SPECTRE's *global* embedding throws away where in the scan each crop
sits; this script recovers that by using SAMv2 to locate 6 healthy-lung landmarks (Right/Left x
Upper/Middle/Lower) in every scan, then pooling SPECTRE's 64 per-crop vectors around each matched
3D point. The result is 6 SPECTRE vectors that are anatomically aligned *across patients*, on top
of SPECTRE's intact global embedding and SAMv2's own matched/pooled descriptors.

Pipeline per scan
  1. SAMv2 (samv2_model.py): dense (fine, coarse, semantic) 128-d maps, as in demo_samv2.py.
     The 6 atlas landmarks are query vectors computed once on study_0010 (CT-0, healthy), each
     voxel picked by HU (aerated lung parenchyma, see build_atlas()). For every scan, each
     landmark is matched by argmax cosine similarity (mean of the 3 branches, as in
     `get_sim_embed_semantic_loc`), giving:
       - samv2_zone_coords_mm   (6, 3)   true LPS physical-space mm of each match (via
                                          SimpleITK's direction/origin-aware
                                          TransformContinuousIndexToPhysicalPoint, not just
                                          index*spacing, so it is correct under gantry tilt etc.)
       - samv2_zone_embeddings  (6, 128) mean of (fine, coarse, semantic) at the matched voxel
       - samv2_zone_scores      (6,)     the matched cosine similarity (match-quality diagnostic)
       - samv2_global_pooled    (384,)   global average pool of (fine, coarse, semantic) over the
                                          whole dense map, concatenated
  2. SPECTRE (extract_spectre.py): same LoadImage->...->ResizeWithPadOrCrop(512,512,256) pipeline,
     but the backbone is called directly (not via the model's full forward) so both the 64
     per-crop CLS tokens (1080-d each) and the combiner's whole-scan CLS token survive:
       - spectre_global    (1080,)   intact whole-scan embedding, identical to extract_spectre.py
       - spectre_regional  (6, 1080) each per-crop CLS token weighted by a Gaussian in mm of its
                                      crop-centre's distance to a SAMv2-matched zone coordinate,
                                      then averaged -> one 1080-d vector per anatomical zone.
     Crop centres are converted from ROI-tensor voxel indices to mm via MONAI's tracked
     `MetaTensor.affine` (which, as in any NIfTI, maps to RAS+ world mm); flipping the sign of
     x and y converts that to the same LPS convention SimpleITK returns for SAMv2's coordinates,
     so the two mm frames are directly comparable (verified against this file's own nibabel/ITK
     affines: they agree to the last bit once RAS is flipped to LPS).

Output .npz (outputs/spectre_samv2_fusion.npz): per-scan arrays as above, plus `study_ids`,
`labels`, `paths`, `class_names`, `zone_names`, `model_id`, `preprocessing`, and a convenience
`embeddings` = concat(spectre_global, samv2_global_pooled) (1464-d) so this file also works
directly with evaluate_embeddings.py.

Checkpointing/resume, atomic save and the --save-every cadence follow extract_spectre.py /
extract_ct_fm.py exactly. GPU preprocessing (the only step touching the full-resolution volume,
~190M voxels for SPECTRE, a 2mm-isotropic resample for SAMv2) falls back to CPU on CUDA OOM, as in
extract_spectre.py; this is what lets outlier scans such as study_0958 (unusually large FOV/slice
count) complete instead of aborting the whole run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from extract_ct_fm import CLASSES, DEFAULT_DATA, find_studies
from extract_spectre import build_preprocess as build_spectre_preprocess
from extract_spectre import load_model as load_spectre_model
from extract_spectre import to_crops
from samv2_model import combined_maps, load_samv2, preprocess as samv2_preprocess, query_vectors, similarity_map

MODEL_ID = "spectre-samv2-fusion (cclaess/SPECTRE + SAMv2/UAE-S)"
SAMV2_CKPT_DEFAULT = Path("/workspace/external/samv2/checkpoints/SAMv2_iter_20000.pth")

ATLAS_STUDY_ID = 10  # study_0010, CT-0 (healthy), same reference scan as demo_samv2.py
ZONES = ["Right Upper", "Right Middle", "Right Lower", "Left Upper", "Left Middle", "Left Lower"]
# Voxel (x, y, z) in study_0010's own LPS grid (same convention as demo_samv2.py's LANDMARKS).
# Found by: body mask per slice -> lung = body & HU in (-1000,-500) -> split left/right at the
# body's midline x -> split each side into inferior/middle/superior thirds by z -> centroid of
# each third, snapped to the nearest actual lung voxel. Verified: all 6 HU values are in the
# -850..-920 range, i.e. aerated lung parenchyma, not vessel/bronchus/partial-volume edge.
ATLAS_LANDMARKS_VOXEL = {
    "Right Upper": (179, 259, 38),  # HU -914.0
    "Right Middle": (159, 252, 26),  # HU -852.0
    "Right Lower": (151, 303, 15),  # HU -885.0
    "Left Upper": (324, 257, 38),  # HU -881.0
    "Left Middle": (348, 254, 25),  # HU -893.0
    "Left Lower": (348, 291, 16),  # HU -920.0
}

SPECTRE_ROI = (512, 512, 256)
SPECTRE_CROP = (128, 128, 64)
SPECTRE_GRID = tuple(r // c for r, c in zip(SPECTRE_ROI, SPECTRE_CROP))  # (4, 4, 4)
GAUSSIAN_SIGMA_MM_DEFAULT = 40.0  # ~ 0.6x the 64mm crop pitch: blends into ~1 neighbouring crop


# --------------------------------------------------------------------------------------- SAMv2
def load_ct_lps_image(path: str) -> tuple[sitk.Image, np.ndarray, tuple[float, float, float]]:
    """Like samv2_model.load_ct_lps, but also returns the sitk.Image (needed for its direction-
    and origin-aware physical-point transform; index*spacing alone ignores both)."""
    img = sitk.DICOMOrient(sitk.ReadImage(path), "LPS")
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    sp = tuple(float(s) for s in img.GetSpacing())
    return img, vol, sp


def samv2_forward(samv2: torch.nn.Module, vol: np.ndarray, sp, device: torch.device):
    """Dense SAMv2 maps for one scan. Resampling (the only step touching the full-res volume)
    falls back to CPU on CUDA OOM; the (much smaller) 2mm-isotropic volume is then moved back to
    the model's device for the actual forward pass."""
    try:
        x, ratio = samv2_preprocess(vol, sp, device)
    except RuntimeError as ex:
        if device.type != "cuda":
            raise
        print(f"    SAMv2 GPU resample failed ({type(ex).__name__}), retrying on CPU", flush=True)
        torch.cuda.empty_cache()
        x, ratio = samv2_preprocess(vol, sp, torch.device("cpu"))
        x = x.to(device)
    with torch.no_grad():
        maps = combined_maps(samv2(x))
    return maps, ratio


def build_atlas(samv2: torch.nn.Module, device: torch.device, data_dir: Path) -> dict:
    """SAMv2 query vectors for the 6 healthy-lung zones, computed once on study_0010."""
    path = next(data_dir.glob(f"*/study_{ATLAS_STUDY_ID:04d}.nii*"))
    _, vol, sp = load_ct_lps_image(str(path))
    maps, ratio = samv2_forward(samv2, vol, sp, device)
    atlas = {}
    for zone, pt in ATLAS_LANDMARKS_VOXEL.items():
        hu = float(vol[pt[2], pt[1], pt[0]])
        atlas[zone] = dict(query=query_vectors(maps, pt, ratio), hu=hu)
        print(f"[atlas] {zone:<12} voxel(x,y,z)={pt} HU={hu:.1f}", flush=True)
    return atlas


# --------------------------------------------------------------------------------------- SPECTRE
def crop_centers_mm_lps(affine: np.ndarray) -> np.ndarray:
    """(64, 3) LPS-world mm centre of each SPECTRE crop, in grid_patch's n=(h*n_w+w)*n_d+d order
    (verified against spectre.grid_patch's source, depth fastest)."""
    n_h, n_w, n_d = SPECTRE_GRID
    c_h, c_w, c_d = SPECTRE_CROP
    centers_vox = np.array([
        (h * c_h + c_h / 2, w * c_w + c_w / 2, d * c_d + c_d / 2)
        for h in range(n_h) for w in range(n_w) for d in range(n_d)
    ])
    homo = np.concatenate([centers_vox, np.ones((len(centers_vox), 1))], axis=1)
    world_ras = (affine @ homo.T).T[:, :3]
    return world_ras * np.array([-1.0, -1.0, 1.0])  # RAS -> LPS: flip x, y


def spectre_backbone_features(model: torch.nn.Module, crops: torch.Tensor, max_crops_per_forward: int) -> torch.Tensor:
    """(64, 1, 128, 128, 64) crops -> (64, 513, 1080) backbone tokens (CLS + 512 patches),
    chunked to bound VRAM like extract_spectre.py's max_crops_per_forward."""
    param = next(model.backbone.parameters())
    crops = crops.to(device=param.device, dtype=param.dtype)
    chunk = max_crops_per_forward or crops.shape[0]
    return torch.cat([model.backbone(part) for part in torch.split(crops, chunk, dim=0)], dim=0)


@torch.no_grad()
def spectre_forward(model, crops, grid, device, amp_dtype, max_crops):
    """Returns (global 1080-d CLS embedding, 64x1080 per-crop CLS tokens)."""
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        feats = spectre_backbone_features(model, crops, max_crops)  # (64, 513, 1080)
        per_crop_cls = feats[:, 0, :].float()  # (64, 1080)
        pooled = model._pool_crop_tokens(feats)  # (64, 2160): bit-identical to combine_features's input
        combined = model.feature_combiner(pooled.unsqueeze(0).float(), grid)  # (1, T', 1080)
    return combined[0, 0, :].float(), per_crop_cls


def gaussian_regional_pool(per_crop_cls: np.ndarray, crop_centers_mm: np.ndarray,
                            zone_coords_mm: np.ndarray, sigma_mm: float) -> np.ndarray:
    """(6, 3) zone coords + (64, 1080) per-crop vectors + (64, 3) crop centres -> (6, 1080)."""
    d2 = ((crop_centers_mm[None, :, :] - zone_coords_mm[:, None, :]) ** 2).sum(-1)  # (6, 64)
    w = np.exp(-d2 / (2 * sigma_mm ** 2))
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)  # zones far outside the 256mm ROI still sum to ~1
    return w @ per_crop_cls  # (6, 1080)


# --------------------------------------------------------------------------------------- driver
def prep_tag(args: argparse.Namespace) -> str:
    return f"fusion_atlas{ATLAS_STUDY_ID}_sigma{args.sigma_mm}_roi{SPECTRE_ROI}_crop{SPECTRE_CROP}"


def load_checkpoint(out: Path, args: argparse.Namespace) -> list[dict]:
    if not out.exists():
        return []
    d = np.load(out)
    if str(d["model_id"]) != MODEL_ID or str(d["preprocessing"]) != prep_tag(args):
        raise SystemExit(f"{out} was made with {d['model_id']}/{d['preprocessing']}; use another --out or --overwrite.")
    rows = []
    for i in range(len(d["study_ids"])):
        rows.append(dict(
            sid=int(d["study_ids"][i]), label=int(d["labels"][i]), path=str(d["paths"][i]),
            spectre_global=d["spectre_global"][i], spectre_regional=d["spectre_regional"][i],
            samv2_zone_coords_mm=d["samv2_zone_coords_mm"][i], samv2_zone_embeddings=d["samv2_zone_embeddings"][i],
            samv2_zone_scores=d["samv2_zone_scores"][i], samv2_global_pooled=d["samv2_global_pooled"][i],
        ))
    return rows


def save(out: Path, rows: list[dict], args: argparse.Namespace) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    spectre_global = np.stack([r["spectre_global"] for r in rows]).astype(np.float32)
    samv2_global_pooled = np.stack([r["samv2_global_pooled"] for r in rows]).astype(np.float32)
    np.savez(
        tmp,
        embeddings=np.concatenate([spectre_global, samv2_global_pooled], axis=1),  # evaluate_embeddings.py-ready
        spectre_global=spectre_global,
        spectre_regional=np.stack([r["spectre_regional"] for r in rows]).astype(np.float32),
        samv2_zone_coords_mm=np.stack([r["samv2_zone_coords_mm"] for r in rows]).astype(np.float32),
        samv2_zone_embeddings=np.stack([r["samv2_zone_embeddings"] for r in rows]).astype(np.float32),
        samv2_zone_scores=np.stack([r["samv2_zone_scores"] for r in rows]).astype(np.float32),
        samv2_global_pooled=samv2_global_pooled,
        labels=np.array([r["label"] for r in rows], dtype=np.int64),
        study_ids=np.array([r["sid"] for r in rows], dtype=np.int64),
        paths=np.array([r["path"] for r in rows]),
        class_names=np.array(CLASSES),
        zone_names=np.array(ZONES),
        atlas_study_id=ATLAS_STUDY_ID,
        gaussian_sigma_mm=args.sigma_mm,
        model_id=MODEL_ID,
        preprocessing=prep_tag(args),
    )
    tmp.replace(out)  # atomic: a crash mid-write never corrupts the existing checkpoint


def process_one(sid, path, samv2, spectre_model, atlas, preprocess_gpu, preprocess_cpu,
                 device, amp_dtype, max_crops, sigma_mm) -> dict:
    # --- SAMv2: dense maps, 6-zone match -> mm coords + embeddings + scores + global pool ---
    img_lps, vol, sp = load_ct_lps_image(str(path))
    maps, ratio = samv2_forward(samv2, vol, sp, device)

    zone_coords_mm = np.zeros((len(ZONES), 3), dtype=np.float32)
    zone_embeddings = np.zeros((len(ZONES), 128), dtype=np.float32)
    zone_scores = np.zeros(len(ZONES), dtype=np.float32)
    for i, zone in enumerate(ZONES):
        sim = similarity_map(atlas[zone]["query"], maps, vol.shape)
        zb, yb, xb = np.unravel_index(int(torch.argmax(sim)), sim.shape)
        zone_scores[i] = float(sim[zb, yb, xb])
        zone_coords_mm[i] = img_lps.TransformContinuousIndexToPhysicalPoint((float(xb), float(yb), float(zb)))
        qv = query_vectors(maps, (xb, yb, zb), ratio)
        zone_embeddings[i] = torch.stack(qv).mean(dim=0).cpu().numpy()
    samv2_global_pooled = torch.cat([m.mean(dim=(1, 2, 3)) for m in maps]).cpu().numpy().astype(np.float32)

    # --- SPECTRE: intact global embedding + Gaussian-weighted regional pooling around the 6 zones ---
    try:
        x = preprocess_gpu(str(path))
    except RuntimeError as ex:
        if device.type != "cuda":
            raise
        print(f"    study_{sid:04d}: SPECTRE GPU preprocessing failed ({type(ex).__name__}), retrying on CPU", flush=True)
        torch.cuda.empty_cache()
        x = preprocess_cpu(str(path))
    affine = np.asarray(x.affine)
    crops, grid = to_crops(x)
    global_emb, per_crop_cls = spectre_forward(spectre_model, crops, grid, device, amp_dtype, max_crops)
    centers_mm = crop_centers_mm_lps(affine)
    regional = gaussian_regional_pool(per_crop_cls.cpu().numpy(), centers_mm, zone_coords_mm, sigma_mm)

    out = dict(
        spectre_global=global_emb.cpu().numpy().astype(np.float32),
        spectre_regional=regional.astype(np.float32),
        samv2_zone_coords_mm=zone_coords_mm,
        samv2_zone_embeddings=zone_embeddings,
        samv2_zone_scores=zone_scores,
        samv2_global_pooled=samv2_global_pooled,
    )
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise RuntimeError(f"non-finite values in {k}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="the studies/ directory")
    ap.add_argument("--out", type=Path, default=Path("/workspace/outputs/spectre_samv2_fusion.npz"))
    ap.add_argument("--samv2-ckpt", type=Path, default=SAMV2_CKPT_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N studies (by id)")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16", help="autocast dtype for SPECTRE on CUDA")
    ap.add_argument("--max-crops-per-forward", type=int, default=64, help="lower this if you run out of VRAM")
    ap.add_argument("--sigma-mm", type=float, default=GAUSSIAN_SIGMA_MM_DEFAULT, help="Gaussian pooling bandwidth, mm")
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

    samv2 = load_samv2(str(args.samv2_ckpt), device)
    spectre_model = load_spectre_model(device)
    preprocess_gpu = build_spectre_preprocess(device)
    preprocess_cpu = build_spectre_preprocess(torch.device("cpu"))
    print(f"[model] SAMv2 {sum(p.numel() for p in samv2.parameters())/1e6:.1f}M params; "
          f"SPECTRE {sum(p.numel() for p in spectre_model.parameters())/1e6:.1f}M params, amp={amp_dtype}", flush=True)

    atlas = build_atlas(samv2, device, args.data_dir)

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
            out = process_one(sid, path, samv2, spectre_model, atlas, preprocess_gpu, preprocess_cpu,
                               device, amp_dtype, args.max_crops_per_forward, args.sigma_mm)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        except Exception as ex:  # log and keep going; a rerun retries anything not in the checkpoint
            failed.append(sid)
            print(f"[{i}/{len(todo)}] study_{sid:04d} {cls} FAILED: {type(ex).__name__}: {ex}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        rows.append(dict(sid=sid, label=CLASSES.index(cls), path=str(path), **out))
        times.append(t1 - t0)
        new += 1
        eta_h = np.mean(times) * (len(todo) - i) / 3600
        mem = f" peakVRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GiB" if device.type == "cuda" else ""
        mean_zone_dist = np.mean([np.linalg.norm(out["samv2_zone_coords_mm"][a] - out["samv2_zone_coords_mm"][b])
                                   for a in range(6) for b in range(a + 1, 6)])
        print(f"[{i}/{len(todo)}] study_{sid:04d} {cls} spectre_global={out['spectre_global'].shape} "
              f"spectre_regional={out['spectre_regional'].shape} zone_sim={out['samv2_zone_scores'].mean():.3f} "
              f"mean_pairwise_zone_dist={mean_zone_dist:.1f}mm t={t1-t0:.1f}s{mem} "
              f"total={len(rows)} ETA={eta_h:.2f}h", flush=True)
        if new % args.save_every == 0:
            save(args.out, rows, args)
            print(f"[checkpoint] {len(rows)} records -> {args.out}", flush=True)

    if rows:
        rows.sort(key=lambda r: r["sid"])
        save(args.out, rows, args)
    mean_t = f"{np.mean(times):.1f}s/scan" if times else "n/a"
    print(f"[done] {len(rows)} records in {args.out} ({new} new, {mean_t}); failed={failed}", flush=True)


if __name__ == "__main__":
    main()
