#!/usr/bin/env python
"""Bonus: anatomical landmark matching between two MosMedData chest CTs with SAMv2 (UAE-S).

SAMv2 produces a dense 128-d anatomical embedding at every voxel (fine, coarse and semantic maps, see
samv2_model.py). A point picked in Scan A is located in Scan B as the voxel whose embedding has the highest
cosine similarity (mean of the fine, coarse and semantic similarities, as in the official
`get_sim_embed_semantic_loc`).

Pipeline:
  1. Load two scans (default: study_0010 CT-0, 51 slices, and study_1060 CT-2, 39 slices) in LPS orientation.
  2. Compute dense SAMv2 embeddings on the GPU (standalone PyTorch port, checkpoint loaded strict=True).
  3. Four query landmarks in Scan A (picked manually and checked by HU): carina, aortic arch,
     right upper lung, left lower lung.
  4. For each: argmax cosine similarity in Scan B; report source/target voxel + mm coordinates, the score,
     the mean +/- std similarity to 5,000 random body voxels of Scan B (baseline), and the cycle-consistency
     error (match B->A back to Scan A, distance to the original query).
  5. Save a side-by-side figure (results/samv2_anatomical_matching.png).

Weights: SAMv2_iter_20000.pth from ModelScope (viggin/Self-supervised_Anatomical_Embeddings, v1.0.0). Downloaded
automatically to --ckpt if missing and verified by SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage as ndi

from samv2_model import combined_maps, load_ct_lps, load_samv2, preprocess, query_vectors, similarity_map

STUDIES = Path("/workspace/mosmed_full_data/mosmed_data_unzipped/studies")
CKPT_URL = ("https://www.modelscope.cn/api/v1/models/viggin/Self-supervised_Anatomical_Embeddings/repo"
            "?Revision=v1.0.0&FilePath=checkpoints/SAMv2_iter_20000.pth")
CKPT_SHA256 = "a094d5eef867504defdc4c8e1d950835c4eb8aaa19de2027bb1a194781e423e3"

# Query landmarks in Scan A (study_0010), original voxel indices (x, y, z) in LPS orientation:
# x -> patient left, y -> posterior, z -> superior. Picked on axial slices and checked by HU.
LANDMARKS = [
    ("Carina (tracheal bifurcation)", (242, 263, 34), "mediastinum"),  # spur between main bronchi, ~-70 HU
    ("Aortic arch", (280, 245, 38), "mediastinum"),                     # blood pool, ~33 HU
    ("Right upper lung", (165, 255, 39), "lung"),                       # parenchyma, ~-890 HU
    ("Left lower lung", (372, 335, 19), "lung"),                        # parenchyma, ~-880 HU
]
COLORS = ["#e6194b", "#f58231", "#3cb44b", "#4363d8"]
WINDOWS = {"lung": (-600, 1500), "mediastinum": (40, 400)}  # (level, width) in HU


def ensure_ckpt(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ckpt] downloading SAMv2 weights -> {path}")
        urllib.request.urlretrieve(CKPT_URL, path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != CKPT_SHA256:
        raise SystemExit(f"SHA256 mismatch for {path}")
    return path


def find_study(sid: int) -> Path:
    hits = list(STUDIES.glob(f"CT-*/study_{sid:04d}.nii*"))
    if len(hits) != 1:
        raise SystemExit(f"study_{sid:04d} not found under {STUDIES}")
    return hits[0]


def body_mask(vol: np.ndarray) -> np.ndarray:
    """Body (incl. lungs): largest HU>-500 component per slice, holes filled."""
    m = np.zeros(vol.shape, bool)
    for z in range(vol.shape[0]):
        lab, n = ndi.label(vol[z] > -500)
        if n:
            largest = lab == (np.argmax(ndi.sum(np.ones_like(lab), lab, range(1, n + 1))) + 1)
            m[z] = ndi.binary_fill_holes(largest)
    return m


def window(img, level, width):
    return np.clip((img - (level - width / 2)) / width, 0, 1)


def mm(pt_xyz, spacing_xyz):
    return np.asarray(pt_xyz) * np.asarray(spacing_xyz)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-a", type=int, default=10, help="study id of the query scan")
    ap.add_argument("--scan-b", type=int, default=1060, help="study id of the target scan")
    ap.add_argument("--ckpt", type=Path, default=Path("/workspace/external/samv2/checkpoints/SAMv2_iter_20000.pth"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/results/samv2_anatomical_matching.png"))
    ap.add_argument("--n-random", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dev = torch.device("cuda")
    model = load_samv2(str(ensure_ckpt(args.ckpt)), dev)
    print(f"[model] SAMv2 (UAE-S) standalone port, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"{torch.cuda.get_device_name()}")

    scans = {}
    for tag, sid in (("A", args.scan_a), ("B", args.scan_b)):
        path = find_study(sid)
        vol, sp = load_ct_lps(str(path))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        x, ratio = preprocess(vol, sp, dev)
        maps = combined_maps(model(x))
        torch.cuda.synchronize()
        scans[tag] = dict(sid=sid, cls=path.parent.name, vol=vol, sp=sp, ratio=ratio, maps=maps)
        print(f"[scan {tag}] study_{sid:04d} ({path.parent.name}) shape(z,y,x)={vol.shape} "
              f"spacing(x,y,z)=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.1f}) mm -> input {tuple(x.shape[2:])} @2mm, "
              f"fine grid {tuple(maps[0].shape[1:])}, {time.perf_counter()-t0:.2f}s, "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    A, B = scans["A"], scans["B"]

    rng = np.random.default_rng(args.seed)
    bm = np.argwhere(body_mask(B["vol"]))  # (z,y,x)
    rand_zyx = torch.as_tensor(bm[rng.choice(len(bm), args.n_random, replace=False)], device=dev)

    results = []
    print(f"\n{'#':<2} {'landmark':<30} {'A voxel (x,y,z)':<17} {'B match (x,y,z)':<17} {'B match mm':<22} "
          f"{'cos':>6} {'random mean±std':>16} {'z-score':>8} {'cycle err':>10}")
    for i, (name, pa, win) in enumerate(LANDMARKS, 1):
        q = query_vectors(A["maps"], pa, A["ratio"])
        sim = similarity_map(q, B["maps"], B["vol"].shape)
        zb, yb, xb = np.unravel_index(int(torch.argmax(sim)), sim.shape)
        score = float(sim[zb, yb, xb])
        rnd = sim[rand_zyx[:, 0], rand_zyx[:, 1], rand_zyx[:, 2]].float().cpu().numpy()
        # cycle consistency: B match -> back to A
        sim_back = similarity_map(query_vectors(B["maps"], (xb, yb, zb), B["ratio"]), A["maps"], A["vol"].shape)
        za, ya, xa = np.unravel_index(int(torch.argmax(sim_back)), sim_back.shape)
        cyc = float(np.linalg.norm(mm((xa, ya, za), A["sp"]) - mm(pa, A["sp"])))
        hu_b = float(B["vol"][zb, yb, xb])
        r = dict(i=i, name=name, win=win, pa=pa, pb=(int(xb), int(yb), int(zb)), score=score, rnd_mean=float(rnd.mean()),
                 rnd_std=float(rnd.std()), cycle_mm=cyc, hu_a=float(A["vol"][pa[2], pa[1], pa[0]]), hu_b=hu_b)
        results.append(r)
        pmm = mm(r["pb"], B["sp"])
        print(f"{i:<2} {name:<30} {str(pa):<17} {str(r['pb']):<17} "
              f"({pmm[0]:5.1f},{pmm[1]:5.1f},{pmm[2]:5.1f})  {score:6.3f} {rnd.mean():7.3f} ± {rnd.std():.3f} "
              f"{(score-rnd.mean())/rnd.std():8.1f} {cyc:8.1f}mm   HU A={r['hu_a']:.0f} B={hu_b:.0f}")
    s = np.array([r["score"] for r in results]); rm = np.array([r["rnd_mean"] for r in results])
    print(f"\n[summary] mean matched cosine = {s.mean():.3f}; mean random-point cosine = {rm.mean():.3f} "
          f"(n={args.n_random} body voxels/landmark); mean cycle error = "
          f"{np.mean([r['cycle_mm'] for r in results]):.1f} mm")

    # ---------------------------------------------------------------------------------------------- figure
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(len(results), 2, figsize=(9.2, 4.45 * len(results)), facecolor="white")
    for r, (axa, axb), c in zip(results, axes, COLORS):
        lvl, wid = WINDOWS[r["win"]]
        for ax, S, p, side in ((axa, A, r["pa"], "A"), (axb, B, r["pb"], "B")):
            disp = np.maximum(S["vol"][p[2]], -1000)  # display only: -2048 out-of-FOV padding shown as air
            ax.imshow(window(disp, lvl, wid), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
            ax.scatter([p[0]], [p[1]], s=230, facecolors="none", edgecolors=c, linewidths=2.4)
            ax.scatter([p[0]], [p[1]], s=14, color=c)
            ax.text(p[0] + 14, p[1] - 14, str(r["i"]), color=c, fontsize=15, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_edgecolor(c); sp_.set_linewidth(2)
            ax.text(8, 500, "R", color="white", fontsize=9, va="bottom")
            ax.text(504, 500, "L", color="white", fontsize=9, va="bottom", ha="right")
            win_txt = f"W{wid}/L{lvl}"
            if side == "A":
                ax.set_title(f"{r['i']}. {r['name']}\nslice z={p[2]}  ·  voxel (x={p[0]}, y={p[1]})  ·  {win_txt}",
                             fontsize=10, color="black")
            else:
                ax.set_title(f"match: slice z={p[2]}  ·  voxel (x={p[0]}, y={p[1]})\n"
                             f"cosine = {r['score']:.3f}  (random: {r['rnd_mean']:.2f} ± {r['rnd_std']:.2f})",
                             fontsize=10, color="black")
    fig.text(0.27, 0.995, f"Scan A (query): study_{A['sid']:04d} · {A['cls']}\n{A['vol'].shape[0]} slices, "
             f"{A['sp'][0]:.2f} mm pixels", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(0.76, 0.995, f"Scan B (target): study_{B['sid']:04d} · {B['cls']}\n{B['vol'].shape[0]} slices, "
             f"{B['sp'][0]:.2f} mm pixels", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.suptitle("")
    fig.text(0.5, 0.004, "SAMv2 (UAE-S) dense anatomical embeddings · match = argmax cosine similarity "
             "(mean of fine, coarse, semantic) · axial slices, radiological view",
             ha="center", va="bottom", fontsize=8.5, color="#444444")
    plt.tight_layout(rect=(0, 0.012, 1, 0.965), h_pad=1.6, w_pad=0.6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[figure] saved -> {args.out}")


if __name__ == "__main__":
    main()
