"""Standalone PyTorch re-implementation of SAMv2 / UAE-S inference (no mmcv / mmdet).

Source: github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2 (commit 035bbfc), config
`configs/samv2/samv2_NIHLN.py`, framework `Sam_uniform_cross_volume_dual_head_mean_vector`, checkpoint
`SAMv2_iter_20000.pth` (ModelScope `viggin/Self-supervised_Anatomical_Embeddings`, v1.0.0).

The official code needs PyTorch 1.9 + mmcv 1.4.7 + mmdetection 2.20, which cannot coexist with this container's
PyTorch 2.10 / CUDA 13. The modules below mirror the official ones layer for layer, with identical
parameter names, so the checkpoint loads with strict=True:

  backbone       ResNet3d-18: stem conv 3x7x7 stride 1 (no pool1); stages with spatial strides (2,2,2,2),
                 temporal strides (1,1,1,2); 1x3x3 kernels in layer1-2, 3x3x3 in layer3-4 (inflate);
                 max-pool (2,1,1) on depth after layer1 (pool2). BN3d + ReLU.
  neck           FPN3d over layer1-3 -> "fine" 128-d map at 1/2 resolution (lateral 1x1 + top-down nearest
                 upsampling + one 3x3 conv on level 0; no norm, no activation).
  read_out_head  FPN3d over layer4 -> "coarse" 128-d global map.
  semantic_head  FPN3d over layer1-3 -> "semantic" 128-d map (organ-mask supervised in UAE-S).
All three outputs are L2-normalised along channels (extract_feat).

Preprocessing (tools/utils.read_image): LPS orientation, resample to 2x2x2 mm (linear), clip HU to
[-1024, 3071], rescale to [0, 255], subtract 50. Network input layout is (1, 1, z, y, x).
Matching (tools/interfaces.get_sim_embed_semantic_loc): coarse map upsampled (trilinear) to the fine grid and
re-normalised; similarity = mean of the cosine similarities of the fine, coarse and semantic embeddings; the
similarity map is upsampled to the original image grid and its argmax is the match.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_SPACING = 2.0
HU_RANGE = (-1024.0, 3071.0)
INTENSITY_BIAS = 50.0


# ----------------------------------------------------------------------------------------------- modules
class ConvModule(nn.Module):
    """mmcv ConvModule subset: conv -> (bn) -> (relu); keys `conv.*`, `bn.*` as in the checkpoint."""

    def __init__(self, cin, cout, k, stride=1, padding=0, norm=True, act=True, bias=None):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, k, stride=stride, padding=padding, bias=(not norm) if bias is None else bias)
        self.bn = nn.BatchNorm3d(cout) if norm else None
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x) if self.act is not None else x


class BasicBlock3d(nn.Module):
    def __init__(self, cin, planes, spatial_stride, temporal_stride, inflate, downsample):
        super().__init__()
        k, p = ((3, 3, 3), (1, 1, 1)) if inflate else ((1, 3, 3), (0, 1, 1))
        self.conv1 = ConvModule(cin, planes, k, (temporal_stride, spatial_stride, spatial_stride), p)
        self.conv2 = ConvModule(planes, planes, k, 1, p, act=False)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        return self.relu(self.conv2(self.conv1(x)) + identity)


class ResNet3d18(nn.Module):
    def __init__(self, spatial_strides=(2, 2, 2, 2), temporal_strides=(1, 1, 1, 2),
                 inflate=((0, 0), (0, 0), (1, 1), (1, 1))):
        super().__init__()
        self.conv1 = ConvModule(1, 64, (3, 7, 7), 1, (1, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        cin = 64
        for i in range(4):
            planes = 64 * 2 ** i
            ss, ts = spatial_strides[i], temporal_strides[i]
            down = None
            if ss != 1 or cin != planes:  # same condition as mmaction's make_res_layer
                down = ConvModule(cin, planes, 1, (ts, ss, ss), 0, act=False)
            blocks = [BasicBlock3d(cin, planes, ss, ts, inflate[i][0] == 1, down),
                      BasicBlock3d(planes, planes, 1, 1, inflate[i][1] == 1, None)]
            self.add_module(f"layer{i + 1}", nn.Sequential(*blocks))
            cin = planes

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for i in range(4):
            x = getattr(self, f"layer{i + 1}")(x)
            if i == 0:
                x = self.pool2(x)
            outs.append(x)
        return outs


class FPN3d(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList(ConvModule(c, out_channels, 1, norm=False, act=False) for c in in_channels)
        self.fpn_convs = nn.ModuleList([ConvModule(out_channels, out_channels, 3, 1, 1, norm=False, act=False)])

    def forward(self, inputs):
        lat = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[2:], mode="nearest")
        return self.fpn_convs[0](lat[0])


class SAMv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet3d18()
        self.neck = FPN3d([64, 128, 256])
        self.read_out_head = FPN3d([512])
        self.semantic_head = FPN3d([64, 128, 256])

    @torch.no_grad()
    def forward(self, x):
        """x: (1, 1, z, y, x) preprocessed volume -> (fine, coarse, semantic) L2-normalised 128-d maps."""
        f = self.backbone(x)
        fine = F.normalize(self.neck(f[:3]), dim=1)
        coarse = F.normalize(self.read_out_head([f[3]]), dim=1)
        sem = F.normalize(self.semantic_head(f[:3]), dim=1)
        return fine, coarse, sem


def load_samv2(ckpt_path: str, device: torch.device) -> SAMv2:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    model = SAMv2()
    model.load_state_dict(sd, strict=True)  # every parameter and buffer must match by name and shape
    return model.to(device).eval()


# ------------------------------------------------------------------------------------------- preprocessing
def load_ct_lps(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Raw HU array in LPS orientation, indexed (z, y, x) with x->patient left, y->posterior, z->superior,
    and spacing (sx, sy, sz) in mm."""
    img = sitk.DICOMOrient(sitk.ReadImage(path), "LPS")
    return sitk.GetArrayFromImage(img).astype(np.float32), tuple(float(s) for s in img.GetSpacing())


def preprocess(vol_zyx: np.ndarray, spacing_xyz, device) -> tuple[torch.Tensor, np.ndarray]:
    """Resample to 2 mm isotropic (trilinear), clip/rescale intensities as in SAMv2.
    Returns the (1,1,z,y,x) network input and norm_ratio (x,y,z) = original spacing / 2 mm."""
    sx, sy, sz = spacing_xyz
    z, y, x = vol_zyx.shape
    size = [max(1, int(round(n * s / NORM_SPACING))) for n, s in ((z, sz), (y, sy), (x, sx))]
    t = torch.from_numpy(vol_zyx)[None, None].to(device)
    t = F.interpolate(t, size=size, mode="trilinear", align_corners=False)
    t = t.clamp(*HU_RANGE)
    t = (t - HU_RANGE[0]) / (HU_RANGE[1] - HU_RANGE[0]) * 255.0 - INTENSITY_BIAS
    return t, np.array([sx, sy, sz]) / NORM_SPACING


# ------------------------------------------------------------------------------------------------ matching
def combined_maps(emb):
    """Fine-grid fine/coarse/semantic maps, with coarse upsampled and re-normalised (as in the official code)."""
    fine, coarse, sem = emb
    coarse = F.normalize(F.interpolate(coarse, size=fine.shape[2:], mode="trilinear", align_corners=False), dim=1)
    return fine[0], coarse[0], sem[0]  # each (128, z, y, x)


def query_vectors(maps, pt_xyz_orig, norm_ratio):
    """Embedding vectors at a point given in ORIGINAL voxel indices (x, y, z)."""
    p = np.floor(np.asarray(pt_xyz_orig) * norm_ratio / 2.0).astype(int)  # original -> 2 mm voxel -> fine grid
    zf, yf, xf = maps[0].shape[1:]
    x, y, z = np.minimum(p, [xf - 1, yf - 1, zf - 1])
    return [m[:, z, y, x] for m in maps]


def similarity_map(q_vecs, key_maps, out_shape_zyx):
    """Mean cosine similarity (fine, coarse, semantic) over the key volume, upsampled to its original grid."""
    sim = sum(torch.einsum("c,czyx->zyx", q, k) for q, k in zip(q_vecs, key_maps)) / 3.0
    return F.interpolate(sim[None, None], size=tuple(out_shape_zyx), mode="trilinear", align_corners=False)[0, 0]
