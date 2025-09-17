import os
import math
import json
import time
import re
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm

# External deps used in training/CrystalForge/retrivel.py
from ase import Atoms
from ase.io import read as ase_read
from ase.data import atomic_numbers as _an, covalent_radii as _cr

from scipy.ndimage import gaussian_filter, fourier_gaussian
from scipy.signal.windows import tukey as tukey1d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from skimage.feature import peak_local_max, match_template
from skimage.registration import phase_cross_correlation
from skimage.transform import resize, rotate as imrotate, warp_polar
from skimage.metrics import structural_similarity as ssim

from pymatgen.core import Structure


# --------------------- utility I/O ---------------------

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def load_gray(p: str) -> np.ndarray:
    im = Image.open(p).convert("L")
    x = np.asarray(im, dtype=np.float32)
    x -= x.min()
    mx = x.max()
    if mx > 0:
        x /= mx
    return x


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(np.nanmin(x)); mx = float(np.nanmax(x))
    if mx <= mn:
        return x
    return (x - mn) / (mx - mn)


def tukey2d(h: int, w: int, alpha: float = 0.25) -> np.ndarray:
    wy = tukey1d(h, alpha=alpha).astype(np.float32)
    wx = tukey1d(w, alpha=alpha).astype(np.float32)
    return np.outer(wy, wx)


def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = x.mean(); s = x.std()
    return (x - m) / (s + 1e-8)


def bandpass_fft(img: np.ndarray, low_sigma: float = 0.0, high_sigma: float = 2.0) -> np.ndarray:
    F = np.fft.fft2(img)
    if high_sigma > 0:
        low = fourier_gaussian(F, sigma=high_sigma)
        F = F - low
    if low_sigma > 0:
        F = fourier_gaussian(F, sigma=low_sigma)
    out = np.fft.ifft2(F).real
    out -= out.min()
    if out.max() > 0:
        out /= out.max()
    return out


# ------------------ FFT peak/lattice ------------------

def lattice_angle_period(
    img: np.ndarray,
    peak_threshold_rel: float = 0.25,
    min_peak_dist: int = 6,
    num_peaks: int = 16
) -> Tuple[float, float, np.ndarray]:
    h, w = img.shape
    f = np.fft.fftshift(np.fft.fft2(zscore(img)))
    mag = np.log1p(np.abs(f)).astype(np.float32)
    cy, cx = h // 2, w // 2
    r0 = max(3, min(h, w) // 100)
    mag[cy - r0:cy + r0 + 1, cx - r0:cx + r0 + 1] = 0.0
    coords = peak_local_max(
        mag, min_distance=min_peak_dist,
        threshold_rel=peak_threshold_rel, num_peaks=num_peaks
    )
    if len(coords) < 2:
        return 0.0, min(h, w) / 12.0, coords
    vecs = coords - np.array([[cy, cx]])
    r = np.sqrt((vecs ** 2).sum(1))
    order = np.argsort(r)
    v = vecs[order[0]]
    angle = float(np.degrees(np.arctan2(-(v[0]), v[1])))
    period_pix = float(max(8.0, min(h, w) / (r[order[0]] + 1e-6) * 2.0))
    return angle, period_pix, coords


# ------------- Render CIF to 2D circles image ---------

def grid_from_cell_xy(cell, pixel_size_A):
    a2 = cell[0, :2]; b2 = cell[1, :2]
    lx = float(np.linalg.norm(a2)); ly = float(np.linalg.norm(b2))
    nx = max(1, int(np.ceil(lx / float(pixel_size_A))))
    ny = max(1, int(np.ceil(ly / float(pixel_size_A))))
    return nx, ny


def frac_to_pixel(u, v, nx, ny, cell=None, pixel_size_A=None):
    if cell is not None and pixel_size_A is not None:
        a2 = cell[0, :2]; b2 = cell[1, :2]
        xy = np.stack([u, v], axis=1) @ np.stack([a2, b2], axis=1)
        fx = xy[:, 0] / float(pixel_size_A)
        fy = xy[:, 1] / float(pixel_size_A)
    else:
        fx = u * float(nx); fy = v * float(ny)
    cx = np.clip(np.round(fx).astype(int), 0, nx - 1)
    cy = np.clip(np.round(fy).astype(int), 0, ny - 1)
    return fx, fy, cy, cx


def stamp_soft_disk(canvas, yc, xc, r_px, amp, soft_edge_px):
    y0 = int(max(0, np.floor(yc - r_px - 2 * soft_edge_px)))
    y1 = int(min(canvas.shape[0], np.ceil(yc + r_px + 2 * soft_edge_px)))
    x0 = int(max(0, np.floor(xc - r_px - 2 * soft_edge_px)))
    x1 = int(min(canvas.shape[1], np.ceil(xc + r_px + 2 * soft_edge_px)))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    d = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
    t0 = r_px - soft_edge_px
    t1 = r_px + soft_edge_px
    w = np.clip(1.0 - (d - t0) / (t1 - t0 + 1e-6), 0.0, 1.0)
    canvas[y0:y1, x0:x1] += float(amp) * w.astype("float32")


def build_hr_by_circles(
    atoms: Atoms, pixel_size_A: float, z_power: float,
    radius_scale: float = 1.0, soft_edge_px: float = 1.0, post_blur_px: float = 0.6
):
    atoms = atoms.copy(); atoms.wrap()
    cell = atoms.cell.array
    nx, ny = grid_from_cell_xy(cell, pixel_size_A)
    frac = atoms.get_scaled_positions(wrap=True)
    u, v = frac[:, 0], frac[:, 1]
    _, _, cy, cx = frac_to_pixel(u, v, nx, ny, cell=cell, pixel_size_A=pixel_size_A)
    cy = cy.astype("float32"); cx = cx.astype("float32")
    Z = np.array([_an[s] for s in atoms.get_chemical_symbols()], dtype=int)
    rad_A = _cr[Z].astype("float32")
    r_px = (rad_A * float(radius_scale) / float(pixel_size_A)).astype("float32")
    r_px = np.clip(r_px, 0.8, 96.0)
    amp = (Z.astype("float32") ** float(z_power))
    if amp.max() > 0:
        amp = amp / amp.max()
    canvas = np.zeros((ny, nx), dtype="float32")
    for yi, xi, ri, ai in zip(cy, cx, r_px, amp):
        stamp_soft_disk(canvas, yi, xi, float(ri), float(ai), float(soft_edge_px))
    if post_blur_px and post_blur_px > 0:
        canvas = gaussian_filter(canvas, sigma=float(post_blur_px), mode="reflect")
    return normalize01(canvas), nx, ny


def crop_border(img: np.ndarray, border_px: int = 0, border_frac: float = 0.02) -> np.ndarray:
    H, W = img.shape
    b = max(border_px, int(round(min(H, W) * float(border_frac))))
    if b <= 0 or b * 2 >= H or b * 2 >= W:
        return img
    return img[b:H - b, b:W - b]


def render_supercell_maxres(
    cif_path: str, pixel_size_A: float, z_power: float = 1.0,
    radius_scale: float = 0.3, soft_edge_px: float = 0.0, post_blur_px: float = 0.0,
    crop_border_frac: float = 0.02, repeat: int = 8
) -> np.ndarray:
    atoms = ase_read(cif_path)
    sc = atoms.repeat((repeat, repeat, 1))
    img, nx, ny = build_hr_by_circles(
        sc, pixel_size_A, z_power, radius_scale, soft_edge_px, post_blur_px
    )
    img = crop_border(img, border_frac=crop_border_frac)
    return img


# ------------------ Scoring pieces ------------------

def _log_fft_mag(img: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(zscore(img)))
    return np.log1p(np.abs(f)).astype(np.float32)


def _resize_to_common(a: np.ndarray, b: np.ndarray):
    Ha, Wa = a.shape; Hb, Wb = b.shape
    Ht, Wt = min(Ha, Hb), min(Wa, Wb)
    if (Ha, Wa) != (Ht, Wt):
        a = np.array(Image.fromarray((a * 255).astype(np.uint8)).resize((Wt, Ht), Image.BILINEAR)) / 255.0
    if (Hb, Wb) != (Ht, Wt):
        b = np.array(Image.fromarray((b * 255).astype(np.uint8)).resize((Wt, Ht), Image.BILINEAR)) / 255.0
    return a.astype(np.float32), b.astype(np.float32)


def ncc_z(a: np.ndarray, b: np.ndarray, alpha=0.25) -> float:
    H, W = a.shape
    w = tukey2d(H, W, alpha)
    A = zscore(a * w); B = zscore(b * w)
    den = np.linalg.norm(A.ravel()) * np.linalg.norm(B.ravel())
    return 0.0 if den < 1e-8 else float(np.dot(A.ravel(), B.ravel()) / den)


def ssim01(a: np.ndarray, b: np.ndarray) -> float:
    ha, wa = a.shape
    hb, wb = b.shape
    win_lim = min(ha, wa, hb, wb)
    if win_lim < 3:
        return 0.0
    if win_lim % 2 == 0:
        win_lim -= 1
    win_size = max(3, min(7, win_lim))
    return float(ssim(a.astype(np.float32), b.astype(np.float32), data_range=1.0,
                     gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                     win_size=win_size, channel_axis=None))


class BigPrecomp:
    __slots__ = ("B", "thB", "PB", "peaksB", "PB_polar")
    def __init__(self, B, thB, PB, peaksB, PB_polar):
        self.B = B; self.thB = thB; self.PB = PB; self.peaksB = peaksB; self.PB_polar = PB_polar


def precompute_big(
    big: np.ndarray, bandpass=True, bp_low=0.0, bp_high=2.0,
    fft_thr=0.25, fft_min_dist=6,
    theta_bins: int = 360, r_min_frac: float = 0.05, r_max_frac: float = 0.48
) -> BigPrecomp:
    B = bandpass_fft(big, bp_low, bp_high) if bandpass else big.astype(np.float32, copy=True)
    thB, PB, peaksB = lattice_angle_period(B, fft_thr, fft_min_dist)
    MB = _log_fft_mag(B)
    H, W = MB.shape; cy, cx = H // 2, W // 2
    r_max = int(min(H, W) * r_max_frac)
    r_min = int(min(H, W) * r_min_frac)
    r_max = max(r_max, r_min + 8)
    PB_polar = warp_polar(MB, center=(cy, cx), radius=r_max, output_shape=(theta_bins, r_max)).astype(np.float32)
    PB_polar = PB_polar[:, r_min:r_max]
    PB_polar = zscore(PB_polar)
    return BigPrecomp(B, thB, PB, peaksB, PB_polar)


def _global_consistency_score(
    ncc_map: np.ndarray, y0: int, x0: int, tile_h: int, tile_w: int,
    border: int = 0, drop_low_frac: float = 0.2
) -> Tuple[float, int]:
    H, W = ncc_map.shape
    if tile_h <= 0 or tile_w <= 0:
        return 0.0, 0
    ys = [y0]
    k = 1
    while True:
        yy = y0 + k * tile_h
        if yy + border >= H - border: break
        ys.append(yy); k += 1
    k = 1
    while True:
        yy = y0 - k * tile_h
        if yy - border < 0: break
        ys.insert(0, yy); k += 1
    xs = [x0]
    k = 1
    while True:
        xx = x0 + k * tile_w
        if xx + border >= W - border: break
        xs.append(xx); k += 1
    k = 1
    while True:
        xx = x0 - k * tile_w
        if xx - border < 0: break
        xs.insert(0, xx); k += 1
    vals = []
    for yy in ys:
        for xx in xs:
            iy = int(np.clip(yy, 0, H - 1)); ix = int(np.clip(xx, 0, W - 1))
            vals.append(float(ncc_map[iy, ix]))
    if not vals:
        return 0.0, 0
    vals = sorted(vals, reverse=True)
    keep = max(1, int(round(len(vals) * (1.0 - drop_low_frac))))
    return float(np.mean(vals[:keep])), len(vals)


def _spec_score_polar(B: np.ndarray, T: np.ndarray, r_min_frac=0.05, r_max_frac=0.48,
                      theta_bins=360, radius_bins=256, scale_sweep=(0.9, 1.1, 0.02)) -> float:
    MB = _log_fft_mag(B)
    H, W = MB.shape; cy, cx = H // 2, W // 2
    r_max = int(min(H, W) * r_max_frac)
    r_min = int(min(H, W) * r_min_frac)
    r_max = max(r_max, r_min + 8)
    PB = warp_polar(MB, center=(cy, cx), radius=r_max, output_shape=(theta_bins, r_max)).astype(np.float32)
    PB = zscore(PB[:, r_min:r_max])
    TT = _log_fft_mag(T)
    Ht, Wt = TT.shape; cyt, cxt = Ht // 2, Wt // 2
    r_maxt = int(min(Ht, Wt) * r_max_frac)
    r_mint = int(min(Ht, Wt) * r_min_frac)
    r_maxt = max(r_maxt, r_mint + 8)
    PT = warp_polar(TT, center=(cyt, cxt), radius=r_maxt, output_shape=(theta_bins, r_maxt)).astype(np.float32)
    PT = zscore(PT[:, r_mint:r_maxt])
    best = -1.0
    scales = np.arange(scale_sweep[0], scale_sweep[1] + 1e-9, scale_sweep[2])
    fa = np.fft.fft(PB, axis=0)
    for s in scales:
        r_new = max(8, int(round(PT.shape[1] * s)))
        TTp = resize(PT, (PT.shape[0], r_new), order=1, preserve_range=True).astype(np.float32)
        TTp = resize(TTp, PB.shape, order=1, preserve_range=True).astype(np.float32)
        fb = np.fft.fft(TTp, axis=0)
        xcorr = np.fft.ifft(fa * np.conj(fb), axis=0).real
        prof = xcorr.mean(axis=1)
        denom = (np.linalg.norm(PB) * np.linalg.norm(TTp) + 1e-8)
        prof /= denom
        cand = float(np.max(prof))
        if cand > best:
            best = cand
    return float(best)


def lattice_match(
    big: np.ndarray, small: np.ndarray,
    bandpass=True, bp_low=0.0, bp_high=2.0,
    fft_thr=0.25, fft_min_dist=6,
    rot_win=8.0, rot_step=2.0,
    scale_win=0.04, scale_step=0.02,
    shift_win=10,
    w_local=0.5, w_global=0.3, w_spec=0.2,
    big_pre: Optional[BigPrecomp] = None
) -> Dict:
    # Precompute or reuse big
    if big_pre is not None:
        B = big_pre.B; thB, PB, peaksB = big_pre.thB, big_pre.PB, big_pre.peaksB
        PB_polar_big = big_pre.PB_polar
    else:
        B = bandpass_fft(big, bp_low, bp_high) if bandpass else big.copy()
        thB, PB, peaksB = lattice_angle_period(B, fft_thr, fft_min_dist)
        MB = _log_fft_mag(B)
        H, W = MB.shape; cy, cx = H // 2, W // 2
        r_max = int(min(H, W) * 0.48); r_min = int(min(H, W) * 0.05)
        r_max = max(r_max, r_min + 8)
        PB_polar_big = warp_polar(MB, center=(cy, cx), radius=r_max, output_shape=(360, r_max)).astype(np.float32)
        PB_polar_big = zscore(PB_polar_big[:, r_min:r_max])

    S = bandpass_fft(small, bp_low, bp_high) if bandpass else small.copy()
    thS, PS, peaksS = lattice_angle_period(S, fft_thr, fft_min_dist)
    dtheta = thS - thB
    s0 = max(0.1, min(5.0, PS / (PB + 1e-6)))

    rots = np.arange(dtheta - rot_win, dtheta + rot_win + 1e-6, rot_step).tolist()
    scales = (s0 * (1.0 + np.arange(-scale_win, scale_win + 1e-9, scale_step))).tolist()
    coarse = {"score": -1.0}
    H, W = B.shape
    for s in scales:
        h = max(8, int(round(S.shape[0] * s)))
        w = max(8, int(round(S.shape[1] * s)))
        if h >= H or w >= W: continue
        tmpl_s = resize(S, (h, w), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
        tmpl_s = normalize01(tmpl_s)
        for ang in rots:
            tmpl_sr = imrotate(tmpl_s, ang, resize=False, preserve_range=True).astype(np.float32)
            tmpl_sr = normalize01(tmpl_sr)
            res = match_template(B, tmpl_sr, pad_input=True)
            ij = np.unravel_index(np.argmax(res), res.shape)
            score = float(res[ij])
            if score > coarse["score"]:
                coarse = {"score": score, "y": int(ij[0]), "x": int(ij[1]),
                          "h": h, "w": w, "scale": s, "rot": ang, "tmpl_best": tmpl_sr, "ncc_map": res}

    if coarse["score"] < 0:
        return {"ok": False, "reason": "coarse_failed"}

    best = {"score": -1.0}
    y0, x0 = coarse["y"], coarse["x"]
    hh, ww = coarse["h"], coarse["w"]
    tmpl = coarse["tmpl_best"]
    step = max(1, shift_win // 4)
    for dy in range(-shift_win, shift_win + 1, step):
        for dx in range(-shift_win, shift_win + 1, step):
            top = y0 + dy; left = x0 + dx
            patch = B[top:top + hh, left:left + ww]
            if patch.shape != tmpl.shape: continue
            try:
                shift, _, _ = phase_cross_correlation(patch, tmpl, upsample_factor=10)
                sy, sx = shift
            except Exception:
                sy, sx = 0.0, 0.0
            ty = int(round(top + sy)); tx = int(round(left + sx))
            patch2 = B[ty:ty + hh, tx:tx + ww]
            if patch2.shape != tmpl.shape: continue
            nccv = ncc_z(patch2, tmpl, alpha=0.25)
            ssv = ssim01(patch2, tmpl)
            fused_local = 0.7 * nccv + 0.3 * ssv
            if fused_local > best.get("score", -1.0):
                best = {"score": fused_local, "ncc": nccv, "ssim": ssv,
                        "y": ty, "x": tx, "h": hh, "w": ww,
                        "scale": coarse["scale"], "rot": coarse["rot"]}

    if best["score"] < 0:
        best = {"score": float(coarse["score"]), "ncc": float(coarse["score"]), "ssim": float("nan"),
                "y": int(coarse["y"]), "x": int(coarse["x"]), "h": int(coarse["h"]), "w": int(coarse["w"]),
                "scale": float(coarse["scale"]), "rot": float(coarse["rot"])}

    glob_score, _ = _global_consistency_score(coarse["ncc_map"], y0=coarse["y"], x0=coarse["x"],
                                              tile_h=coarse["h"], tile_w=coarse["w"], border=2, drop_low_frac=0.2)
    try:
        spec_score = _spec_score_polar(B, tmpl, r_min_frac=0.05, r_max_frac=0.48,
                                       theta_bins=360, radius_bins=256, scale_sweep=(0.9, 1.1, 0.02))
    except Exception:
        spec_score = 0.0
    fused_score = float(w_local * best['score'] + w_global * glob_score + w_spec * spec_score)
    out = dict(best)
    out.update({"ok": True, "coarse_angle": float(dtheta), "coarse_scale": float(s0),
                "local_score": float(best['score']),
                "global_score": float(glob_score),
                "spec_score": float(spec_score),
                "fused_score": fused_score})
    return out


# ------------------ CIF library + filter ------------------

def get_cif_elements(cif_path: str) -> List[str]:
    try:
        s = Structure.from_file(cif_path)
        return sorted([el.symbol for el in s.composition.elements])
    except Exception:
        return []


def build_cif_index(cif_lib: str, cache_json: Optional[str] = None) -> List[Dict]:
    if cache_json and os.path.exists(cache_json):
        with open(cache_json, "r", encoding="utf-8") as f:
            return json.load(f)
    items = []
    for root, _, files in os.walk(cif_lib):
        for fn in files:
            if fn.lower().endswith(".cif"):
                p = os.path.join(root, fn)
                elems = get_cif_elements(p)
                if elems:
                    items.append({"path": p, "elements": elems})
    if cache_json:
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    return items


def parse_elements_text(s: str) -> List[str]:
    if s is None:
        return []
    s = str(s)
    if not s:
        return []
    parts = re.split(r"[\s,;\-]+", s.strip())
    parts = [p for p in parts if p]
    return sorted(set(parts))


def filter_by_elements(index_items: List[Dict], query_elements: List[str]) -> List[Dict]:
    if not query_elements:
        return index_items
    qs = set(query_elements)
    # exact element set match (as in current retrivel.py)
    return [it for it in index_items if set(it["elements"]) == qs]


# ------------------ Convenience: score all candidates ------------------

def score_candidates(
    big_img: np.ndarray,
    index_items: List[Dict],
    pixel_size_A: float = 0.8,
    repeat: int = 8,
    crop_border_frac: float = 0.02,
    bandpass=True,
    bp_low=0.0,
    bp_high=2.0,
    fft_peak_rel=0.25,
    fft_min_dist=6,
    rot_win=8.0,
    rot_step=2.0,
    scale_win=0.04,
    scale_step=0.02,
    shift_win=10,
    w_local=0.5,
    w_global=0.3,
    w_spec=0.2,
    num_workers: int = 0,
    prefer_threads: bool = True,
) -> List[Dict]:
    big_pre = precompute_big(big_img, bandpass=(not (bandpass is False)), bp_low=bp_low, bp_high=bp_high,
                             fft_thr=fft_peak_rel, fft_min_dist=fft_min_dist,
                             theta_bins=360, r_min_frac=0.05, r_max_frac=0.48)

    def _score_one(it):
        tmpl = render_supercell_maxres(
            it["path"], pixel_size_A, z_power=1.0, radius_scale=0.3, soft_edge_px=0.0,
            post_blur_px=0.0, crop_border_frac=crop_border_frac, repeat=repeat
        )
        res = lattice_match(
            big_img, tmpl,
            big_pre=big_pre,
            bandpass=(not (bandpass is False)),
            bp_low=bp_low, bp_high=bp_high,
            fft_thr=fft_peak_rel, fft_min_dist=fft_min_dist,
            rot_win=rot_win, rot_step=rot_step,
            scale_win=scale_win, scale_step=scale_step,
            shift_win=shift_win,
            w_local=w_local, w_global=w_global, w_spec=w_spec,
        )
        res["path"] = it["path"]
        res["elements"] = it["elements"]
        return res

    if num_workers and num_workers > 0:
        prefer_mode = "threads" if prefer_threads else "processes"
        scored = Parallel(n_jobs=num_workers, prefer=prefer_mode)(
            delayed(_score_one)(it) for it in index_items
        )
    else:
        scored = [_score_one(it) for it in index_items]
    return scored
