"""Helpers to load the world-model-gen calibration (f-theta + per-frame extrinsics)
and project ego-frame points onto a camera image.

Coordinate conventions:
  * Ego (front-left-up): x = forward, y = left, z = up. Matches ``labels/*.json``
    which is a list of 4x4 ego_i -> world_0 poses.
  * Camera (OpenCV): x = right, y = down, z = forward. The extrinsics entries in
    ``sensor_extrinsics/<clip>.json`` are 4x4 cam_i -> world_0 poses (verified:
    the static mount ``inv(labels[i]) @ extrinsics[i]`` is constant across frames
    to within ~1e-15, which is the fixed camera mounting on the vehicle).

Intrinsics layout (see the ``comment`` field of the JSON):
  ``[cx, cy, w, h, *poly_6, is_bw_poly]``
  For the wide/cross cameras ``is_bw_poly=1`` and ``poly`` is the BACKWARD
  (radius -> theta) polynomial; for the rear tele cameras ``is_bw_poly=0`` and
  ``poly`` is the FORWARD (theta -> radius) polynomial. We always materialise
  both directions so ``ray2pixel`` works directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import json

import numpy as np
import numpy.polynomial.polynomial as P


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------
def _fit_inverse_poly(th2r_or_r2th: P.Polynomial, mode: str, r_max: float,
                      deg: int) -> P.Polynomial:
    """Numerically invert one direction of the f-theta polynomial by sampling.

    Args:
        th2r_or_r2th: known-direction polynomial.
        mode: 'th2r' (have theta->r, fit r->theta) or 'r2th' (have r->theta, fit theta->r).
        r_max: maximum radius (px) to sample up to.
        deg: degree to refit.

    For ray->pixel we always want the forward (theta->r) polynomial available.
    """
    if mode == "r2th":  # given poly maps r->theta; sample r and refit theta->r
        r_samples = np.linspace(0.0, r_max, 5001)
        th_samples = th2r_or_r2th(r_samples)
        coefs_high_first = np.polyfit(th_samples, r_samples, deg)
        return P.Polynomial(coefs_high_first[::-1])
    elif mode == "th2r":  # given poly maps theta->r; sample theta and refit r->theta
        th_samples = np.linspace(0.0, np.deg2rad(75.0), 5001)
        r_samples = th2r_or_r2th(th_samples)
        coefs_high_first = np.polyfit(r_samples, th_samples, deg)
        return P.Polynomial(coefs_high_first[::-1])
    else:
        raise ValueError(mode)


def _build_intrinsics(v: Sequence, target_w: int | None = None,
                      target_h: int | None = None) -> dict:
    """Build both f-theta polynomial directions from a single intrinsics vector.

    ``v`` is ``[cx, cy, w, h, *poly_6, is_bw_poly]`` (see module docstring).

    If ``target_w``/``target_h`` are given and differ from the native (w, h),
    the model is rescaled to that image size: cx, cy and the radius polynomial
    coefficients are scaled by ``sx = target_w / w``, ``sy = target_h / h``.
    f-theta keeps pixel aspect square at the native resolution, so we scale
    isotropically by the average factor and re-fit the inverse polynomial.
    """
    cx, cy, w, h = v[0], v[1], v[2], v[3]
    poly = np.array(v[4:10], dtype=float)
    is_bw = bool(v[10])
    deg = len(poly) - 1

    scale = 1.0
    if target_w is not None and target_h is not None:
        scale = (target_w / w + target_h / h) / 2.0
        cx *= target_w / w
        cy *= target_h / h
        w, h = int(target_w), int(target_h)
        # radius poly coefficients: r (px) scales by `scale`, theta unchanged.
        # Forward (th->r): r_new = scale * r  => each coef c_k -> scale * c_k.
        # Backward (r->th): theta(r_new/scale) => use poly in scaled r.
        if is_bw:
            poly = poly.copy()
            poly[0] *= scale              # constant term (in px) scales
            # but the r polynomial is theta = sum c_k r^k; rescaling r by s means
            # we want theta = sum c_k (r'/s)^k = sum (c_k / s^k) r'^k.
            for k in range(1, len(poly)):
                poly[k] /= scale ** k
        else:
            poly = poly * scale           # th->r: each coef scales by `scale`

    if is_bw:
        r2th = P.Polynomial(poly)
        th2r = _fit_inverse_poly(r2th, "r2th", max(w, h) * 0.8, deg)
    else:
        th2r = P.Polynomial(poly)
        r2th = _fit_inverse_poly(th2r, "th2r", max(w, h) * 0.8, deg)
    return {"th2r": th2r, "r2th": r2th, "cx": cx, "cy": cy, "w": int(w), "h": int(h),
            "scale": scale}


def load_intrinsics_for_clip(data_root, clip: str, camera: str,
                             target_w: int | None = None,
                             target_h: int | None = None) -> dict:
    """Load the f-theta intrinsics for one camera.

    ``data_root`` is the wfm_gen_data root (str or Path). Returns
    ``{'th2r': Poly, 'r2th': Poly, 'cx', 'cy', 'w', 'h', 'scale'}``.

    Pass ``target_w``/``target_h`` to rescale to the actual video frame size
    (the intrinsics are stored at the native sensor resolution, e.g. 1920x1080,
    while the decoded mp4 frames may be smaller, e.g. 1024x576).
    """
    data_root = Path(data_root)
    data = json.loads((data_root / "calibration/camera_intrinsics" /
                       f"{clip}.json").read_text())
    return _build_intrinsics(data["intrinsics"][camera], target_w, target_h)


def load_extrinsics(data_root, clip: str) -> dict[str, np.ndarray]:
    """Per-camera array of 4x4 ``cam_i -> world_0`` poses (one per frame)."""
    data_root = Path(data_root)
    raw = json.loads((data_root / "calibration/sensor_extrinsics" /
                      f"{clip}.json").read_text())
    return {name: np.asarray(mats, dtype=np.float64) for name, mats in raw.items()}


def static_mount(ego_poses: np.ndarray, cam_poses: np.ndarray) -> np.ndarray:
    """Static mount ``cam -> ego_0`` (constant across frames).

    ``ego_poses[i]`` and ``cam_poses[i]`` are both expressed in world_0 (i.e. the
    ego_0 frame). So ``inv(ego_poses[i]) @ cam_poses[i]`` maps a camera-frame
    point into the ego frame; it should be the same for every i.

    Equivalent & cheaper: just average over frames or pick frame 0.
    """
    return np.linalg.inv(ego_poses[0]) @ cam_poses[0]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def project_ego_to_camera(P_ego: np.ndarray, ego_to_cam: np.ndarray) -> np.ndarray:
    """``P_ego`` is in ego frame at time t. ``ego_to_cam`` is the 4x4 transform
    mapping ego-t -> cam-t at the SAME instant (static mount, so it is the same
    for all t). Returns ``[N, 3]`` cam-frame points."""
    N = P_ego.shape[0]
    homo = np.concatenate([P_ego, np.ones((N, 1))], axis=1)
    return (ego_to_cam @ homo.T).T[:, :3]


def ftheta_ray2pixel(P_cam: np.ndarray, th2r: P.Polynomial,
                     cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame 3D points to f-theta pixel coords.

    Returns ``(pixels [N, 2], valid [N])``. ``valid`` is False for points behind
    the camera or directly on the optical axis (theta = 0).
    """
    norm = np.linalg.norm(P_cam, axis=1, keepdims=True)
    rays = P_cam / norm
    theta = np.arccos(np.clip(rays[:, 2], -1.0, 1.0))
    r_img = th2r(theta)
    rho = np.linalg.norm(rays[:, :2], axis=1)
    small = rho < 1e-9
    rho_safe = np.where(small, 1.0, rho)
    u = cx + r_img * rays[:, 0] / rho_safe
    v = cy + r_img * rays[:, 1] / rho_safe
    pix = np.stack([u, v], axis=1)
    valid = (P_cam[:, 2] > 0.0) & (~small)
    return pix, valid


def project_ego_world_to_pixels(
    P_world: np.ndarray,
    ego_poses: np.ndarray,
    frame_idx: int,
    ego_to_cam: np.ndarray,
    intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Full chain for projecting world-frame points through the camera at ``frame_idx``.

    ``P_world`` is ``[N, 3]`` of points expressed in world_0 (= ego_0) frame.
    ``ego_poses`` is ``[T, 4, 4]`` (labels). ``ego_to_cam`` is the static 4x4.

    Returns ``(pixels [N, 2], valid [N])``.
    """
    # 1) world -> ego_t0
    t0 = ego_poses[frame_idx]
    R_inv = t0[:3, :3].T
    P_ego_t0 = (R_inv @ (P_world - t0[:3, 3]).T).T

    # 2) ego_t0 -> cam_t0 (static)
    P_cam = project_ego_to_camera(P_ego_t0, ego_to_cam)

    # 3) f-theta
    return ftheta_ray2pixel(P_cam, intrinsics["th2r"], intrinsics["cx"], intrinsics["cy"])


def project_ego_local_to_pixels(
    P_ego_local: np.ndarray,
    ego_to_cam: np.ndarray,
    intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-t0-frame points through the camera at the same instant.

    This is what AR1's prediction uses: ``pred_xyz`` is already in the t0-local
    ego frame, so we skip the world transform step.
    """
    P_cam = project_ego_to_camera(P_ego_local, ego_to_cam)
    return ftheta_ray2pixel(P_cam, intrinsics["th2r"], intrinsics["cx"], intrinsics["cy"])


def draw_polyline_on_image(img: np.ndarray,
                           pts_pix: np.ndarray,
                           color=(255, 0, 255),
                           colors: np.ndarray | None = None,
                           radius: int = 4,
                           thickness: int = 2,
                           skip_out_of_bounds: bool = True,
                           label_every: int = 0,
                           labels=None) -> np.ndarray:
    """Draw a connected polyline with circular markers on a copy of ``img``.

    ``pts_pix`` is ``[N, 2]`` (u, v) in RGB pixel coords. ``colors`` may be
    ``[N, 3]`` RGB to give each segment/marker a gradient color (overrides
    ``color``). If ``label_every > 0`` and ``labels`` is given, every k-th
    waypoint is annotated with ``labels[k]``. Out-of-bounds points break the
    polyline (the segment is skipped).
    """
    import cv2
    out = img.copy()
    H, W = out.shape[:2]

    def _bgr(c):
        return (int(c[2]), int(c[1]), int(c[0]))

    for i in range(len(pts_pix)):
        u, v = pts_pix[i]
        ui, vi = int(round(u)), int(round(v))
        c = colors[i] if colors is not None else color
        if 0 <= ui < W and 0 <= vi < H:
            cv2.circle(out, (ui, vi), radius, _bgr(c), -1)
            cv2.circle(out, (ui, vi), radius + 1, (0, 0, 0), 1)
        if i > 0:
            u0, v0 = pts_pix[i - 1]
            ui0, vi0 = int(round(u0)), int(round(v0))
            oob0 = not (0 <= ui0 < W and 0 <= vi0 < H)
            oob1 = not (0 <= ui < W and 0 <= vi < H)
            if skip_out_of_bounds and (oob0 or oob1):
                continue
            c0 = colors[i - 1] if colors is not None else color
            cv2.line(out, (ui0, vi0), (ui, vi), _bgr(c), thickness, cv2.LINE_AA)
        if label_every and labels is not None and i % label_every == 0 \
                and 0 <= ui < W and 0 <= vi < H:
            cv2.putText(out, str(labels[i]), (ui + 8, vi - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _bgr(c), 1, cv2.LINE_AA)
    return out


def gradient_colors(n: int, start=(255, 0, 0), end=(255, 255, 0)) -> np.ndarray:
    """Linear RGB gradient of ``n`` colors from ``start`` to ``end`` ([n, 3])."""
    s, e = np.array(start, float), np.array(end, float)
    t = np.linspace(0, 1, n)[:, None]
    return (s[None, :] * (1 - t) + e[None, :] * t).astype(np.uint8)


def project_corridor(P_ego_local: np.ndarray, half_width: float,
                     ego_to_cam: np.ndarray, intrinsics: dict
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project a ground-plane "driving corridor" around a centre line.

    ``P_ego_local`` is ``[N, 3]`` (centre line, assumed on/near z=0). For each
    point the left/right edges are offset by ``half_width`` (m) perpendicular to
    the centre line's tangent in the xy plane, then both edges are projected.

    Returns ``(left_pix [N,2], left_valid, right_pix [N,2], right_valid)``.
    """
    xy = P_ego_local[:, :2]
    # tangent by central differences
    tang = np.zeros_like(xy)
    tang[1:-1] = xy[2:] - xy[:-2]
    tang[0] = xy[1] - xy[0]
    tang[-1] = xy[-1] - xy[-2]
    n = np.linalg.norm(tang, axis=1, keepdims=True)
    n_safe = np.where(n < 1e-9, 1.0, n)
    t_hat = tang / n_safe
    # perpendicular in xy (rotate +90 deg): (tx,ty) -> (-ty, tx) = left in FLU
    perp = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    left3 = np.concatenate([xy + half_width * perp,
                            P_ego_local[:, 2:3]], axis=1)
    right3 = np.concatenate([xy - half_width * perp,
                             P_ego_local[:, 2:3]], axis=1)
    lp, lv = project_ego_local_to_pixels(left3, ego_to_cam, intrinsics)
    rp, rv = project_ego_local_to_pixels(right3, ego_to_cam, intrinsics)
    return lp, lv, rp, rv


def draw_corridor_on_image(img: np.ndarray,
                           left_pix: np.ndarray, left_valid: np.ndarray,
                           right_pix: np.ndarray, right_valid: np.ndarray,
                           color=(0, 255, 255), alpha: float = 0.45) -> np.ndarray:
    """Fill the polygon between projected left/right edges as a translucent band.

    Pixels whose projection is invalid (behind camera) are dropped, breaking the
    band into sub-polygons. Drawn additively (cv2.addWeighted) for translucency.
    """
    import cv2
    out = img.copy()
    H, W = out.shape[:2]

    def _poly(pts, valid):
        ok = valid & np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
        return pts[ok].astype(np.int32)

    # Build contiguous valid runs and fill each as a quad strip polygon.
    band = np.zeros_like(out)
    N = len(left_pix)
    i = 0
    polys = []
    while i < N - 1:
        if not (left_valid[i] and left_valid[i + 1]
                and right_valid[i] and right_valid[i + 1]):
            i += 1
            continue
        j = i
        while j < N - 1 and (left_valid[j + 1] and right_valid[j + 1]):
            j += 1
        # run [i..j]
        left_run = left_pix[i:j + 1]
        right_run = right_pix[i:j + 1][::-1]
        poly = np.concatenate([left_run, right_run], axis=0).astype(np.int32)
        polys.append(poly)
        i = j + 1
    for poly in polys:
        cv2.fillPoly(band, [poly], color)
    mask = band.sum(axis=2) > 0
    out[mask] = (img[mask].astype(float) * (1 - alpha)
                 + band[mask].astype(float) * alpha).astype(np.uint8)
    # solid outline
    for poly in polys:
        cv2.polylines(out, [poly], True, color, 2, cv2.LINE_AA)
    return out