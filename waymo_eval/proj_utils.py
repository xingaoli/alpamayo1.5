"""Helpers to project ego-frame points onto a pinhole camera image and draw
trajectory overlays — used by the waymo rds_hq visualization scripts.

Standalone pinhole-only counterpart of ``wfm_gen_test/coc_test/proj_utils.py``
(that one also handles the f-theta cameras of the wfm_gen_data dataset; waymo
ships pinhole only).

Coordinate conventions:
  * Ego (front-left-up): x = forward, y = left, z = up.  Ego poses come from
    the rds_hq ``vehicle_pose`` stream (the vehicle frame at t0 is the local
    reference for predictions).
  * Camera (OpenCV): x = right, y = down, z = forward.

Intrinsics: rds_hq ``pinhole_intrinsic`` npy = ``[fx, fy, cx, cy, w, h]``;
``build_pinhole_intrinsics`` converts that to the dict used here (with
optional rescale to the decoded video size).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------
def build_pinhole_intrinsics(v: Sequence, target_w: int | None,
                             target_h: int | None) -> dict:
    """Pinhole intrinsics. ``v = [fx, fy, cx, cy, w, h]`` (rds_hq layout).

    Pass ``target_w``/``target_h`` to rescale to the actual decoded frame size
    (the intrinsics are stored at the native sensor resolution, e.g.
    1920x1280, while the decoded mp4 frames may be smaller).
    """
    fx, fy, cx, cy, w, h = v
    scale = 1.0
    if target_w is not None and target_h is not None:
        scale = (target_w / w + target_h / h) / 2.0
        cx *= target_w / w
        cy *= target_h / h
        w, h = int(target_w), int(target_h)
        fx *= target_w / v[4] if v[4] else 1.0
        fy *= target_h / v[5] if v[5] else 1.0
    return {"pinhole": True, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "w": int(w), "h": int(h), "scale": scale}


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


def pinhole_project(P_cam: np.ndarray, fx: float, fy: float,
                    cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame 3D points to pinhole pixel coords.

    Returns ``(pixels [N, 2], valid [N])``. ``valid`` is False for points with
    z <= 0 (behind the camera).
    """
    z = P_cam[:, 2]
    valid = z > 1e-9
    z_safe = np.where(valid, z, 1.0)
    u = cx + fx * P_cam[:, 0] / z_safe
    v = cy + fy * P_cam[:, 1] / z_safe
    pix = np.stack([u, v], axis=1)
    return pix, valid


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
    return pinhole_project(P_cam, intrinsics["fx"], intrinsics["fy"],
                           intrinsics["cx"], intrinsics["cy"])


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


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
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
