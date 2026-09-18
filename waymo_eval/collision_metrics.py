#!/usr/bin/env python3
"""Offline collision metrics from stored eval predictions.

Reads the preds.npz written by eval_traj.py and replays the ground-truth
world from the rds_hq set via (clip_id, t0_us): other-road-users' future
boxes (all_object_info tars, world-frame 4x4 + lwh per keyframe) and the road
edge (3d_road_boundaries polylines). No model, no GPU.

Two rates per model (each also computed for the GT trajectory as a sanity
baseline — GT should be near zero):
  object collision    ego OBB overlaps an agent OBB at any future step
                      (rotated-box SAT test, agents replayed at their
                      ground-truth future poses)
  boundary collision  ego box crosses a road-boundary segment of a line the
                      GT never hugs: per clip, polylines any GT trajectory
                      comes within --exclude-d of are treated as huggable
                      centre/opposing-lane lines and dropped (the rds set
                      mixes true curbs with lines the ego legally rides);
                      the remaining lines behave as physical road edges, so
                      the GT reference rate stays ~0

Counting requires ego speed >= MIN_SPEED at that step, so a stationary ego
parked next to a parked car does not count as a collision. Ego yaw is
recovered from trajectory tangents (the model's unicycle parameterization
makes the tangent the heading by construction).

Usage:
  python collision_metrics.py waymo_eval/eval_outputs/val_full_xxx
  python collision_metrics.py <dir> --ego-lwh 4.91,2.07,1.85 --min-speed 0.2
Writes <dir>/collision.csv and prints the summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parent)

import argparse
import csv
import json
import tarfile

import numpy as np

from rds_dataset import RDSWaymoDataset

DT = 0.1
N_FUT = 64
EGO_LWH_DEFAULT = (4.91, 2.07, 1.85)  # waymo test vehicle, metres (module const)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline collision metrics from preds.npz.")
    p.add_argument("eval_dir", help="eval output dir containing preds.npz")
    p.add_argument("--split", default=None, help="default: read from summary.json")
    p.add_argument("--ego-lwh", default=",".join(f"{v:g}" for v in EGO_LWH_DEFAULT),
                   help="ego length,width,height in metres")
    p.add_argument("--min-speed", type=float, default=0.2,
                   help="ego speed (m/s) required to count a contact")
    p.add_argument("--exclude-d", type=float, default=1.2,
                   help="polyline-to-GT min distance (m) below which a boundary "
                        "line is treated as huggable and excluded")
    return p.parse_args()


def yaw_from_traj(xy: np.ndarray) -> np.ndarray:
    d = np.diff(xy, axis=0)                                    # [T-1, 2]
    yaw = np.zeros(len(xy))
    for k in range(len(xy)):
        if k == 0:
            v = d[0]
        elif k == len(xy) - 1:
            v = d[-1]
        else:
            v = xy[k + 1] - xy[k - 1]
        if np.hypot(*v) > 1e-3:
            yaw[k] = np.arctan2(v[1], v[0])
        else:
            yaw[k] = yaw[k - 1]
    return yaw


def obb_corners(cx, cy, yaw, hl, hw) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return (np.stack([cx, cy])[:, None]
            + R @ np.array([[hl, hl, -hl, -hl],
                            [hw, -hw, -hw, hw]])).T


def obb_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Rotated-box overlap via SAT. ``a``/``b`` are [4, 2] corner sets."""
    for box in (a, b):
        for i in range(4):
            e = box[(i + 1) % 4] - box[i]
            n = np.array([-e[1], e[0]])
            n = n / (np.hypot(*n) + 1e-12)
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def seg_cross_box(edges_a: np.ndarray, segs: np.ndarray) -> bool:
    """Any of ``edges_a`` [4, 2, 2] intersecting any segment in ``segs`` [M, 2, 2]."""
    p1, p2 = edges_a[:, 0:1, :], edges_a[:, 1:2, :]     # [4, 1, 2]
    q1, q2 = segs[None, :, 0, :], segs[None, :, 1, :]   # [1, M, 2]
    r, s = p2 - p1, q2 - q1
    denom = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((q1 - p1)[..., 0] * s[..., 1] - (q1 - p1)[..., 1] * s[..., 0]) / denom
        u = ((q1 - p1)[..., 0] * r[..., 1] - (q1 - p1)[..., 1] * r[..., 0]) / denom
    hit = (np.abs(denom) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return bool(hit.any())


# ------------------------------------------------------------------ replay
class ClipReplay:
    """Per-clip ground-truth world: agent tracks + road boundary, world frame."""

    def __init__(self, clip: "RDSWaymoDataset.clip_ids.__class__" | object):
        self.clip = clip
        self.agents = self._load_agents()
        self.boundary_segs = self._load_boundary()

    def _load_agents(self) -> dict[str, dict]:
        import json as J
        cid, root, split = self.clip.clip_id, self.clip.root, self.clip.split
        frames = {}
        with tarfile.open(root / split / "all_object_info" / f"{cid}.tar") as f:
            for m in f.getmembers():
                if not m.name.endswith(".all_object_info.json"):
                    continue
                idx = int(m.name.split("/")[-1][len(cid) + 1:].split(".")[0])
                frames[idx] = J.loads(f.extractfile(m).read())
        ts_of = {int(k): int(t) for k, t in zip(self.clip.keyframe_idx, self.clip.ts_us)}
        tracks: dict[str, list[tuple[int, np.ndarray, float, tuple, bool, str]]] = {}
        for idx, objs in frames.items():
            t = ts_of.get(idx)
            if t is None:
                continue
            for oid, o in objs.items():
                T = np.asarray(o["object_to_world"], dtype=np.float64)
                yaw = np.arctan2(T[1, 0], T[0, 0])
                tracks.setdefault(oid, []).append(
                    (t, T[:3, 3].copy(), yaw,
                     tuple(o["object_lwh"]), bool(o["object_is_moving"]),
                     str(o["object_type"])))
        out = {}
        for oid, rows in tracks.items():
            rows.sort(key=lambda r: r[0])
            out[oid] = {
                "ts": np.asarray([r[0] for r in rows], dtype=np.int64),
                "xyz": np.stack([r[1] for r in rows]),
                "yaw": np.unwrap([r[2] for r in rows]),
                "lwh": np.asarray(rows[0][3]),
                "moving": any(r[4] for r in rows),
                "type": rows[0][5],
            }
        return out

    def _load_boundary(self) -> list[np.ndarray]:
        """Road-boundary polylines, world frame, as a list of [V, 3] vertices."""
        labels = self.clip.map_features("3d_road_boundaries").get("labels", [])
        return [np.asarray(lab["labelData"]["shape3d"]["polyline3d"]["vertices"],
                           dtype=np.float64)
                for lab in labels]


    def local_frame(self, t0_us: int):
        """(world->t0-local fn, yaw offset) matching build_sample's to_local."""
        P = self.clip.pose_at(t0_us)[0]
        R, t = P[:3, :3], P[:3, 3]
        yaw0 = np.arctan2(R[1, 0], R[0, 0])

        def to_local(p_world: np.ndarray) -> np.ndarray:
            return (R.T @ (p_world - t).T).T

        def yaw_local(yaw_world: np.ndarray) -> np.ndarray:
            return yaw_world - yaw0

        return to_local, yaw_local


def traj_collisions(xy: np.ndarray, ego_hw, agents_local, boundary_local, min_speed):
    """(hit_object, hit_boundary) for one [T, 2] trajectory in the t0-local frame.

    ``agents_local``: list of (pos [T, 2], yaw [T], half_lwh [2]).
    ``boundary_local``: [M, 2, 2] segments in the t0-local frame.
    """
    yaw = yaw_from_traj(xy)
    speed = np.hypot(*np.diff(xy, axis=0).T) / DT
    speed = np.concatenate([speed[:1], speed])                 # [T]
    active = speed >= min_speed
    hl, hw = ego_hw
    hit_obj = hit_bnd = False
    for k in range(len(xy)):
        if not active[k]:
            continue
        box = obb_corners(xy[k, 0], xy[k, 1], yaw[k], hl, hw)
        if not hit_obj:
            for pos, ayaw, ahlw in agents_local:
                if abs(pos[k, 0] - xy[k, 0]) > 12 or abs(pos[k, 1] - xy[k, 1]) > 12:
                    continue
                a_box = obb_corners(pos[k, 0], pos[k, 1], ayaw[k], *ahlw[:2])
                if obb_overlap(box, a_box):
                    hit_obj = True
                    break
        if not hit_bnd and len(boundary_local):
            if seg_cross_box(box_edges(box), boundary_local):
                hit_bnd = True
        if hit_obj and hit_bnd:
            break
    return hit_obj, hit_bnd


def box_edges(box: np.ndarray) -> np.ndarray:
    """[4,2] corners -> [4, 2, 2] edges."""
    return np.stack([box, np.roll(box, -1, axis=0)], axis=1)


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    ego_lwh = tuple(float(v) for v in args.ego_lwh.split(","))
    ego_hw = (ego_lwh[0] / 2, ego_lwh[1] / 2)

    split = args.split
    if split is None:
        summary = eval_dir / "summary.json"
        split = json.load(open(summary))["config"]["split"] if summary.exists() else "val"

    data = np.load(eval_dir / "preds.npz")
    preds, gts = data["preds"], data["gts"]
    clip_ids, t0s = data["clip_ids"], data["t0_us"]

    dataset = RDSWaymoDataset(
        split=split, clip_ids=sorted(set(str(c) for c in clip_ids)), preload=False)
    fut_rel = (np.arange(1, N_FUT + 1) * DT * 1e6).astype(np.int64)  # us after t0

    replay = None
    kept_polys = []  # per-clip boundary polylines the GT never hugs (world frame)
    rows = []
    for n, (cid, t0) in enumerate(zip(clip_ids, t0s)):
        if replay is None or replay.clip.clip_id != cid:
            replay = ClipReplay(dataset._get_clip(str(cid)))
            # The rds "road boundary" set mixes true curbs with centre lines /
            # opposing-lane edges that the GT legally hugs or crosses. A line
            # any GT trajectory of this clip comes within --exclude-d of is
            # treated as huggable and dropped; what remains behaves like a
            # physical road edge the GT never touches.
            kept_polys = []
            for v_w in replay.boundary_segs:
                clip_min = 1e9
                for j in np.where(clip_ids == cid)[0]:
                    tl, _ = replay.local_frame(int(t0s[j]))
                    v_l = tl(v_w)[:, :2]
                    clip_min = min(clip_min, np.linalg.norm(
                        gts[j][:, None, :2] - v_l[None, :, :], axis=-1).min())
                if clip_min >= args.exclude_d:
                    kept_polys.append(v_w)

        to_local, yaw_local = replay.local_frame(int(t0))
        fut_ts = int(t0) + fut_rel

        # agents replayed at their GT future poses, in the t0-local frame
        agents_local = []
        pred_all = preds[n, :, :, :2].reshape(-1, 2)
        for tr in replay.agents.values():
            t, xyz, yaw = tr["ts"], tr["xyz"], tr["yaw"]
            alive = (fut_ts >= t[0]) & (fut_ts <= t[-1])
            if not alive.any():
                continue  # track not alive anywhere in the horizon
            # interpolate full 3D: the road is tilted and to_local is a 3D
            # rotation — zeroing z would leak the terrain height into x/y.
            pos_w = np.stack([np.interp(fut_ts, t, xyz[:, i]) for i in range(3)],
                             axis=1)
            pos = to_local(pos_w)[:, :2]
            if (np.linalg.norm(pos - gts[n, :, :2], axis=1).min() > 80
                    and np.linalg.norm(pos[alive][:, None, :] - pred_all[None, :, :],
                                       axis=2).min() > 80):
                continue  # never near this sample's trajectories
            pos = pos.copy()
            pos[~alive] = 1e9  # unreachable coordinate once the track ends
            ang = yaw_local(np.interp(fut_ts, t, yaw))
            agents_local.append((pos, ang, tr["lwh"] / 2))

        # boundary segments (kept polylines only) near this sample, t0-local
        if kept_polys:
            loc = [to_local(v)[:, :2] for v in kept_polys]
            segs_l = np.concatenate(
                [np.stack([v[:-1], v[1:]], axis=1) for v in loc])
            mid = segs_l.mean(axis=1)
            lo = np.minimum(pred_all.min(axis=0), gts[n, :, :2].min(axis=0)) - 25
            hi = np.maximum(pred_all.max(axis=0), gts[n, :, :2].max(axis=0)) + 25
            keep = ((mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0])
                    & (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1]))
            segs_l = segs_l[keep]
        else:
            segs_l = np.zeros((0, 2, 2))

        obj_cols, bnd_cols = [], []
        for s in range(preds.shape[1]):
            o, b = traj_collisions(preds[n, s, :, :2], ego_hw, agents_local, segs_l,
                                   args.min_speed)
            obj_cols.append(o)
            bnd_cols.append(b)
        gt_o, gt_b = traj_collisions(gts[n, :, :2], ego_hw, agents_local, segs_l,
                                     args.min_speed)
        rows.append({"clip_id": str(cid), "t0_us": int(t0),
                     "obj_col_trajs": int(sum(obj_cols)),
                     "bnd_col_trajs": int(sum(bnd_cols)),
                     "gt_obj_col": int(gt_o), "gt_bnd_col": int(gt_b)})
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(clip_ids)} samples ...", flush=True)

    csv_path = eval_dir / "collision.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    N, S = len(rows), preds.shape[1]
    obj_t = sum(r["obj_col_trajs"] for r in rows)
    bnd_t = sum(r["bnd_col_trajs"] for r in rows)
    summary = {
        "n_samples": N, "n_traj_samples": S,
        "object": {
            "traj_rate": obj_t / (N * S),
            "sample_rate_any": sum(r["obj_col_trajs"] > 0 for r in rows) / N,
            "gt_traj_rate": sum(r["gt_obj_col"] for r in rows) / N,
        },
        "boundary": {
            "traj_rate": bnd_t / (N * S),
            "sample_rate_any": sum(r["bnd_col_trajs"] > 0 for r in rows) / N,
            "gt_traj_rate": sum(r["gt_bnd_col"] for r in rows) / N,
        },
        "ego_lwh": list(ego_lwh), "min_speed_mps": args.min_speed,
    }
    with open(eval_dir / "collision_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {eval_dir} ===")
    print(f"{'':<14}{'traj rate':>10}{'sample(any)':>12}{'GT (ref)':>10}")
    for kind in ("object", "boundary"):
        s = summary[kind]
        print(f"{kind + ' coll.':<14}{s['traj_rate']:>10.4f}"
              f"{s['sample_rate_any']:>12.4f}{s['gt_traj_rate']:>10.4f}")
    print(f"\nper-sample csv -> {csv_path}\nsummary        -> {eval_dir / 'collision_summary.json'}")


if __name__ == "__main__":
    main()
