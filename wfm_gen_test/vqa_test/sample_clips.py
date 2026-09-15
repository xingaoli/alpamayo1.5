"""Randomly sample wfm_gen_data clips for batch VQA inference.

Picks ``--num`` clip names from the labels under ``<wfm-root>/labels`` that
start with ``--prefix`` and have a matching front-wide mp4, then writes them
(one per line) to the output file for ``vqa_inference.py --clip ...``.

Example::

    PYTHONPATH=src python wfm_gen_test/vqa_test/sample_clips.py \
        --prefix fire-seed1 --num 50 --seed 1 \
        --out wfm_gen_test/vqa_test/ar1_5_wfm_outputs/fire-seed1-1/sampled_clips.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample wfm_gen_data clips for batch VQA.",
    )
    parser.add_argument(
        "--wfm-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "wfm_gen_data",
        help="Root directory of the locally-generated wfm_gen_data.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="fire-seed1",
        help="Clip-name prefix to filter labels by.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=50,
        help="Number of clips to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Sampling seed (recorded in the output file for reproducibility).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output text file: one sampled clip name per line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labels_dir = args.wfm_root / "labels"
    front_wide_dir = args.wfm_root / "camera" / "camera_front_wide_120fov"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    labeled = {
        p.stem for p in labels_dir.glob(f"{args.prefix}*.json")
    }
    mp4s = {p.stem for p in front_wide_dir.glob(f"{args.prefix}*.mp4")}
    pool = sorted(labeled & mp4s)
    missing_mp4 = sorted(labeled - mp4s)
    if missing_mp4:
        print(
            f"[sample] WARNING: {len(missing_mp4)} labeled clip(s) have no "
            f"front-wide mp4; excluded."
        )
    if not pool:
        raise RuntimeError(
            f"No clips with prefix {args.prefix!r} found in {labels_dir}"
        )
    if args.num > len(pool):
        raise RuntimeError(
            f"Requested {args.num} clips but only {len(pool)} available"
        )

    sampled = sorted(random.Random(args.seed).sample(pool, args.num))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(
            f"# sampled_clips prefix={args.prefix} seed={args.seed} "
            f"num={args.num} pool={len(pool)}\n"
        )
        f.writelines(f"{name}\n" for name in sampled)

    print(
        f"[sample] Wrote {len(sampled)} clip names (seed={args.seed}, "
        f"pool={len(pool)}) to {args.out}"
    )


if __name__ == "__main__":
    main()
