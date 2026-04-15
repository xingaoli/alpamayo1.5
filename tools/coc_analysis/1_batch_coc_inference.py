#!/usr/bin/env python3
"""
CoC (Chain-of-Causation) Batch Inference Script.

This script performs batch CoC inference on keyframes extracted from meta_actions.
For each keyframe, it infers 21 time points from t0_us-1s to t0_us+1s with 0.1s intervals.

Usage:
    python3 batch_coc_inference.py --chunks chunk_0000 chunk_0001
    python3 batch_coc_inference.py --chunks all  # process all chunks
    python3 batch_coc_inference.py --resume  # resume from last checkpoint
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import physical_ai_av
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset_local import load_physical_aiavdataset_local
from alpamayo1_5 import helper


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def load_env(env_path: str = ".env") -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def load_meta_actions(meta_actions_json_path: Path) -> Optional[Dict]:
    """
    Load meta_actions JSON file.
    
    Returns:
        Dict with keyframes info or None if error
    """
    try:
        with open(meta_actions_json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  Warning: Failed to load {meta_actions_json_path}: {e}")
        return None


def generate_timestamps_us(center_timestamp_us: int, window_sec: float = 1.5, interval_sec: float = 0.1) -> List[int]:
    """
    Generate 31 timestamps in microseconds from (center - 1.5s) to (center + 1.5s).

    Args:
        center_timestamp_us: Center timestamp in microseconds
        window_sec: Time window in seconds (default 1.5s)
        interval_sec: Time interval in seconds (default 0.1s)

    Returns:
        List of 31 timestamps in microseconds
    """
    window_us = int(window_sec * 1_000_000)  # 1.5s = 1,500,000 us
    interval_us = int(interval_sec * 1_000_000)  # 0.1s = 100,000 us

    start_us = center_timestamp_us - window_us
    timestamps_us = [start_us + i * interval_us for i in range(31)]  # 31 points

    return timestamps_us


def run_coc_inference(
    model: Alpamayo1_5,
    processor,
    avdi,
    clip_id: str,
    data_dir: str,
    timestamp_us: int,
) -> Optional[Dict]:
    """
    Run CoC inference for a single timestamp.
    
    Args:
        model: Alpamayo1.5 model
        processor: Tokenizer processor
        avdi: PhysicalAI-AV dataset interface
        clip_id: Clip ID (video UUID)
        timestamp_us: Timestamp in microseconds
    
    Returns:
        Dict with CoC result or None if error
    """
    try:
        # Load data for this timestamp
        data = load_physical_aiavdataset_local(
            clip_id,
            data_dir,
            t0_us=timestamp_us,
        )
        
        # Create messages for inference
        messages = helper.create_message(
            data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        
        # Apply chat template
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Prepare model inputs
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")
        
        # Run inference
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
        
        return {
            "timestamp_us": timestamp_us,
            "timestamp_sec": round(timestamp_us / 1_000_000, 2),
            "cot": extra["cot"][0],
            "pred_xyz": pred_xyz.cpu().tolist(),
            "pred_rot": pred_rot.cpu().tolist(),
        }
        
    except Exception as e:
        print(f"    Warning: Inference failed for clip {clip_id} at t0_us={timestamp_us}: {e}")
        return None


def process_single_video(
    clip_id: str,
    data_dir: str,
    meta_actions_data: Dict,
    model: Alpamayo1_5,
    processor,
    avdi,
    output_json_path: Path,
    resume: bool = False,
) -> Optional[Dict]:
    """
    Process a single video (clip_id) with all its keyframes.
    
    For each keyframe, run inference on 21 timestamps (t0_us-1s to t0_us+1s).
    
    Args:
        clip_id: Video UUID
        meta_actions_data: Meta actions data for this video
        model: Alpamayo1.5 model
        processor: Tokenizer processor
        avdi: PhysicalAI-AV dataset interface
        output_json_path: Path to save results
        resume: Whether to resume from existing checkpoint
    
    Returns:
        Dict with all CoC results or None if error
    """
    # Check if already processed (resume support)
    if resume and output_json_path.exists():
        try:
            with open(output_json_path, 'r') as f:
                existing_result = json.load(f)
            print(f"  [Resume] Found existing result for {clip_id}, skipping...")
            return existing_result
        except Exception as e:
            print(f"  [Resume] Failed to load existing result for {clip_id}, reprocessing...")
    
    keyframes = meta_actions_data.get("keyframes", [])
    if not keyframes:
        print(f"  Warning: No keyframes found for {clip_id}")
        return None
    
    print(f"  Processing {clip_id} ({len(keyframes)} keyframes)")
    
    # Result structure: organized by keyframe
    coc_results = {
        "clip_id": clip_id,
        "video_uuid": meta_actions_data.get("video_uuid", clip_id),
        "num_keyframes": len(keyframes),
        "keyframe_coc_results": [],  # Each entry corresponds to one keyframe
        "inference_timestamp": datetime.now().isoformat(),
    }
    
    total_timestamps = 0
    success_timestamps = 0
    
    # Process each keyframe
    for kf_idx, keyframe in enumerate(keyframes):
        kf_timestamp_us = keyframe["timestamp_us"]
        kf_frame_index = keyframe["frame_index"]
        kf_long_action = keyframe.get("long_action", "Unknown")
        kf_lat_action = keyframe.get("lat_action", "Unknown")
        
        # Generate 21 timestamps around this keyframe
        timestamps_us = generate_timestamps_us(kf_timestamp_us)
        total_timestamps += len(timestamps_us)
        
        print(f"    Keyframe {kf_idx + 1}/{len(keyframes)}: "
              f"frame={kf_frame_index}, t0_us={kf_timestamp_us} "
              f"({kf_long_action}, {kf_lat_action})")
        
        # Inference for this keyframe
        keyframe_result = {
            "keyframe_index": kf_idx,
            "keyframe_timestamp_us": kf_timestamp_us,
            "keyframe_timestamp_sec": keyframe.get("timestamp_sec"),
            "frame_index": kf_frame_index,
            "long_action": kf_long_action,
            "lat_action": kf_lat_action,
            "coc": None,  # CoC result for the keyframe timestamp itself
            "inference_timestamps": [],  # 21 inference results
        }

        # Run inference for each of the 21 timestamps
        for ts_us in tqdm(timestamps_us, desc=f"      Inference", leave=False):
            result = run_coc_inference(model, processor, avdi, clip_id, data_dir, ts_us)

            if result is not None:
                keyframe_result["inference_timestamps"].append(result)
                success_timestamps += 1
                
                # Store the CoC result for the keyframe timestamp itself
                if ts_us == kf_timestamp_us:
                    keyframe_result["coc"] = result
            else:
                # Still record the failed timestamp
                keyframe_result["inference_timestamps"].append({
                    "timestamp_us": ts_us,
                    "timestamp_sec": round(ts_us / 1_000_000, 2),
                    "error": "Inference failed",
                })
                
                # Mark as failed if keyframe timestamp itself failed
                if ts_us == kf_timestamp_us:
                    keyframe_result["coc"] = {
                        "timestamp_us": ts_us,
                        "timestamp_sec": round(ts_us / 1_000_000, 2),
                        "error": "Inference failed",
                    }
        
        coc_results["keyframe_coc_results"].append(keyframe_result)
    
    # Add statistics
    coc_results["statistics"] = {
        "total_inference_count": total_timestamps,
        "success_inference_count": success_timestamps,
        "failed_inference_count": total_timestamps - success_timestamps,
        "success_rate": f"{success_timestamps / total_timestamps * 100:.1f}%" if total_timestamps > 0 else "0%",
    }
    
    # Save results
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(coc_results, f, indent=2, cls=NumpyEncoder)

    print(f"  ✓ Saved results for {clip_id} to {output_json_path.name}")
    print(f"    Success: {success_timestamps}/{total_timestamps} "
          f"({success_timestamps / total_timestamps * 100:.1f}%)")
    
    return coc_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='CoC (Chain-of-Causation) Batch Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific chunks
  python3 batch_coc_inference.py --chunks chunk_0000 chunk_0001
  
  # Process all chunks
  python3 batch_coc_inference.py --chunks all
  
  # Resume from checkpoint
  python3 batch_coc_inference.py --resume
        """
    )
    parser.add_argument(
        '--chunks',
        nargs='+',
        default=None,
        help='Specific chunk IDs to process (e.g., chunk_0000), or "all" to process all'
    )
    parser.add_argument(
        '--meta-actions-dir',
        type=str,
        default=None,
        help='Directory containing meta_actions (default: data_dir/labels/meta_actions)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for CoC results (default: data_dir/labels/coc)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing checkpoint files'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default="ckpts/Alpamayo-1.5-10B",
        help='Path to Alpamayo 1.5 model (default: nvidia/Alpamayo-1.5-10B)'
    )
    args = parser.parse_args()
    
    # Load environment
    script_dir = Path(__file__).parent.parent.parent
    env_path = script_dir / ".env"
    print(env_path)
    env_vars = load_env(env_path)
    
    data_dir = Path(env_vars.get('ALPAMAYO_DATA_DIR', '/home/xingao/data/PhysicalAI-Autonomous-Vehicles'))
    meta_actions_dir = Path(args.meta_actions_dir) if args.meta_actions_dir else data_dir / "labels" / "meta_actions"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "labels" / "coc"
    
    print(f"Data directory: {data_dir}")
    print(f"Meta-actions directory: {meta_actions_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Resume: {args.resume}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine chunks to process
    if args.chunks is None or (len(args.chunks) == 1 and args.chunks[0] == "all"):
        # Find all meta_actions chunks
        chunk_dirs = sorted(meta_actions_dir.glob("meta_actions.chunk_*"))
    else:
        # Process specified chunks
        chunk_dirs = [meta_actions_dir / f"meta_actions.{c}" for c in args.chunks]
        chunk_dirs = [d for d in chunk_dirs if d.exists()]
    
    if not chunk_dirs:
        print(f"No meta_actions chunks found in {meta_actions_dir}")
        return
    
    print(f"\nWill process {len(chunk_dirs)} chunks")
    print("="*80)
    
    # Initialize model and dataset interface
    print("\nLoading model...")
    model = Alpamayo1_5.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    print("✓ Model loaded")
    
    # Process each chunk
    all_results = []
    total_videos = 0
    success_videos = 0
    
    for chunk_dir in chunk_dirs:
        if not chunk_dir.exists():
            print(f"\nWarning: Chunk directory {chunk_dir.name} does not exist, skipping...")
            continue
        
        chunk_name = chunk_dir.name.replace('meta_actions.', '')
        print(f"\n{'='*80}")
        print(f"Processing {chunk_name}...")
        print(f"{'='*80}")
        
        # Find all meta_actions JSON files in this chunk
        meta_actions_files = sorted(chunk_dir.glob("*.meta_actions.json"))
        
        if not meta_actions_files:
            print(f"  No meta_actions files found in {chunk_dir}")
            continue
        
        print(f"  Found {len(meta_actions_files)} videos to process")
        
        # Create output directory for this chunk
        coc_chunk_dir = output_dir / f"coc.{chunk_name}"
        coc_chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each video
        chunk_success = 0
        for meta_actions_file in tqdm(meta_actions_files, desc=f"  {chunk_name}"):
            clip_id = meta_actions_file.stem.replace('.meta_actions', '')
            
            # Load meta_actions data
            meta_actions_data = load_meta_actions(meta_actions_file)
            if meta_actions_data is None:
                continue
            
            total_videos += 1
            
            # Output path for this video's CoC results
            output_json_path = coc_chunk_dir / f"{clip_id}.coc.json"
            
            # Process this video
            result = process_single_video(
                clip_id=clip_id,
                data_dir=data_dir,
                meta_actions_data=meta_actions_data,
                model=model,
                processor=processor,
                avdi=avdi,
                output_json_path=output_json_path,
                resume=args.resume,
            )
            
            if result is not None:
                success_videos += 1
                chunk_success += 1
                all_results.append(result)
        
        print(f"\n  ✓ {chunk_name} complete: {chunk_success}/{len(meta_actions_files)} videos succeeded")
    
    # Generate summary statistics
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        total_inference = sum(r["statistics"]["total_inference_count"] for r in all_results)
        success_inference = sum(r["statistics"]["success_inference_count"] for r in all_results)
        
        print(f"Total videos processed: {success_videos}/{total_videos}")
        print(f"Total inference calls: {total_inference}")
        print(f"Successful inferences: {success_inference}")
        print(f"Failed inferences: {total_inference - success_inference}")
        print(f"Overall success rate: {success_inference / total_inference * 100:.1f}%" if total_inference > 0 else "0%")
    
    print("\n" + "="*80)
    print("✓ CoC batch inference complete!")
    print(f"Output saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
