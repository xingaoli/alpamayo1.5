#!/usr/bin/env python3
"""
CoC (Chain-of-Causation) Custom Time Range Inference & Visualization Script.

This script performs CoC inference on user-specified time ranges for given clip IDs.
For each time range, it infers at 0.1s intervals, then automatically generates visualization videos.

Usage:
    # Single clip, single time range (with auto-visualization)
    python3 batch_coc_inference_custom.py --clip-id <uuid> --time-range 2.1-7.3
    
    # Single clip, multiple time ranges
    python3 batch_coc_inference_custom.py --clip-id <uuid> --time-range 2.1-7.3 8.5-10.1
    
    # Multiple clips (applies same time ranges to all clips)
    python3 batch_coc_inference_custom.py --clip-id <uuid1> <uuid2> --time-range 2.1-7.3 8.5-10.1
    
    # Disable auto-visualization
    python3 batch_coc_inference_custom.py --clip-id <uuid> --time-range 2.1-7.3 --no-visualize
    
    # Resume from checkpoint
    python3 batch_coc_inference_custom.py --resume
"""
import os
os.chdir('/home/xingao/code/Alpamayo1.5')
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import torch
import numpy as np
import cv2
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


def get_frame_from_dataset(data: Dict) -> np.ndarray:
    """
    Extract the front-view camera frame from dataset.
    
    Returns:
        RGB image as numpy array (H, W, C)
    """
    image_frames = data["image_frames"]  # Shape: (num_cameras, num_frames, C, H, W)
    
    # Find front wide camera (camera index 1 = CAMERA_FRONT_WIDE_120FOV)
    camera_indices = data["camera_indices"].numpy()
    front_wide_idx = np.where(camera_indices == 1)[0]
    
    if len(front_wide_idx) == 0:
        # If front wide not available, use first camera
        camera_idx = 0
    else:
        camera_idx = front_wide_idx[0]
    
    # Get the last frame (most recent) from this camera
    # Shape: (num_frames, C, H, W) -> take last frame -> (C, H, W)
    frame = image_frames[camera_idx, -1]  # Last frame
    
    # Convert CHW to HWC and ensure it's in [0, 255] range
    if frame.dtype == torch.float32:
        # Check if values are in [0, 1] or [0, 255]
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clamp(0, 255)
        frame = frame.to(torch.uint8)
    
    # CHW -> HWC
    frame_hwc = frame.permute(1, 2, 0).numpy()
    
    return frame_hwc


def create_coc_overlay(
    base_image: np.ndarray,
    coc_text: str,
    timestamp_sec: float,
    font_scale=0.5,
    thickness=1,
) -> np.ndarray:
    """
    Create an overlay image with CoC text.
    
    Args:
        base_image: RGB image (H, W, C)
        coc_text: Chain-of-Causation text
        timestamp_sec: Current timestamp in seconds
        font_scale: Font scale for text
        thickness: Text thickness
    
    Returns:
        Overlay image with CoC text
    """
    # Make a copy
    overlay = base_image.copy()
    h, w = overlay.shape[:2]
    
    # Parse CoC text (handle both string and list formats)
    if isinstance(coc_text, list):
        # Flatten nested list structure
        lines = []
        for item in coc_text:
            if isinstance(item, list):
                lines.extend([str(line) for line in item])
            else:
                lines.append(str(item))
    else:
        lines = str(coc_text).split('\n')
    
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 20
    padding = 10
    max_width = w - 2 * padding
    
    # Calculate text box size
    text_lines = []
    total_height = padding
    for line in lines:
        # Wrap long lines
        while len(line) > 80:
            text_lines.append(line[:80])
            line = line[80:]
        text_lines.append(line)
        total_height += line_height
    total_height += padding
    
    # Draw semi-transparent background for text
    overlay_text_bg = overlay.copy()
    cv2.rectangle(overlay_text_bg, 
                  (0, 0), (w, min(total_height, h // 2)),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay_text_bg, 0.7, overlay, 0.3, 0, overlay)
    
    # Draw timestamp
    cv2.putText(overlay, f"Time: {timestamp_sec:.1f}s",
                (padding, padding + 15),
                font, 0.6, (0, 255, 0), 2)
    
    # Draw CoC text
    y_offset = padding + 35
    for i, line in enumerate(text_lines[:10]):  # Limit to 10 lines
        y = y_offset + i * line_height
        if y > h // 2 - padding:
            break
        
        # Text color: white
        cv2.putText(overlay, line, (padding, y),
                    font, font_scale, (255, 255, 255), thickness)
    
    return overlay


def create_video_for_time_range(
    clip_id: str,
    time_range: Tuple[float, float],
    coc_results: Optional[Dict],
    output_video_path: Path,
    fps: int = 10,
    image_size: Optional[Tuple[int, int]] = (640, 480),
) -> None:
    """
    Create a video for a specific time range with CoC overlays.
    
    Args:
        clip_id: Clip ID
        time_range: (start_sec, end_sec)
        coc_results: CoC results dict
        output_video_path: Output video path
        fps: Frames per second
        image_size: Output image size (width, height), None to use original
    """
    start_sec, end_sec = time_range
    
    # Build a lookup for CoC results
    coc_lookup = {}
    if coc_results is not None:
        for tr in coc_results.get("time_ranges", []):
            # Match by index or by time range
            if abs(tr["start_sec"] - start_sec) < 0.01 and abs(tr["end_sec"] - end_sec) < 0.01:
                for ts_result in tr["inference_timestamps"]:
                    coc_lookup[ts_result["timestamp_us"]] = ts_result
                break
    
    # Generate timestamps
    interval_us = int(0.1 * 1_000_000)
    start_us = int(start_sec * 1_000_000)
    end_us = int(end_sec * 1_000_000)
    timestamps_us = []
    current_us = start_us
    while current_us <= end_us:
        timestamps_us.append(current_us)
        current_us += interval_us
    
    print(f"    Creating video: {start_sec}s - {end_sec}s ({len(timestamps_us)} frames)")
    
    # Create video writer
    video_writer = None
    
    for ts_us in tqdm(timestamps_us, desc="      Rendering", leave=False):
        try:
            # Load data for this timestamp
            data = load_physical_aiavdataset_local(
                clip_id,
                t0_us=ts_us,
            )
            
            # Get frame
            frame = get_frame_from_dataset(data)
            
            # Resize if needed
            if image_size is not None:
                frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_LINEAR)
            
            h, w = frame.shape[:2]
            
            # Set video writer size on first frame
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    str(output_video_path), fourcc, fps, (w, h)
                )
            
            # Create overlay with CoC text
            ts_sec = ts_us / 1_000_000
            if ts_us in coc_lookup:
                coc_text = coc_lookup[ts_us].get("cot", "No CoC result available")
            else:
                coc_text = "Loading..."
            
            overlay = create_coc_overlay(frame, coc_text, ts_sec)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            # Write frame
            video_writer.write(frame_bgr)
            
        except Exception as e:
            print(f"\n      Warning: Failed to render frame at {ts_us}us: {e}")
            continue
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"    ✓ Saved video: {output_video_path.name}")
    else:
        print(f"    Warning: No frames were rendered")


def visualize_coc_results(
    clip_id: str,
    time_ranges: List[Tuple[float, float]],
    coc_results: Dict,
    output_dir: Path,
    fps: int = 10,
    output_name: Optional[str] = None,
) -> List[Path]:
    """
    Generate visualization videos for CoC results.
    
    Args:
        clip_id: Clip ID
        time_ranges: List of (start_sec, end_sec) tuples
        coc_results: CoC results dict
        output_dir: Output directory for videos
        fps: Frames per second
        output_name: Optional output video name prefix
    
    Returns:
        List of output video paths
    """
    print(f"\n  {'='*60}")
    print(f"  Generating visualization videos...")
    print(f"  {'='*60}")
    
    output_videos = []
    safe_clip_id = clip_id.replace('/', '_').replace('\\', '_')
    
    for tr_idx, (start_sec, end_sec) in enumerate(time_ranges):
        # Determine output video name
        if output_name and len(time_ranges) == 1:
            video_name = output_name if output_name.endswith('.mp4') else f"{output_name}.mp4"
        else:
            safe_start = str(start_sec).replace('.', '_')
            safe_end = str(end_sec).replace('.', '_')
            video_name = f"{safe_clip_id}_{safe_start}-{safe_end}s.mp4"
        
        output_video_path = output_dir / video_name
        
        # Create video
        create_video_for_time_range(
            clip_id=clip_id,
            time_range=(start_sec, end_sec),
            coc_results=coc_results,
            output_video_path=output_video_path,
            fps=fps,
        )
        
        output_videos.append(output_video_path)
    
    return output_videos


def parse_time_range(time_range_str: str) -> Tuple[float, float]:
    """
    Parse time range string like "2.1-7.3" into (start_sec, end_sec).
    
    Args:
        time_range_str: Time range string in format "start-end"
    
    Returns:
        Tuple of (start_sec, end_sec)
    """
    parts = time_range_str.strip().split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid time range format: '{time_range_str}', expected 'start-end' (e.g., '2.1-7.3')")
    
    try:
        start_sec = float(parts[0])
        end_sec = float(parts[1])
    except ValueError:
        raise ValueError(f"Invalid time values in: '{time_range_str}', expected numbers")
    
    if start_sec >= end_sec:
        raise ValueError(f"Start time must be less than end time: '{time_range_str}'")
    
    return start_sec, end_sec


def generate_timestamps_us(start_sec: float, end_sec: float, interval_sec: float = 0.1) -> List[int]:
    """
    Generate timestamps in microseconds from start_sec to end_sec with given interval.
    
    Args:
        start_sec: Start time in seconds
        end_sec: End time in seconds
        interval_sec: Time interval in seconds (default 0.1s)
    
    Returns:
        List of timestamps in microseconds
    """
    interval_us = int(interval_sec * 1_000_000)  # 0.1s = 100,000 us
    start_us = int(start_sec * 1_000_000)
    end_us = int(end_sec * 1_000_000)
    
    timestamps_us = []
    current_us = start_us
    while current_us <= end_us:
        timestamps_us.append(current_us)
        current_us += interval_us
    
    return timestamps_us


def run_coc_inference(
    model: Alpamayo1_5,
    processor,
    clip_id: str,
    timestamp_us: int,
) -> Optional[Dict]:
    """
    Run CoC inference for a single timestamp.
    
    Args:
        model: Alpamayo1.5 model
        processor: Tokenizer processor
        clip_id: Clip ID (video UUID)
        timestamp_us: Timestamp in microseconds
    
    Returns:
        Dict with CoC result or None if error
    """
    try:
        # Load data for this timestamp
        data = load_physical_aiavdataset_local(
            clip_id,
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
        print(f"      Warning: Inference failed for clip {clip_id} at t0_us={timestamp_us}: {e}")
        return None


def process_single_clip(
    clip_id: str,
    time_ranges: List[Tuple[float, float]],
    model: Alpamayo1_5,
    processor,
    output_json_path: Path,
    resume: bool = False,
    visualize: bool = True,
    fps: int = 10,
) -> Optional[Dict]:
    """
    Process a single clip with specified time ranges.

    Args:
        clip_id: Video UUID
        time_ranges: List of (start_sec, end_sec) tuples
        model: Alpamayo1.5 model
        processor: Tokenizer processor
        output_json_path: Path to save results
        resume: Whether to resume from existing checkpoint
        visualize: Whether to generate visualization videos
        fps: Frames per second for output videos

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
    
    print(f"  Processing {clip_id} ({len(time_ranges)} time ranges)")
    
    # Result structure: organized by time range
    coc_results = {
        "clip_id": clip_id,
        "time_ranges": [],
        "inference_timestamp": datetime.now().isoformat(),
    }
    
    total_timestamps = 0
    success_timestamps = 0
    
    # Process each time range
    for tr_idx, (start_sec, end_sec) in enumerate(time_ranges):
        # Generate timestamps for this range
        timestamps_us = generate_timestamps_us(start_sec, end_sec)
        total_timestamps += len(timestamps_us)
        
        print(f"    Time Range {tr_idx + 1}/{len(time_ranges)}: "
              f"{start_sec}s - {end_sec}s ({len(timestamps_us)} points)")
        
        # Inference for this time range
        time_range_result = {
            "range_index": tr_idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "num_points": len(timestamps_us),
            "inference_timestamps": [],
        }
        
        # Run inference for each timestamp
        for ts_us in tqdm(timestamps_us, desc=f"      Inference", leave=False):
            result = run_coc_inference(model, processor, clip_id, ts_us)
            
            if result is not None:
                time_range_result["inference_timestamps"].append(result)
                success_timestamps += 1
            else:
                # Still record the failed timestamp
                time_range_result["inference_timestamps"].append({
                    "timestamp_us": ts_us,
                    "timestamp_sec": round(ts_us / 1_000_000, 2),
                    "error": "Inference failed",
                })
        
        # Add statistics for this time range
        time_range_result["statistics"] = {
            "total_count": len(timestamps_us),
            "success_count": len(time_range_result["inference_timestamps"]) - 
                            sum(1 for t in time_range_result["inference_timestamps"] if "error" in t),
            "failed_count": sum(1 for t in time_range_result["inference_timestamps"] if "error" in t),
        }
        
        coc_results["time_ranges"].append(time_range_result)
    
    # Add overall statistics
    coc_results["statistics"] = {
        "total_time_ranges": len(time_ranges),
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

    # Auto-visualize
    if visualize:
        output_video_dir = output_json_path.parent
        visualize_coc_results(
            clip_id=clip_id,
            time_ranges=time_ranges,
            coc_results=coc_results,
            output_dir=output_video_dir,
            fps=fps,
        )

    return coc_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='CoC (Chain-of-Causation) Custom Time Range Inference & Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single clip, single time range (with auto-visualization)
  python3 batch_coc_inference_custom.py --clip-id <uuid> --time-range 2.1-7.3
  
  # Single clip, multiple time ranges
  python3 batch_coc_inference_custom.py --clip-id <uuid> --time-range 2.1-7.3 8.5-10.1 15.0-20.5
  
  # Multiple clips (applies same time ranges to all clips)
  python3 batch_coc_inference_custom.py --clip-id <uuid1> <uuid2> --time-range 2.1-7.3 8.5-10.1
  
  # Disable auto-visualization
  python3 batch_coc_inference_custom.py --clip-id <uuid> --time-range 2.1-7.3 --no-visualize
  
  # Resume from checkpoint
  python3 batch_coc_inference_custom.py --resume
        """
    )
    parser.add_argument(
        '--clip-id',
        nargs='+',
        required=True,
        help='One or more clip IDs (video UUIDs) to process'
    )
    parser.add_argument(
        '--time-range',
        nargs='+',
        required=True,
        help='Time ranges in seconds, format: "start-end" (e.g., "2.1-7.3" "8.5-10.1")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for CoC results and videos (default: data_dir/labels/coc_test)'
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
        help='Path to Alpamayo 1.5 model (default: ckpts/Alpamayo-1.5-10B)'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable auto-visualization after inference'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Output video FPS (default: 10)'
    )
    args = parser.parse_args()
    
    # Parse time ranges
    try:
        time_ranges = [parse_time_range(tr) for tr in args.time_range]
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print(f"Clip IDs: {len(args.clip_id)} clips")
    print(f"Time ranges: {len(time_ranges)} ranges")
    for idx, (start, end) in enumerate(time_ranges):
        print(f"  Range {idx + 1}: {start}s - {end}s")
    
    # Determine output directory
    script_dir = Path(__file__).parent.parent
    data_dir = Path(script_dir / "data" / "PhysicalAI-Autonomous-Vehicles-base")
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "labels" / "coc_test"
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Resume: {args.resume}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model and dataset interface
    print("\nLoading model...")
    model = Alpamayo1_5.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("✓ Model loaded")
    
    # Process each clip
    all_results = []
    success_clips = 0
    
    for clip_idx, clip_id in enumerate(args.clip_id):
        print(f"\n{'='*80}")
        print(f"Processing clip {clip_idx + 1}/{len(args.clip_id)}: {clip_id}")
        print(f"{'='*80}")
        
        # Create a safe filename (replace special characters)
        safe_clip_id = clip_id.replace('/', '_').replace('\\', '_')
        output_json_path = output_dir / f"{safe_clip_id}.coc.json"
        
        # Process this clip
        result = process_single_clip(
            clip_id=clip_id,
            time_ranges=time_ranges,
            model=model,
            processor=processor,
            output_json_path=output_json_path,
            resume=args.resume,
            visualize=not args.no_visualize,
            fps=args.fps,
        )
        
        if result is not None:
            success_clips += 1
            all_results.append(result)
    
    # Generate summary statistics
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        total_clips = len(args.clip_id)
        total_inference = sum(r["statistics"]["total_inference_count"] for r in all_results)
        success_inference = sum(r["statistics"]["success_inference_count"] for r in all_results)
        
        print(f"Total clips processed: {success_clips}/{total_clips}")
        print(f"Total inference calls: {total_inference}")
        print(f"Successful inferences: {success_inference}")
        print(f"Failed inferences: {total_inference - success_inference}")
        print(f"Overall success rate: {success_inference / total_inference * 100:.1f}%" if total_inference > 0 else "0%")
    
    print("\n" + "="*80)
    print("✓ CoC custom time range inference complete!")
    print(f"Output saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
