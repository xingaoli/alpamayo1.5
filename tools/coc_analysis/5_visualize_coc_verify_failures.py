#!/usr/bin/env python3
"""
Visualize CoC verification failures.

This script:
1. Reads coc_verify results to find frames marked as False
2. Filters out invalid change frames (negative or >=200)
3. Decodes the corresponding image from video
4. Draws the CoC text on the image
5. Saves to labels/coc_verify_vis/chunk_XXXX/ directory

Usage Examples:
    # 1. Visualize chunk 0 failures
    python tools/5_visualize_coc_verify_failures.py --chunks 0

    # 2. Visualize multiple chunks
    python tools/5_visualize_coc_verify_failures.py --chunks 0-2
"""

import json
import os
import zipfile
import io
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import pandas as pd
import physical_ai_av.video as video
from dotenv import load_dotenv


def decode_frame_at_timestamp(data_dir: str, clip_id: str, chunk_number: int, 
                               timestamp_us: int, camera_feature: str = "camera_front_wide_120fov") -> np.ndarray:
    """
    Extract a frame from video at the specified timestamp.
    """
    camera_zip_path = os.path.join(
        data_dir, "camera", camera_feature, 
        f"{camera_feature}.chunk_{chunk_number:04d}.zip"
    )
    
    if not os.path.exists(camera_zip_path):
        raise FileNotFoundError(f"Camera zip not found: {camera_zip_path}")
    
    with zipfile.ZipFile(camera_zip_path, 'r') as zf:
        video_data = io.BytesIO(zf.read(f"{clip_id}.{camera_feature}.mp4"))
        
        timestamps_df = pd.read_parquet(
            io.BytesIO(zf.read(f"{clip_id}.{camera_feature}.timestamps.parquet"))
        )
        frame_timestamps = timestamps_df["timestamp"].values
        
        reader = video.SeekVideoReader(
            video_data=video_data,
            timestamps=frame_timestamps,
        )
        
        target_timestamps = np.array([timestamp_us], dtype=np.int64)
        frames, _ = reader.decode_images_from_timestamps(target_timestamps)
        
        if frames.shape[0] == 0:
            raise ValueError(f"No frame decoded for timestamp {timestamp_us} us")
        
        return frames[0]


def draw_text_on_image(image: np.ndarray, text: str, position: tuple = (10, 30),
                       font_scale: float = 0.7, thickness: int = 2,
                       text_color: tuple = (255, 255, 255),
                       bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Draw text with background on image.
    
    Args:
        image: Image array (H, W, 3) in RGB format
        text: Text to draw
        position: (x, y) position
        font_scale: Font scale
        thickness: Text thickness
        text_color: RGB text color
        bg_color: RGB background color
    
    Returns:
        Image with text drawn on it
    """
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color_bgr = tuple(text_color[::-1])
    bg_color_bgr = tuple(bg_color[::-1])
    
    # Split long text into multiple lines
    max_chars_per_line = 60
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_chars_per_line:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw each line
    x, y = position
    line_height = 30
    
    for i, line in enumerate(lines):
        y_pos = y + i * line_height
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            line, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            img_bgr,
            (x - 5, y_pos - text_height - 5),
            (x + text_width + 5, y_pos + baseline + 5),
            bg_color_bgr,
            -1
        )
        
        # Draw text
        cv2.putText(
            img_bgr,
            line,
            (x, y_pos),
            font,
            font_scale,
            text_color_bgr,
            thickness
        )
    
    # Convert back to RGB
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def process_chunk(chunk_number: int, coc_verify_path: Path, coc_change_path: Path,
                  data_dir: str, output_dir: Path):
    """
    Process a single chunk and visualize failures.
    """
    chunk_name = f"chunk_{chunk_number:04d}"
    
    # Load verification results
    verify_file = coc_verify_path / f"coc_verify.{chunk_name}.json"
    if not verify_file.exists():
        print(f"  Warning: Verification file not found: {verify_file}")
        return
    
    with open(verify_file, 'r', encoding='utf-8') as f:
        verify_data = json.load(f)
    
    # Load original coc_change data
    change_file = coc_change_path / f"coc_change.{chunk_name}.json"
    if not change_file.exists():
        print(f"  Warning: Change file not found: {change_file}")
        return
    
    with open(change_file, 'r', encoding='utf-8') as f:
        change_data = json.load(f)
    
    # Create output directory for this chunk
    chunk_output_dir = output_dir / chunk_name
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {chunk_name}...")
    
    # Count failures and statistics
    failure_count = 0
    total_valid_frames = 0  # 有效帧(0-199),会被送入VLM的
    vlm_false_count = 0
    
    for clip_verify, clip_change in tqdm(zip(verify_data, change_data), 
                                         desc=f"  [{chunk_name}] clips", 
                                         total=len(verify_data)):
        clip_id = clip_verify['clip_id']
        
        for key_frame_str, verify_value in clip_verify['coc_lists'].items():
            # Get original value from coc_change
            original_value = clip_change['coc_lists'].get(key_frame_str, -999)
            
            # Skip invalid frames (negative or >=200)
            if not isinstance(original_value, int) or original_value < 0 or original_value >= 200:
                continue
            
            # 这是一个有效帧，会被送入VLM
            total_valid_frames += 1
            
            # Count VLM failures
            if verify_value == False:
                vlm_false_count += 1
            
            # Only visualize failures
            if verify_value != False:
                continue
            
            failure_count += 1
            
            # Get CoC text from coc_combined
            coc_combined_file = Path(data_dir) / "labels" / "coc_combined" / f"coc_combined.{chunk_name}" / f"{clip_id}.coc_combined.json"
            if not coc_combined_file.exists():
                continue
            
            with open(coc_combined_file, 'r', encoding='utf-8') as f:
                coc_data = json.load(f)
            
            if key_frame_str not in coc_data:
                continue
            
            coc_entry = coc_data[key_frame_str]
            coc_list_data = coc_entry.get("coc_list", {})
            
            change_frame_str = str(original_value)
            if change_frame_str not in coc_list_data:
                continue
            
            coc_text_raw = coc_list_data[change_frame_str]
            
            # Extract CoC text
            if isinstance(coc_text_raw, list) and len(coc_text_raw) > 0:
                if isinstance(coc_text_raw[0], list) and len(coc_text_raw[0]) > 0:
                    coc_text = coc_text_raw[0][0]
                else:
                    coc_text = str(coc_text_raw[0])
            else:
                coc_text = str(coc_text_raw)
            
            # Decode image
            try:
                timestamp_us = int(original_value / 10 * 1_000_000)
                image = decode_frame_at_timestamp(data_dir, clip_id, chunk_number, timestamp_us)
                
                # Draw CoC text on image
                image_with_text = draw_text_on_image(
                    image, 
                    f"[FAIL] {coc_text}",
                    position=(20, 50),
                    font_scale=0.6,
                    thickness=2,
                    text_color=(255, 255, 255),
                    bg_color=(200, 50, 50)  # Reddish background for failures
                )
                
                # Save image
                output_filename = f"{clip_id}_key{key_frame_str}_frame{original_value}.jpg"
                output_path = chunk_output_dir / output_filename
                
                # Convert RGB to BGR for saving
                image_bgr = cv2.cvtColor(image_with_text, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
            except Exception as e:
                print(f"    Error processing {clip_id} key_frame {key_frame_str}: {e}")
                continue
    
    # Print statistics
    print(f"\n  {'='*50}")
    print(f"  统计信息 (已过滤无效帧):")
    print(f"  总有效帧数 (0-199): {total_valid_frames}")
    print(f"  VLM判定为False: {vlm_false_count}")
    
    if total_valid_frames > 0:
        false_rate = vlm_false_count / total_valid_frames * 100
        true_count = total_valid_frames - vlm_false_count
        true_rate = true_count / total_valid_frames * 100
        print(f"  VLM判定为True: {true_count} ({true_rate:.1f}%)")
        print(f"  VLM判定为False: {vlm_false_count} ({false_rate:.1f}%)")
    
    print(f"  保存失败案例可视化: {failure_count} 张")
    print(f"  保存路径: {chunk_output_dir}")
    print(f"  {'='*50}")


def main():
    # Load environment variables
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
    
    # Get data directory
    default_data_dir = "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles"
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", default_data_dir)
    
    # Default paths
    default_coc_verify_dir = os.path.join(data_dir, "labels", "coc_verify")
    default_coc_change_dir = os.path.join(data_dir, "labels", "coc_change")
    default_output_dir = os.path.join(data_dir, "labels", "coc_verify_vis")
    
    parser = argparse.ArgumentParser(description="Visualize CoC verification failures")
    parser.add_argument("--coc-verify-dir", type=str, default=default_coc_verify_dir,
                        help="Input directory with coc_verify files")
    parser.add_argument("--coc-change-dir", type=str, default=default_coc_change_dir,
                        help="Input directory with coc_change files")
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help="Output directory for visualization results")
    parser.add_argument("--data-dir", type=str, default=data_dir,
                        help="Base data directory")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Specify chunk numbers to process. Supports: "
                             "comma-separated (e.g., '0,5,10'), "
                             "range (e.g., '0-5'), "
                             "or mixed (e.g., '0,3-5,8'). "
                             "If not specified, process all chunks.")
    
    args = parser.parse_args()
    
    print(f"Data directory: {args.data_dir}")
    print(f"CoC verify directory: {args.coc_verify_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.chunks:
        print(f"Specified chunks: {args.chunks}")
    
    coc_verify_path = Path(args.coc_verify_dir)
    coc_change_path = Path(args.coc_change_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all verify chunk files
    all_chunk_files = sorted(coc_verify_path.glob("coc_verify.chunk_*.json"))
    
    print(f"\nFound {len(all_chunk_files)} total coc_verify chunk files")
    
    if not all_chunk_files:
        print("Error: No coc_verify chunk files found")
        return
    
    # Parse chunk specification
    def parse_chunk_spec(chunk_spec: str) -> set:
        chunk_numbers = set()
        parts = chunk_spec.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = part.split('-', 1)
                start_num = int(start.strip())
                end_num = int(end.strip())
                chunk_numbers.update(range(start_num, end_num + 1))
            else:
                chunk_numbers.add(int(part))
        return chunk_numbers
    
    # Filter chunks
    if args.chunks:
        target_chunk_numbers = parse_chunk_spec(args.chunks)
        chunk_files = []
        for f in all_chunk_files:
            chunk_num_str = f.name.replace("coc_verify.chunk_", "").replace(".json", "")
            chunk_number = int(chunk_num_str)
            if chunk_number in target_chunk_numbers:
                chunk_files.append(f)
        
        chunk_files = sorted(chunk_files, key=lambda f: int(f.name.replace("coc_verify.chunk_", "").replace(".json", "")))
        
        print(f"Filtered to {len(chunk_files)} specified chunk(s)")
        if not chunk_files:
            print(f"Error: None of the specified chunks exist")
            return
    else:
        chunk_files = all_chunk_files
    
    # Process each chunk
    for chunk_file in chunk_files:
        chunk_number = int(chunk_file.name.replace("coc_verify.chunk_", "").replace(".json", ""))
        
        process_chunk(
            chunk_number,
            coc_verify_path,
            coc_change_path,
            args.data_dir,
            output_path
        )
    
    print(f"\n{'='*60}")
    print(f"All done! Visualizations saved to: {output_path}")


if __name__ == "__main__":
    main()
