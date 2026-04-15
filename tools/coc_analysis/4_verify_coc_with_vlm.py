#!/usr/bin/env python3
"""
Verify CoC change frames using VLM (Vision Language Model).

This script:
1. Reads coc_change.chunk_XXXX.json to find valid change frames for each clip
2. Gets actual CoC text from coc_combined.chunk_XXXX/{clip_id}.coc_combined.json
3. Extracts corresponding frame from camera_front_wide_120fov video
4. Calls VLM API to verify if the CoC is reasonable for that frame
5. Writes results to labels/coc_verify directory

Usage Examples:
    # 1. Dry run: 只处理 chunk 0 的前 2 个 clips（调试用）
    python tools/4_verify_coc_with_vlm.py --dry-run --chunks 0 --max-clips-per-chunk 2

    # 2. 处理指定范围的 chunks
    python tools/4_verify_coc_with_vlm.py --chunks 0-2

    # 3. 处理所有 chunks（默认行为）
    python tools/4_verify_coc_with_vlm.py
"""

import json
import os
import glob
import zipfile
import io
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import pandas as pd
import base64
import physical_ai_av.video as video
from dotenv import load_dotenv


def decode_frame_at_timestamp(data_dir: str, clip_id: str, chunk_number: int, 
                               timestamp_us: int, camera_feature: str = "camera_front_wide_120fov") -> np.ndarray:
    """
    Extract a frame from video at the specified timestamp using physical_ai_av.video.SeekVideoReader.
    
    Args:
        data_dir: Base data directory
        clip_id: The clip ID
        chunk_number: Chunk number
        timestamp_us: Timestamp in microseconds
        camera_feature: Camera feature name (default: camera_front_wide_120fov)
    
    Returns:
        Image as numpy array (H, W, 3) in RGB format
    """
    # Build video zip path
    camera_zip_path = os.path.join(
        data_dir, "camera", camera_feature, 
        f"{camera_feature}.chunk_{chunk_number:04d}.zip"
    )
    
    if not os.path.exists(camera_zip_path):
        raise FileNotFoundError(f"Camera zip not found: {camera_zip_path}")
    
    with zipfile.ZipFile(camera_zip_path, 'r') as zf:
        # Read video data
        video_data = io.BytesIO(zf.read(f"{clip_id}.{camera_feature}.mp4"))
        
        # Read timestamps
        timestamps_df = pd.read_parquet(
            io.BytesIO(zf.read(f"{clip_id}.{camera_feature}.timestamps.parquet"))
        )
        frame_timestamps = timestamps_df["timestamp"].values
        
        # Use SeekVideoReader to decode at the requested timestamp
        reader = video.SeekVideoReader(
            video_data=video_data,
            timestamps=frame_timestamps,
        )
        
        # Decode frame at the specific timestamp
        # decode_images_from_timestamps expects an array of timestamps
        target_timestamps = np.array([timestamp_us], dtype=np.int64)
        frames, _ = reader.decode_images_from_timestamps(target_timestamps)
        
        # frames shape: (1, H, W, 3)
        if frames.shape[0] == 0:
            raise ValueError(f"No frame decoded for timestamp {timestamp_us} us")
        
        # Return first frame (H, W, 3) in RGB format
        return frames[0]


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode numpy image array to base64 string for API.
    
    Args:
        image: Image array (H, W, 3) in RGB format
    
    Returns:
        Base64 encoded JPEG image string
    """
    # Convert RGB to BGR for OpenCV encoding
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64


def build_vlm_prompt(coc_text: str) -> str:
    """
    Build prompt for VLM to verify CoC correctness.
    Focus on verifying that objects/targets mentioned in CoC actually exist in the image.
    
    Args:
        coc_text: The CoC description text
    
    Returns:
        Prompt string for VLM
    """
    prompt = f"""You are verifying if a driving behavior description (CoC) matches the current camera image.

**Important Context**: This CoC was predicted using historical information, but we only have ONE current frame to verify it. Therefore, focus on checking if the **objects and targets mentioned in the CoC actually exist** in the image.

### Verification Rules:
1. **Check Objects/Targets**: Verify if the specific objects mentioned (traffic lights, pedestrians, vehicles, obstacles, etc.) are actually present in the image.
   - Example: If CoC says "accelerate because the light is green" → Check if there IS a green traffic light visible
   - Example: If CoC says "decelerate due to pedestrian crossing" → Check if there IS a pedestrian in the image
   - Example: If CoC says "stop for the red car ahead" → Check if there IS a red car in front

2. **Ignore Dynamic Behaviors**: Since we only have one frame, DO NOT try to verify motion/dynamics (e.g., whether something is "moving", "accelerating", "crossing"). Just check if the object exists.
   - ✅ Correct approach: "Is there a pedestrian?" 
   - ❌ Wrong approach: "Is the pedestrian crossing the road?" (can't tell from one frame)

3. **Return False ONLY for Clear Mismatches**:
   - CoC mentions "green light" but image shows red light → **False**
   - CoC mentions "pedestrian" but no pedestrian visible → **False**
   - CoC mentions "stopped vehicle" but road is clear → **False**

4. **Return True if Objects Match**:
   - The mentioned objects/targets are visible in the image (even if you can't verify the exact action)
   - Minor differences in description are acceptable as long as the core objects match

### CoC Description:
{coc_text}

### Question:
Do the key objects/targets mentioned in this CoC description actually exist in the current image?

### Output Format:
Return ONLY "True" if the objects/targets match, or "False" if there is a clear mismatch.
No explanation, just True or False."""

    return prompt


def call_vlm_api(image_base64: str, coc_text: str, client: OpenAI, model: str = "default") -> bool:
    """
    Call VLM API to verify if CoC matches the image.
    Uses OpenAI-compatible API with vision support.

    Args:
        image_base64: Base64 encoded image
        coc_text: CoC description text
        client: OpenAI client
        model: Model name to use

    Returns:
        True if CoC is correct, False otherwise
    """
    prompt = build_vlm_prompt(coc_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=100,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse True/False from response
        if result_text.lower() == "true":
            return True
        elif result_text.lower() == "false":
            return False
        else:
            print(f"  Warning: Unexpected VLM response: {result_text}")
            return False
    
    except Exception as e:
        print(f"  Error: VLM API call failed: {e}")
        return False


def process_single_clip(
    clip_data: dict,
    coc_combined_dir: Path,
    data_dir: str,
    chunk_number: int,
    client: OpenAI,
    dry_run: bool = False,
    model: str = "default"
) -> dict:
    """
    Process a single clip: verify all its change frames.
    
    Args:
        clip_data: Dict with clip_id and coc_lists from coc_change
        coc_combined_dir: Path to coc_combined.chunk_XXXX directory
        data_dir: Base data directory
        chunk_number: Chunk number
        client: OpenAI client
        dry_run: If True, skip actual VLM calls
    
    Returns:
        Updated clip_data with verification results
    """
    clip_id = clip_data["clip_id"]
    coc_lists = clip_data["coc_lists"]
    
    result = {
        "clip_id": clip_id,
        "coc_lists": {}
    }
    
    # Find the coc_combined file for this clip
    coc_combined_file = coc_combined_dir / f"{clip_id}.coc_combined.json"
    if not coc_combined_file.exists():
        print(f"  Warning: CoC combined file not found: {coc_combined_file}")
        # Return all False for all keys
        for key in coc_lists:
            result["coc_lists"][key] = False
        return result
    
    # Load coc_combined data
    with open(coc_combined_file, 'r', encoding='utf-8') as f:
        coc_data = json.load(f)
    
    # Process each key_frame in coc_lists
    for key_frame_str, change_frame_idx in tqdm(coc_lists.items(), desc=f"    [{clip_id[:8]}...]", leave=False):
        key_frame = int(key_frame_str)
        
        # Check if this is a valid change frame (non-negative and < 200)
        is_valid_change = isinstance(change_frame_idx, int) and 0 <= change_frame_idx < 200
        
        if not is_valid_change:
            # Not a valid change frame, mark as False
            result["coc_lists"][key_frame_str] = False
            continue
        
        if dry_run:
            # change_frame_idx / 10 = seconds
            timestamp_us = int(change_frame_idx / 10 * 1_000_000)
            print(f"    [Dry Run] key_frame: {key_frame}, change_frame: {change_frame_idx}, timestamp: {timestamp_us} us")
            result["coc_lists"][key_frame_str] = True  # Placeholder for dry run
            continue
        
        # Get the CoC text for this change frame
        if key_frame_str not in coc_data:
            print(f"    Warning: key_frame {key_frame} not found in coc_combined")
            result["coc_lists"][key_frame_str] = False
            continue
        
        coc_entry = coc_data[key_frame_str]
        coc_list_data = coc_entry.get("coc_list", {})
        
        # Get the CoC text for the change_frame_idx
        change_frame_str = str(change_frame_idx)
        if change_frame_str not in coc_list_data:
            print(f"    Warning: change_frame {change_frame_idx} not found in coc_list")
            result["coc_lists"][key_frame_str] = False
            continue
        
        coc_text_raw = coc_list_data[change_frame_str]
        
        # Extract actual CoC text (handle nested structure)
        if isinstance(coc_text_raw, list) and len(coc_text_raw) > 0:
            if isinstance(coc_text_raw[0], list) and len(coc_text_raw[0]) > 0:
                coc_text = coc_text_raw[0][0]
            else:
                coc_text = str(coc_text_raw[0])
        else:
            coc_text = str(coc_text_raw)
        
        # Convert change_frame_idx to timestamp in microseconds
        # change_frame_idx / 10 = seconds -> microseconds
        timestamp_us = int(change_frame_idx / 10 * 1_000_000)
        
        try:
            # Decode frame at the specific timestamp
            image = decode_frame_at_timestamp(data_dir, clip_id, chunk_number, timestamp_us)
            image_base64 = encode_image_to_base64(image)
            
            # Call VLM API
            is_correct = call_vlm_api(image_base64, coc_text, client)
            result["coc_lists"][key_frame_str] = is_correct
            
            status = "✓" if is_correct else "✗"
            print(f"    [{status}] key_frame: {key_frame}, change_frame: {change_frame_idx}, "
                  f"timestamp: {timestamp_us} us, VLM: {is_correct}")
        
        except Exception as e:
            print(f"    Error: Failed to process frame {change_frame_idx} (timestamp {timestamp_us} us): {e}")
            result["coc_lists"][key_frame_str] = False
    
    return result


def main():
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
    
    # Get data directory from environment variable or use default
    default_data_dir = "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles"
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", default_data_dir)
    
    # Default paths based on data directory
    default_coc_change_dir = os.path.join(data_dir, "labels", "coc_change")
    default_coc_combined_dir = os.path.join(data_dir, "labels", "coc_combined")
    default_output_dir = os.path.join(data_dir, "labels", "coc_verify")
    
    parser = argparse.ArgumentParser(description="Verify CoC change frames using VLM")
    parser.add_argument("--coc-change-dir", type=str, default=default_coc_change_dir,
                        help="Input directory with coc_change files")
    parser.add_argument("--coc-combined-dir", type=str, default=default_coc_combined_dir,
                        help="Input directory with coc_combined files")
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help="Output directory for verification results")
    parser.add_argument("--data-dir", type=str, default=data_dir,
                        help="Base data directory")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="OpenAI API Key (default EMPTY)")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8000/v1",
                        help="OpenAI API Base URL")
    parser.add_argument("--model", type=str, default="ckpts/Qwen3.5-9B",
                        help="Model name to use")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test mode, do not call VLM")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Specify chunk numbers to process. Supports: "
                             "comma-separated (e.g., '0,5,10'), "
                             "range (e.g., '0-5'), "
                             "or mixed (e.g., '0,3-5,8'). "
                             "Chunk numbers can be with or without leading zeros. "
                             "If not specified, process all chunks.")
    parser.add_argument("--max-clips-per-chunk", type=int, default=None,
                        help="Max number of clips to process per chunk (for debugging)")
    
    args = parser.parse_args()
    
    print(f"Data directory: {args.data_dir}")
    print(f"CoC change directory: {args.coc_change_dir}")
    print(f"CoC combined directory: {args.coc_combined_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.chunks:
        print(f"Specified chunks: {args.chunks}")
    if args.max_clips_per_chunk:
        print(f"Max clips per chunk: {args.max_clips_per_chunk}")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # Find all coc_change chunk files
    coc_change_path = Path(args.coc_change_dir)
    all_chunk_files = sorted(coc_change_path.glob("coc_change.chunk_*.json"))
    
    print(f"\nFound {len(all_chunk_files)} total coc_change chunk files")
    
    if not all_chunk_files:
        print("Error: No coc_change chunk files found, please check input directory")
        return
    
    # Parse chunk specification
    def parse_chunk_spec(chunk_spec: str) -> set:
        """Parse chunk specification string into a set of chunk numbers (as integers)"""
        chunk_numbers = set()
        parts = chunk_spec.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                # Range specification
                start, end = part.split('-', 1)
                start_num = int(start.strip())
                end_num = int(end.strip())
                chunk_numbers.update(range(start_num, end_num + 1))
            else:
                # Single chunk number
                chunk_numbers.add(int(part))
        return chunk_numbers
    
    # Filter chunks if --chunks is specified
    if args.chunks:
        target_chunk_numbers = parse_chunk_spec(args.chunks)
        chunk_files = []
        for f in all_chunk_files:
            # Extract chunk number from filename like "coc_change.chunk_0000.json"
            chunk_num_str = f.name.replace("coc_change.chunk_", "").replace(".json", "")
            chunk_number = int(chunk_num_str)
            if chunk_number in target_chunk_numbers:
                chunk_files.append(f)
        # Sort by chunk number
        chunk_files = sorted(chunk_files, key=lambda f: int(f.name.replace("coc_change.chunk_", "").replace(".json", "")))
        
        print(f"Filtered to {len(chunk_files)} specified chunk(s)")
        if not chunk_files:
            print(f"Error: None of the specified chunks exist. Available chunks: "
                  f"{[f.name for f in all_chunk_files[:5]]}...")
            return
    else:
        chunk_files = all_chunk_files
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each chunk file
    total_clips = 0
    for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
        # Extract chunk number
        chunk_number = int(chunk_file.name.replace("coc_change.chunk_", "").replace(".json", ""))
        chunk_name = f"chunk_{chunk_number:04d}"
        
        print(f"\n{'='*60}")
        print(f"Processing {chunk_name}")
        print(f"{'='*60}")
        
        # Load coc_change data
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
        
        print(f"Found {len(chunk_data)} clips in this chunk")
        
        # Limit clips per chunk if specified
        if args.max_clips_per_chunk:
            original_count = len(chunk_data)
            chunk_data = chunk_data[:args.max_clips_per_chunk]
            print(f"Limited to {len(chunk_data)} clips (from {original_count})")
        
        # Find corresponding coc_combined directory
        coc_combined_dir = Path(args.coc_combined_dir) / f"coc_combined.{chunk_name}"
        if not coc_combined_dir.exists():
            print(f"  Warning: CoC combined directory not found: {coc_combined_dir}")
            continue
        
        chunk_results = []
        
        for clip_data in tqdm(chunk_data, desc=f"  [{chunk_name}] clips"):
            try:
                result = process_single_clip(
                    clip_data,
                    coc_combined_dir,
                    args.data_dir,
                    chunk_number,
                    client,
                    args.dry_run,
                    args.model
                )
                chunk_results.append(result)
            except Exception as e:
                print(f"  Error: Failed to process clip {clip_data.get('clip_id', 'unknown')}: {e}")
                chunk_results.append({
                    "clip_id": clip_data.get("clip_id", "error"),
                    "coc_lists": {},
                    "error": str(e)
                })
        
        # Save chunk result
        chunk_filename = f"coc_verify.chunk_{chunk_number:04d}.json"
        chunk_output_path = output_path / chunk_filename
        
        with open(chunk_output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nChunk {chunk_name} saved to: {chunk_output_path}")
        print(f"Clips processed in this chunk: {len(chunk_results)}")
        
        success_count = sum(1 for r in chunk_results if "error" not in r)
        error_count = len(chunk_results) - success_count
        print(f"Success: {success_count}, Failed: {error_count}")
        
        total_clips += len(chunk_results)
    
    print(f"\n{'='*60}")
    print(f"All done! Results saved to: {output_path}")
    print(f"Total chunks processed: {len(chunk_files)}")
    print(f"Total clips processed: {total_clips}")


if __name__ == "__main__":
    main()
