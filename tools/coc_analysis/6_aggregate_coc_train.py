#!/usr/bin/env python3
"""
Aggregate CoC data from coc_change (and optionally coc_verify) into training format.

This script supports two modes:
1. Mode 1 (verify-based): Extract CoC for frames marked as True in coc_verify results
2. Mode 2 (change-based): Extract CoC for all valid change frames (0-199) from coc_change

Output format follows coc_train.json structure:
{
  "chunk_id": "chunk_0000",
  "clip_id": "...",
  "ori_key_frame_long_action": "...",
  "ori_key_frame_lat_action": "...",
  "coc_lists": {
    "frame_idx": "coc text",
    ...
  }
}

Usage:
    # Mode 1 (default): Extract verified True frames from coc_verify
    python tools/6_aggregate_coc_train.py --mode verify

    # Mode 2: Extract all valid frames from coc_change
    python tools/6_aggregate_coc_train.py --mode change
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from dotenv import load_dotenv


def load_coc_combined(data_dir: str, chunk_name: str, clip_id: str) -> dict:
    """Load coc_combined data for a specific clip."""
    coc_combined_dir = Path(data_dir) / "labels" / "coc_combined" / f"coc_combined.{chunk_name}"
    coc_combined_file = coc_combined_dir / f"{clip_id}.coc_combined.json"
    
    if not coc_combined_file.exists():
        return None
    
    with open(coc_combined_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_coc_text(coc_list_data: dict, frame_idx: int) -> str:
    """Extract CoC text from coc_list data for a specific frame index."""
    frame_str = str(frame_idx)
    
    if frame_str not in coc_list_data:
        return None
    
    coc_text_raw = coc_list_data[frame_str]
    
    # Handle nested structure
    if isinstance(coc_text_raw, list) and len(coc_text_raw) > 0:
        if isinstance(coc_text_raw[0], list) and len(coc_text_raw[0]) > 0:
            return coc_text_raw[0][0]
        else:
            return str(coc_text_raw[0])
    else:
        return str(coc_text_raw)


def aggregate_mode_verify(data_dir: str, output_path: Path):
    """
    Mode 1: Aggregate CoC for frames marked as True in coc_verify.
    Scans all chunks under coc_verify directory.
    """
    coc_verify_dir = Path(data_dir) / "labels" / "coc_verify"
    coc_change_dir = Path(data_dir) / "labels" / "coc_change"
    
    # Find all verify chunk files
    verify_chunk_files = sorted(coc_verify_dir.glob("coc_verify.chunk_*.json"))
    
    if not verify_chunk_files:
        print(f"Error: No coc_verify chunk files found in {coc_verify_dir}")
        return
    
    print(f"Found {len(verify_chunk_files)} coc_verify chunk files")
    
    all_results = []
    total_clips = 0
    total_coc_extracted = 0
    
    for verify_chunk_file in tqdm(verify_chunk_files, desc="Processing chunks"):
        chunk_number = int(verify_chunk_file.name.replace("coc_verify.chunk_", "").replace(".json", ""))
        chunk_name = f"chunk_{chunk_number:04d}"
        
        # Load verification results
        with open(verify_chunk_file, 'r', encoding='utf-8') as f:
            verify_data = json.load(f)
        
        # Load original coc_change data (to get key_frame metadata)
        change_chunk_file = coc_change_dir / f"coc_change.{chunk_name}.json"
        if not change_chunk_file.exists():
            print(f"  Warning: coc_change file not found: {change_chunk_file}")
            continue
        
        with open(change_chunk_file, 'r', encoding='utf-8') as f:
            change_data = json.load(f)
        
        for clip_verify, clip_change in zip(verify_data, change_data):
            clip_id = clip_verify['clip_id']
            
            # Load coc_combined to get metadata and CoC text
            coc_data = load_coc_combined(data_dir, chunk_name, clip_id)
            if coc_data is None:
                continue
            
            # Build coc_lists with only True frames
            coc_lists = {}
            
            for key_frame_str, verify_value in clip_verify['coc_lists'].items():
                # Only extract frames marked as True
                if verify_value != True:
                    continue
                
                # Get the original change frame value (this is the actual frame_idx)
                original_value = clip_change['coc_lists'].get(key_frame_str, -999)
                
                # Skip if not a valid frame
                if not isinstance(original_value, int) or original_value < 0 or original_value >= 200:
                    continue
                
                # Extract CoC text for this frame
                if key_frame_str not in coc_data:
                    continue
                
                coc_entry = coc_data[key_frame_str]
                coc_list_data = coc_entry.get("coc_list", {})
                
                coc_text = extract_coc_text(coc_list_data, original_value)
                if coc_text is None:
                    continue
                
                # Use the actual frame_idx (original_value) as key
                coc_lists[str(original_value)] = coc_text
                total_coc_extracted += 1
            
            # Only add clips that have at least one CoC
            if coc_lists:
                # Get original key frame metadata (use first key frame)
                first_key_frame = list(clip_verify['coc_lists'].keys())[0]
                if first_key_frame in coc_data:
                    ori_key_frame_data = coc_data[first_key_frame]
                    long_action = ori_key_frame_data.get("long_action", "Unknown")
                    lat_action = ori_key_frame_data.get("lat_action", "Unknown")
                else:
                    long_action = "Unknown"
                    lat_action = "Unknown"
                
                result_entry = {
                    "chunk_id": chunk_name,
                    "clip_id": clip_id,
                    "ori_key_frame_long_action": long_action,
                    "ori_key_frame_lat_action": lat_action,
                    "coc_lists": coc_lists
                }
                
                all_results.append(result_entry)
                total_clips += 1
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Mode: Verify-based (True frames only)")
    print(f"Total clips with valid CoC: {total_clips}")
    print(f"Total CoC entries extracted: {total_coc_extracted}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}")


def aggregate_mode_change(data_dir: str, output_path: Path):
    """
    Mode 2: Aggregate CoC for all valid change frames (0-199) from coc_change.
    Scans all chunks under coc_change directory.
    """
    coc_change_dir = Path(data_dir) / "labels" / "coc_change"
    
    # Find all change chunk files
    change_chunk_files = sorted(coc_change_dir.glob("coc_change.chunk_*.json"))
    
    if not change_chunk_files:
        print(f"Error: No coc_change chunk files found in {coc_change_dir}")
        return
    
    print(f"Found {len(change_chunk_files)} coc_change chunk files")
    
    all_results = []
    total_clips = 0
    total_coc_extracted = 0
    
    for change_chunk_file in tqdm(change_chunk_files, desc="Processing chunks"):
        chunk_number = int(change_chunk_file.name.replace("coc_change.chunk_", "").replace(".json", ""))
        chunk_name = f"chunk_{chunk_number:04d}"
        
        # Load coc_change data
        with open(change_chunk_file, 'r', encoding='utf-8') as f:
            change_data = json.load(f)
        
        for clip_data in change_data:
            clip_id = clip_data['clip_id']
            
            # Load coc_combined to get metadata and CoC text
            coc_data = load_coc_combined(data_dir, chunk_name, clip_id)
            if coc_data is None:
                continue
            
            # Build coc_lists with all valid frames
            coc_lists = {}
            
            for key_frame_str, change_frame_idx in clip_data['coc_lists'].items():
                # Only extract valid frames (0-199)
                if not isinstance(change_frame_idx, int) or change_frame_idx < 0 or change_frame_idx >= 200:
                    continue
                
                # Extract CoC text for this frame
                if key_frame_str not in coc_data:
                    continue
                
                coc_entry = coc_data[key_frame_str]
                coc_list_data = coc_entry.get("coc_list", {})
                
                coc_text = extract_coc_text(coc_list_data, change_frame_idx)
                if coc_text is None:
                    continue
                
                # Use the actual frame_idx as key
                coc_lists[str(change_frame_idx)] = coc_text
                total_coc_extracted += 1
            
            # Only add clips that have at least one CoC
            if coc_lists:
                # Get original key frame metadata (use first key frame)
                first_key_frame = list(clip_data['coc_lists'].keys())[0]
                if first_key_frame in coc_data:
                    ori_key_frame_data = coc_data[first_key_frame]
                    long_action = ori_key_frame_data.get("long_action", "Unknown")
                    lat_action = ori_key_frame_data.get("lat_action", "Unknown")
                else:
                    long_action = "Unknown"
                    lat_action = "Unknown"
                
                result_entry = {
                    "chunk_id": chunk_name,
                    "clip_id": clip_id,
                    "ori_key_frame_long_action": long_action,
                    "ori_key_frame_lat_action": lat_action,
                    "coc_lists": coc_lists
                }
                
                all_results.append(result_entry)
                total_clips += 1
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Mode: Change-based (all valid frames 0-199)")
    print(f"Total clips with valid CoC: {total_clips}")
    print(f"Total CoC entries extracted: {total_coc_extracted}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}")


def main():
    # Load environment variables
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
    
    # Get data directory
    default_data_dir = "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles"
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", default_data_dir)
    
    parser = argparse.ArgumentParser(description="Aggregate CoC data into training format")
    parser.add_argument("--data-dir", type=str, default=data_dir,
                        help="Base data directory")
    parser.add_argument("--mode", type=str, choices=['verify', 'change'], default='verify',
                        help="Aggregation mode: 'verify' (True frames only, default) or 'change' (all valid frames)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: coc_train_{mode}.json in project root)")
    
    args = parser.parse_args()
    
    print(f"Data directory: {args.data_dir}")
    print(f"Mode: {args.mode}")
    
    # Set default output path
    if args.output is None:
        output_dir = Path(args.data_dir) / "labels" / "coc_train"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"coc_train_{args.mode}.json")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output file: {output_path}")
    
    # Run appropriate mode
    if args.mode == 'verify':
        aggregate_mode_verify(args.data_dir, output_path)
    elif args.mode == 'change':
        aggregate_mode_change(args.data_dir, output_path)


if __name__ == "__main__":
    main()
