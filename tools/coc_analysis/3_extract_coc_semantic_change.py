#!/usr/bin/env python3
"""
Extract clip_id and coc_list from coc_combined.json files,
call LLM to identify semantic change frames, and collect results into chunked json files.
Each chunk directory's results are saved as a single file named coc_change.chunk_XXXX.json.

Usage Examples:
    # 1. Dry run: 只处理 chunk 0 的前 2 个 clips（调试用）
    python tools/3_extract_coc_semantic_change.py --dry-run --chunks 0 --max-clips-per-chunk 2

    # 2. 处理指定范围的 chunks
    python tools/3_extract_coc_semantic_change.py --chunks 0-2

    # 3. 处理多个不连续的 chunks，每个最多 10 个 clips
    python tools/3_extract_coc_semantic_change.py --chunks 0,5,10 --max-clips-per-chunk 10

    # 4. 处理所有 chunks（默认行为）
    python tools/3_extract_coc_semantic_change.py
"""

import json
import os
import glob
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import argparse
import re
from dotenv import load_dotenv


def build_prompt(coc_list: dict) -> str:
    """Build prompt for LLM"""
    # Sort by frame_idx
    sorted_frames = sorted(coc_list.keys(), key=lambda x: int(x))
    
    # Only use middle 21 frames [5:26]
    sorted_frames = sorted_frames[5:26]

    # Build coc text
    coc_text = ""
    for frame_idx in sorted_frames:
        coc_entry = coc_list[frame_idx]
        if isinstance(coc_entry, list) and len(coc_entry) > 0:
            if isinstance(coc_entry[0], list) and len(coc_entry[0]) > 0:
                description = coc_entry[0][0]
            else:
                description = str(coc_entry[0])
        else:
            description = str(coc_entry)
        coc_text += f"Frame {frame_idx}: {description}\n"

    prompt = f"""You are analyzing a sequence of {len(sorted_frames)} consecutive driving behavior descriptions (CoC).

Your goal is to identify the **FIRST critical moment** where the driving maneuver fundamentally changes.

### Analysis Rules (Strict Priority):
1. **Focus on ACTION (The Verb):** 
   - The CoC structure is typically "[Action] because [Reason]". Focus primarily on the **[Action]** part.
   - A change is valid ONLY if the core driving maneuver changes (e.g., "Keep Going" → "Decelerate", "Steer Left" → "Steer Right").
   
2. **Ignore Minor Fluctuations:**
   - **Do NOT flag** minor rewording of the reason (e.g., "red light" → "traffic signal").
   - **Do NOT flag** slight intensity changes (e.g., "decelerate gently" → "decelerate moderately").

### Output Format:
- Return the **EXACT frame index** (e.g., 81) where this fundamental Action change occurs.
- If the action remains semantically consistent throughout the entire sequence (only reasons or objects change slightly), return **-1**.
- **ONLY return the number or -1**. No explanation.

### Sequence Data:
{coc_text}"""

    return prompt


def call_llm(prompt: str, client: OpenAI, model: str = "default") -> int:
    """Call LLM and parse result"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at detecting semantic changes in autonomous vehicle behavior descriptions. Analyze the frame sequence and find where the behavior meaning first changes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1024,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        result_text = response.choices[0].message.content.strip()

        # Extract answer from last line
        lines = result_text.strip().split('\n')

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            numbers = re.findall(r'-?\d+', line)
            if numbers:
                return int(numbers[-1])

        return -1

    except Exception as e:
        print(f"  Error: LLM call failed: {e}")
        return -2


def process_single_file(file_path: str, client: OpenAI, dry_run: bool = False) -> dict:
    """Process single coc_combined.json file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    clip_id = data.get("clip_id", "unknown")
    result = {
        "clip_id": clip_id,
        "coc_lists": {}
    }

    coc_list_keys = [k for k in data.keys() if k != "clip_id"]

    if not coc_list_keys:
        print(f"Warning: No coc_list found in {file_path}")
        return result

    for coc_key in tqdm(coc_list_keys, desc=f"    [{clip_id[:8]}...]", leave=False):
        coc_list_data = data[coc_key]
        coc_list = coc_list_data.get("coc_list", {})

        if not coc_list:
            print(f"    Warning: No coc_list in coc_key {coc_key}")
            result["coc_lists"][coc_key] = -1
            continue

        if dry_run:
            print(f"    [Dry Run] coc_key: {coc_key}, frames: {len(coc_list)}")
            result["coc_lists"][coc_key] = -999
            continue

        prompt = build_prompt(coc_list)
        change_frame = call_llm(prompt, client)
        result["coc_lists"][coc_key] = change_frame

        status = f"change_frame: {change_frame}" if change_frame >= -1 else "failed"

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
    default_input_dir = os.path.join(data_dir, "labels", "coc_combined")
    default_output_dir = os.path.join(data_dir, "labels", "coc_change")
    
    parser = argparse.ArgumentParser(description="Extract COC data and call LLM to detect semantic changes")
    parser.add_argument("--input-dir", type=str, default=default_input_dir,
                        help="Input directory path")
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help="Output directory path for chunked results")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="OpenAI API Key (default EMPTY)")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8000/v1",
                        help="OpenAI API Base URL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test mode, do not call LLM")
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

    print(f"Data directory: {data_dir}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.chunks:
        print(f"Specified chunks: {args.chunks}")
    if args.max_clips_per_chunk:
        print(f"Max clips per chunk: {args.max_clips_per_chunk}")

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    input_path = Path(args.input_dir)
    # Find all chunk directories
    all_chunk_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("coc_combined.chunk_")])

    print(f"Found {len(all_chunk_dirs)} total chunk directories")

    if not all_chunk_dirs:
        print("Error: No chunk directories found, please check input directory")
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
        chunk_dirs = []
        for d in all_chunk_dirs:
            chunk_number = int(d.name.replace("coc_combined.chunk_", ""))
            if chunk_number in target_chunk_numbers:
                chunk_dirs.append(d)
        # Sort by chunk number
        chunk_dirs = sorted(chunk_dirs, key=lambda d: int(d.name.replace("coc_combined.chunk_", "")))
        
        print(f"Filtered to {len(chunk_dirs)} specified chunk(s)")
        if not chunk_dirs:
            print(f"Error: None of the specified chunks exist. Available chunks: "
                  f"{[d.name for d in all_chunk_dirs[:5]]}...")
            return
    else:
        chunk_dirs = all_chunk_dirs

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each chunk directory
    total_clips = 0
    for chunk_dir in tqdm(chunk_dirs, desc="Processing chunks"):
        chunk_name = chunk_dir.name  # e.g., "coc_combined.chunk_0000"
        # Extract chunk number, e.g., "0000"
        chunk_number = chunk_name.replace("coc_combined.chunk_", "")
        
        print(f"\n{'='*60}")
        print(f"Processing chunk: {chunk_name}")
        print(f"{'='*60}")
        
        # Find all coc_combined.json files in this chunk
        json_files = sorted(chunk_dir.glob("*.coc_combined.json"))
        print(f"Found {len(json_files)} files in this chunk")
        
        # Limit clips per chunk if specified
        if args.max_clips_per_chunk:
            original_count = len(json_files)
            json_files = json_files[:args.max_clips_per_chunk]
            print(f"Limited to {len(json_files)} clips (from {original_count})")

        chunk_results = []

        for file_path in tqdm(json_files, desc=f"  [{chunk_name}] files"):
            try:
                result = process_single_file(str(file_path), client, args.dry_run)
                chunk_results.append(result)
            except Exception as e:
                print(f"  Error: Failed to process {file_path}: {e}")
                chunk_results.append({
                    "clip_id": "error",
                    "file": str(file_path),
                    "error": str(e)
                })

        # Save chunk result
        chunk_filename = f"coc_change.chunk_{chunk_number}.json"
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
    print(f"Total chunks processed: {len(chunk_dirs)}")
    print(f"Total clips processed: {total_clips}")


if __name__ == "__main__":
    main()
