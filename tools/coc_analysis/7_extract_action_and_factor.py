#!/usr/bin/env python3
"""
Extract action and factor from COC sentences using LLM.
Reads coc_train_change.json (or similar), extracts actions and factors from coc_lists,
and saves to a new JSON file with action_lists and factor_lists.

Usage Examples:
    # 1. Dry run: 只处理前 5 条（调试用）
    python tools/7_extract_action_and_factor.py --dry-run --max-items 5

    # 2. 处理指定范围
    python tools/7_extract_action_and_factor.py --start-idx 0 --end-idx 100

    # 3. 默认处理全部
    python tools/7_extract_action_and_factor.py
"""

import json
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import argparse
import re
from dotenv import load_dotenv


def build_prompt(coc_sentence: str) -> str:
    """Build prompt for LLM to extract actions and factors"""

    prompt = f"""You are an expert at analyzing autonomous vehicle behavior descriptions.

Given a Chain-of-Thought (CoC) sentence describing driving behavior, extract:
1. **Actions**: The driving maneuvers/behaviors (verbs or verb phrases)
2. **Factors**: The reasons/causes that trigger those actions (typically after "since", "because", "due to")

### Examples:

Input: "Keep lane since the lane is clear ahead"
{{
  "actions": ["Keep lane"],
  "factors": ["the lane is clear ahead"]
}}

Input: "Slow down to create a gap for merging left because our right lane is ending ahead and a vehicle is occupying the left lane alongside"
{{
  "actions": ["Slow down to create a gap for merging left"],
  "factors": ["our right lane is ending ahead", "a vehicle is occupying the left lane alongside"]
}}

Input: "Yield to the pedestrian since they are crossing at the crosswalk ahead"
{{
  "actions": ["Yield to the pedestrian"],
  "factors": ["they are crossing at the crosswalk ahead"]
}}

Input: "Adjust speed and keep safe distance due to the lead vehicle ahead"
{{
  "actions": ["Adjust speed", "keep safe distance"],
  "factors": ["the lead vehicle ahead"]
}}

Input: "Change lane to the left to avoid the bike ahead and stop at the pedestrian crossing for the red traffic light."
{{
  "actions": ["Change lane to the left", "stop at the pedestrian crossing"],
  "factors": ["avoid the bike ahead", "the red traffic light"]
}}

### Rules:
- Extract ALL actions mentioned (there may be multiple)
- Extract ALL factors/reasons mentioned (there may be multiple)
- Keep the exact phrasing from the original text
- If no clear factor is present, return an empty list for factors
- Actions should be verb phrases (e.g., "Yield", "Slow down", "stop at the pedestrian crossing")
- Factors should describe the situation/object that caused the action

### Input:
"{coc_sentence}"

### Output (JSON format only):
"""
    return prompt


def call_llm(prompt: str, client: OpenAI, model: str = "default") -> dict:
    """Call LLM and parse result"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at extracting actions and factors from autonomous vehicle behavior descriptions. Return ONLY valid JSON with 'actions' and 'factors' as keys, both being lists of strings."},
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
        
        # Remove thinking tags if present
        result_text = re.sub(r'<[^>]*>', '', result_text)
        
        # Try to extract JSON from the response
        # Strategy 1: Try to find complete JSON object
        json_match = re.search(r'\{[^{}]*"actions"\s*:\s*\[[^\]]*\]\s*,\s*"factors"\s*:\s*\[[^\]]*\][^{}]*\}', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                result_dict = json.loads(json_str)
                return result_dict
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Try to extract actions and factors separately using regex
        actions_match = re.findall(r'"actions"\s*:\s*\[([^\]]*)\]', result_text, re.DOTALL)
        factors_match = re.findall(r'"factors"\s*:\s*\[([^\]]*)\]', result_text, re.DOTALL)
        
        actions = []
        factors = []
        
        if actions_match:
            # Extract strings from the actions array
            actions = re.findall(r'"([^"]*)"', actions_match[0])
        
        if factors_match:
            # Extract strings from the factors array
            factors = re.findall(r'"([^"]*)"', factors_match[0])
        
        if actions or factors:
            return {"actions": actions, "factors": factors}
        
        # If all else fails, return empty result
        print(f"  Warning: Could not parse LLM response: {result_text[:150]}")
        return {"actions": [], "factors": []}

    except Exception as e:
        print(f"  Error: LLM call failed: {e}")
        return {"actions": [], "factors": []}


def process_single_coc(coc_text: str, client: OpenAI, dry_run: bool = False, model: str = "default") -> dict:
    """Process single COC sentence"""
    if dry_run:
        print(f"    [Dry Run] Processing: {coc_text[:80]}...")
        return {"actions": ["<dry_run>"], "factors": ["<dry_run>"]}
    
    prompt = build_prompt(coc_text)
    result = call_llm(prompt, client, model)
    return result


def main():
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")

    # Default paths
    default_data_dir = "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles"
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", default_data_dir)

    parser = argparse.ArgumentParser(description="Extract actions and factors from COC sentences using LLM")
    parser.add_argument("--input-file", type=str, 
                        default=os.path.join(data_dir, "labels", "coc_train", "coc_train_change.json"),
                        help="Input JSON file path")
    parser.add_argument("--output-file", type=str,
                        default=os.path.join(data_dir, "labels", "coc_train", "coc_train_action_factor.json"),
                        help="Output JSON file path")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="OpenAI API Key (default EMPTY)")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8000/v1",
                        help="OpenAI API Base URL")
    parser.add_argument("--model", type=str, default="ckpts/Qwen3.5-9B",
                        help="Model name to use")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test mode, do not call LLM")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Max number of items to process (for debugging)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index for processing")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End index for processing")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Save checkpoint every N items (default: 100)")

    args = parser.parse_args()

    print(f"Data directory: {data_dir}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    if args.max_items:
        print(f"Max items to process: {args.max_items}")
    if args.start_idx > 0 or args.end_idx:
        print(f"Processing range: {args.start_idx} to {args.end_idx or 'end'}")

    # Load input data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    print(f"Loaded {len(input_data)} items from input file")

    # Slice data if needed
    start_idx = args.start_idx
    end_idx = args.end_idx or len(input_data)
    if args.max_items:
        end_idx = min(end_idx, start_idx + args.max_items)
    
    data_to_process = input_data[start_idx:end_idx]
    print(f"Processing {len(data_to_process)} items (from index {start_idx} to {end_idx})")

    # Initialize client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    # Process data
    output_data = []
    success_count = 0
    error_count = 0

    for idx, item in enumerate(tqdm(data_to_process, desc="Processing")):
        actual_idx = start_idx + idx
        
        # Create output item with same base structure
        output_item = {
            "chunk_id": item.get("chunk_id", "unknown"),
            "clip_id": item.get("clip_id", "unknown"),
            "ori_key_frame_long_action": item.get("ori_key_frame_long_action", ""),
            "ori_key_frame_lat_action": item.get("ori_key_frame_lat_action", ""),
            "coc_lists": item.get("coc_lists", {}),
            "action_lists": {},
            "factor_lists": {}
        }

        # Process each COC sentence
        coc_lists = item.get("coc_lists", {})
        
        for frame_idx, coc_text in coc_lists.items():
            try:
                result = process_single_coc(coc_text, client, args.dry_run, args.model)
                
                # Store extracted actions and factors
                output_item["action_lists"][frame_idx] = result.get("actions", [])
                output_item["factor_lists"][frame_idx] = result.get("factors", [])
                
                success_count += 1
            except Exception as e:
                print(f"\n  Error processing frame {frame_idx} in item {actual_idx}: {e}")
                output_item["action_lists"][frame_idx] = []
                output_item["factor_lists"][frame_idx] = []
                error_count += 1

        output_data.append(output_item)

        # Save checkpoint periodically
        if (idx + 1) % args.batch_size == 0:
            checkpoint_path = args.output_file.replace('.json', f'_checkpoint_{actual_idx}.json')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n  Checkpoint saved: {checkpoint_path}")

    # Save final output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Output saved to: {args.output_file}")
    print(f"Total items processed: {len(output_data)}")
    print(f"Successful extractions: {success_count}")
    print(f"Errors: {error_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
