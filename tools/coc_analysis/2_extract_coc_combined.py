#!/usr/bin/env python3
"""
Extract and combine CoC (Chain-of-Causation) results into combined format.

This script reads all .coc.json files from the coc directory, extracts
the 31 inference timestamps for each keyframe, and writes them to the
coc_combined directory in the combined format.

Usage:
    python3 extract_coc_combined.py
    python3 extract_coc_combined.py --chunks chunk_0000 chunk_0001
    python3 extract_coc_combined.py --chunks all
"""
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def extract_coc_combined(input_json_path: Path) -> Dict:
    """
    Extract and transform a single .coc.json file to the combined format.
    
    Args:
        input_json_path: Path to the input .coc.json file
        
    Returns:
        Dict in the coc_combined format with frame_index as key
    """
    with open(input_json_path, 'r') as f:
        coc_data = json.load(f)
    
    # Build the combined format with frame_index as key
    combined_data = {
        "clip_id": coc_data["clip_id"],
    }
    
    # Process each keyframe
    for keyframe in coc_data["keyframe_coc_results"]:
        frame_index = str(keyframe["frame_index"])
        inference_timestamps = keyframe.get("inference_timestamps", [])
        
        # Build coc_list with timestamp index as key and cot as value
        coc_list = {}
        for timestamp_data in inference_timestamps:
            # Convert timestamp_sec to index (e.g., 6.0 -> 60, 4.4 -> 44)
            timestamp_sec = timestamp_data.get("timestamp_sec")
            if timestamp_sec is not None:
                index = int(round(timestamp_sec * 10))
                # Only keep the cot field
                coc_list[str(index)] = timestamp_data.get("cot")
        
        # Add to combined data with frame_index as key
        combined_data[frame_index] = {
            "keyframe_index": keyframe["keyframe_index"],
            "long_action": keyframe.get("long_action", "Unknown"),
            "lat_action": keyframe.get("lat_action", "Unknown"),
            "coc_list": coc_list
        }
    
    return combined_data


def process_all_chunks(
    coc_dir: Path,
    output_dir: Path,
    chunk_ids: List[str] = None
):
    """
    Process all or specified chunks of CoC results.
    Output is organized by chunk: output_dir/coc_combined.chunk_0000/
    
    Args:
        coc_dir: Directory containing coc.chunk_* directories
        output_dir: Output directory for coc_combined results
        chunk_ids: List of chunk IDs to process, or None for all chunks
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all chunk directories
    if chunk_ids is None or (len(chunk_ids) == 1 and chunk_ids[0] == "all"):
        chunk_dirs = sorted(coc_dir.glob("coc.chunk_*"))
    else:
        chunk_dirs = [coc_dir / f"coc.{chunk_id}" for chunk_id in chunk_ids]
        chunk_dirs = [d for d in chunk_dirs if d.exists()]
    
    if not chunk_dirs:
        print(f"No coc chunk directories found in {coc_dir}")
        return
    
    print(f"Found {len(chunk_dirs)} chunk(s) to process")
    print("="*80)
    
    total_files = 0
    success_files = 0
    
    # Process each chunk
    for chunk_dir in chunk_dirs:
        chunk_name = chunk_dir.name.replace('coc.', '')
        print(f"\nProcessing {chunk_name}...")
        
        # Find all .coc.json files in this chunk
        coc_files = sorted(chunk_dir.glob("*.coc.json"))
        
        if not coc_files:
            print(f"  No .coc.json files found in {chunk_dir}")
            continue
        
        print(f"  Found {len(coc_files)} file(s)")
        total_files += len(coc_files)
        
        # Create chunk-specific output directory
        chunk_output_dir = output_dir / f"coc_combined.{chunk_name}"
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        for coc_file in tqdm(coc_files, desc=f"  {chunk_name}"):
            try:
                # Extract and transform
                combined_data = extract_coc_combined(coc_file)
                
                # Determine output path
                output_filename = f"{combined_data['clip_id']}.coc_combined.json"
                output_path = chunk_output_dir / output_filename
                
                # Write to file
                with open(output_path, 'w') as f:
                    json.dump(combined_data, f, indent=2, cls=NumpyEncoder)
                
                success_files += 1
                
            except Exception as e:
                print(f"\n  ✗ Failed to process {coc_file.name}: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"✓ Extraction complete!")
    print(f"  Total files processed: {total_files}")
    print(f"  Successful: {success_files}")
    print(f"  Failed: {total_files - success_files}")
    print(f"  Output directory: {output_dir}")
    print("="*80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract and combine CoC results into combined format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all chunks
  python3 extract_coc_combined.py

  # Process specific chunks
  python3 extract_coc_combined.py --chunks chunk_0000 chunk_0001
        """
    )
    parser.add_argument(
        '--chunks',
        nargs='+',
        default=None,
        help='Specific chunk IDs to process (e.g., chunk_0000), or "all" to process all'
    )
    parser.add_argument(
        '--coc-dir',
        type=str,
        default=None,
        help='Directory containing coc results (default: data/PhysicalAI-Autonomous-Vehicles/labels/coc)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for combined results (default: data/PhysicalAI-Autonomous-Vehicles/labels/coc_combined)'
    )
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent.parent.parent
    data_dir = script_dir / "data" / "PhysicalAI-Autonomous-Vehicles"
    
    coc_dir = Path(args.coc_dir) if args.coc_dir else data_dir / "labels" / "coc"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "labels" / "coc_combined"
    
    print(f"CoC input directory: {coc_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not coc_dir.exists():
        print(f"Error: CoC directory {coc_dir} does not exist!")
        return
    
    # Process all chunks
    process_all_chunks(
        coc_dir=coc_dir,
        output_dir=output_dir,
        chunk_ids=args.chunks
    )


if __name__ == "__main__":
    main()
