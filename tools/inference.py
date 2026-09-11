# # Alpamayo 1.5: Navigation-Conditioned Trajectory Prediction

# Alpamayo 1.5 introduces the ability to **condition trajectory predictions on natural language navigation instructions**. Given a driving scene and an instruction like *"Turn right in 30m"*, the model produces trajectories that follow the specified route intention.
#
# This notebook demonstrates:
# 1. **Basic trajectory prediction** with chain-of-causation reasoning
# 2. **Navigation-conditioned sampling** -- how the same scene yields different trajectory distributions depending on the navigation instruction
# 3. **Manual control** -- how to integrate navigation conditioning into your own pipeline
#
# We use data from the NVIDIA [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).

import os
os.chdir('/home/xingao/code/NVlabs-alpamayo/alpamayo1.5')
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

import json
import mediapy as mp
import matplotlib.pyplot as plt

import torch
torch.cuda.manual_seed_all(42)

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset_local import load_physical_aiavdataset_local
from alpamayo1_5 import helper, nav_utils
from alpamayo1_5.viz_utils import make_camera_grid, plot_bev_comparison

model = Alpamayo1_5.from_pretrained("ckpts/Alpamayo-1.5-10B", dtype=torch.bfloat16).to('cuda')

processor = helper.get_processor(model.tokenizer)
clip_id = "ef4264ed-0fd2-4a64-9831-87e1aae28407"
t0_us = 4_000_000

data_scene1 = load_physical_aiavdataset_local(clip_id, t0_us=t0_us)

messages = helper.create_message(
    data_scene1["image_frames"].flatten(0, 1),
    camera_indices=data_scene1["camera_indices"],
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = helper.to_device(
    {
        "tokenized_data": inputs,
        "ego_history_xyz": data_scene1["ego_history_xyz"],
        "ego_history_rot": data_scene1["ego_history_rot"],
    },
    "cuda",
)


with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )

print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])