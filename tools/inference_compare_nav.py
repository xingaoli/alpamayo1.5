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
os.chdir('/home/xingao/code/Alpamayo1.5')
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

nav_text = "Turn left in 10m"

print(f"Navigation instruction: {nav_text}")
print(f"Counterfactual (swapped): {nav_utils.swap_direction(nav_text)}")

with torch.autocast("cuda", dtype=torch.bfloat16):
    nav_result = nav_utils.compare_nav_conditions(
        model=model,
        processor=processor,
        data=data_scene1,
        nav_text=nav_text,
        num_traj_samples=16,
        top_p=0.98,
        temperature=0.6,
        max_generation_length=256,
        return_extra=True,
        additional_nav_inference_kwargs={
            "diffusion_kwargs": {
                # The temperature for controlling the initial noise. Note that using
                # temperature < 1.0 will result in a more stable sampling with less diversity.
                "temperature": 0.6,
            }
        },
    )

print(f"Trajectories per condition: {nav_result.pred_with_nav.shape[2]}")

camera_grid = make_camera_grid(
    data_scene1["image_frames"], camera_indices=data_scene1["camera_indices"]
)

fig = plot_bev_comparison(
    pred_with_nav=nav_result.pred_with_nav,
    pred_no_nav=nav_result.pred_no_nav,
    pred_counterfactual=nav_result.pred_counterfactual,
    nav_text=nav_result.nav_text,
    nav_text_swapped=nav_result.nav_text_swapped,
    gt_future_xyz=data_scene1.get("ego_future_xyz"),
    camera_images=camera_grid,
    title=f'Navigation: "{nav_text}"',
)
plt.show()