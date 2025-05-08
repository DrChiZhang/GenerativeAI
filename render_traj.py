import os
import cv2
import json

import warnings

from argparse import ArgumentParser
warnings.filterwarnings(action='ignore')

import imageio

import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

from arguments import GSParams
from gaussian_renderer import render
from scene import LayerGaussian

from utils.depth_utils import colorize
from scene.cameras import MiniCam_GS

    
from argparse import ArgumentParser, Namespace

parser = ArgumentParser(description="Step 0 Parameters")
parser.add_argument("--input_dir", default="outputs", type=str)
parser.add_argument("--save_dir", default="outputs", type=str)
args = parser.parse_args()

print('Render from {}'.format(args.input_dir))  

# --- Load camera poses from transforms.json ---
json_path = os.path.join(args.input_dir, 'transforms.json')
with open(json_path, 'r') as f:
    traj_data = json.load(f)

# --- Get intrinsics from JSON (first frame, usually representative for all frames) ---
first_frame = traj_data['frames'][0]
width = first_frame.get('w', 1024)
height = first_frame.get('h', 1024)

if 'fl_x' in first_frame:
    fl_x = first_frame['fl_x']
else:
    # fallback - try 'focal_length' or default to something reasonable
    fl_x = width / (2 * math.tan(math.radians(90) / 2))

fovx = 2 * math.atan(width / (2 * fl_x))
if 'fl_y' in first_frame:
    fl_y = first_frame['fl_y']
    fovy = 2 * math.atan(height / (2 * fl_y))
else:
    fovy = height * fovx / width  # fallback, square pixels assumption

views = []
opt = GSParams()
bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

for frame in traj_data['frames']:
    pose = np.array(frame['transform_matrix'])  # (4,4)
    """
    If you get a black result or “spinning scene”, 
    you may need pose = np.linalg.inv(np.array(frame['transform_matrix'])) depending on your conventions.
    """
    # Use per-frame w/h if available:
    w = frame.get('w', width)
    h = frame.get('h', height)
    # Use per-frame focal if available
    fx = frame.get('fl_x', fl_x)
    fy = frame.get('fl_y', fl_y if 'fl_y' in locals() else fx)
    local_fovx = 2 * math.atan(w / (2 * fx))
    local_fovy = 2 * math.atan(h / (2 * fy))
    cur_cam = MiniCam_GS(pose, w, h, local_fovx, local_fovy)
    views.append(cur_cam)

framelist = []
depthlist = []
dmin, dmax = 1e8, -1e8

with torch.no_grad():
    gaussians = LayerGaussian(opt.sh_degree)
    gaussians.load_ply(os.path.join(args.input_dir, 'gsplat.ply'))
    
    iterable_render = views

    for view in tqdm(iterable_render):
        results = render(view, gaussians, opt, background)
        frame, depth = results['render'], results['depth']
        framelist.append(
            np.round(frame.permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
        depth = -(depth * (depth > 0)).detach().cpu().numpy()
        dmin_local = depth.min().item()
        dmax_local = depth.max().item()
        if dmin_local < dmin:
            dmin = dmin_local
        if dmax_local > dmax:
            dmax = dmax_local
        depthlist.append(depth)


depthlist = [colorize(depth) for depth in depthlist]

os.makedirs(f"{args.save_dir}/renders", exist_ok=True)
id = 0
for frame in framelist:
    id+=1
    frame_pil = Image.fromarray(frame)
    frame_pil.save(f"{args.save_dir}/renders/{id}.png")
    
print('Start Writing Videos...')
imageio.mimwrite(f"{args.save_dir}/render.mp4", framelist, fps=30, quality=8)
imageio.mimwrite(f"{args.save_dir}/render_depth.mp4", depthlist, fps=30, quality=8)
print('End Writing Videos...')
