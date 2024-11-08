# Open Source Model Licensed under the Apache License Version 2.0 and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been 
# modified by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT 
# except for the third-party components listed below. 
# Hunyuan 3D does not impose any additional limitations beyond what is outlined 
# in the repsective licenses of these third-party components. 
# Users must comply with all terms and conditions of original licenses of these third-party 
# components and must ensure that the usage of the third party components adheres to 
# all relevant laws and regulations. 

# For avoidance of doubts, Hunyuan 3D means the large language models and 
# their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.l

import os
import torch
from PIL import Image
import argparse
from datetime import datetime
from tqdm import tqdm
from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer

# ---- Define Functions ----

def get_args():
    parser = argparse.ArgumentParser(description="Pipeline for generating 3D models from text or images.")
    
    # General arguments
    parser.add_argument("--use_lite", default=False, action="store_true", help="Use the lite version of models (saves memory).")
    parser.add_argument("--save_folder", default="./outputs/test/", type=str, help="Folder to save output files.")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device for running the model (e.g., cuda:0 or cpu).")
    
    # Model paths and configuration
    parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str, help="Path to the SVRM config file.")
    parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str, help="Path to the SVRM checkpoint file.")
    parser.add_argument("--text2image_path", default="weights/hunyuanDiT", type=str, help="Path to the text-to-image pre-trained model.")
    
    # Inputs
    parser.add_argument("--text_prompt", default="", type=str, help="Text prompt for image generation.")
    parser.add_argument("--image_prompt", default="", type=str, help="Image prompt for generating views.")
    
    # Randomness and steps control
    parser.add_argument("--t2i_seed", default=0, type=int, help="Seed for text-to-image generation.")
    parser.add_argument("--t2i_steps", default=25, type=int, help="Steps for generating the image from text.")
    parser.add_argument("--gen_seed", default=0, type=int, help="Seed for generating 3D mesh from views.")
    parser.add_argument("--gen_steps", default=50, type=int, help="Steps for generating 3D mesh.")
    
    # Mesh generation settings
    parser.add_argument("--max_faces_num", default=80000, type=int, help="Max number of faces for the generated 3D mesh.")
    parser.add_argument("--do_texture_mapping", default=False, action="store_true", help="Apply texture mapping to the 3D mesh.")
    parser.add_argument("--do_render", default=False, action="store_true", help="Render a rotating gif of the 3D model.")

    # Memory and output settings
    parser.add_argument("--save_memory", default=False, action="store_true", help="Save memory by optimizing model inference.")
    parser.add_argument("--save_intermediate", default=False, action="store_true", help="Save intermediate steps for debugging.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Enable verbose output during processing.")
    
    # Custom output filename
    parser.add_argument("--output_name", type=str, default="output", help="Base name for output files (e.g., output_mesh.obj, output.gif).")
    
    return parser.parse_args()

def check_paths(args):
    """Validate file paths before starting the pipeline."""
    assert args.text_prompt or args.image_prompt, "You must provide either a text prompt or an image prompt."
    assert not (args.text_prompt and args.image_prompt), "You cannot provide both text and image prompts."
    
    # Check model and config paths
    if not os.path.exists(args.mv23d_cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {args.mv23d_cfg_path}")
    if not os.path.exists(args.mv23d_ckt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.mv23d_ckt_path}")
    if not os.path.exists(args.text2image_path):
        raise FileNotFoundError(f"Text-to-image model not found: {args.text2image_path}")

    os.makedirs(args.save_folder, exist_ok=True)

    # Ensure save folder exists
    save_subfolders = ["images", "models", "renders"]
    for subfolder in save_subfolders:
        os.makedirs(os.path.join(args.save_folder, subfolder), exist_ok=True)

def save_image(image, filename, folder):
    """Save images to the specified folder."""
    image.save(os.path.join(folder, filename))

def main():
    args = get_args()

    # Check and validate paths
    check_paths(args)

    # Initialize models
    rembg_model = Removebg()
    image_to_views_model = Image2Views(device=args.device, use_lite=args.use_lite)
    views_to_mesh_model = Views2Mesh(args.mv23d_cfg_path, args.mv23d_ckt_path, args.device, use_lite=args.use_lite)
    
    if args.text_prompt:
        text_to_image_model = Text2Image(pretrain=args.text2image_path, device=args.device, save_memory=args.save_memory)
    if args.do_render:
        gif_renderer = GifRenderer(device=args.device)

    # ---- Stage 1: Text-to-Image Generation ----
    if args.text_prompt:
        if args.verbose:
            print("Generating image from text prompt...")
        res_rgb_pil = text_to_image_model(args.text_prompt, seed=args.t2i_seed, steps=args.t2i_steps)
        save_image(res_rgb_pil, f"{args.output_name}_img.jpg", os.path.join(args.save_folder, "images"))

    elif args.image_prompt:
        if args.verbose:
            print("Loading provided image...")
        res_rgb_pil = Image.open(args.image_prompt)

    # ---- Stage 2: Background Removal ----
    if args.verbose:
        print("Removing background from image...")
    res_rgba_pil = rembg_model(res_rgb_pil)
    save_image(res_rgba_pil, f"{args.output_name}_img_nobg.png", os.path.join(args.save_folder, "images"))

    # ---- Stage 3: Image to Views Generation ----
    if args.verbose:
        print("Generating views from image...")
    views_grid_pil, cond_img = image_to_views_model(res_rgba_pil, seed=args.gen_seed, steps=args.gen_steps)
    save_image(views_grid_pil, f"{args.output_name}_views.jpg", os.path.join(args.save_folder, "images"))

    # ---- Stage 4: Views to Mesh ----
    if args.verbose:
        print("Generating 3D mesh from views...")
    views_to_mesh_model(views_grid_pil, cond_img, seed=args.gen_seed, target_face_count=args.max_faces_num,
                        save_folder=os.path.join(args.save_folder, "models"), do_texture_mapping=args.do_texture_mapping)

    # ---- Stage 5: Render GIF ----
    if args.do_render:
        if args.verbose:
            print("Rendering gif of 3D model...")
        gif_renderer(os.path.join(args.save_folder, "models", f"{args.output_name}_mesh.obj"),
                     gif_dst_path=os.path.join(args.save_folder, "renders", f"{args.output_name}.gif"))

    if args.verbose:
        print(f"Process complete. Output saved in {args.save_folder}.")

if __name__ == "__main__":
    main()
