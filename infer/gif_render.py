# Open Source Model Licensed under the Apache License Version 2.0 
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited 
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

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
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os, sys
sys.path.insert(0, f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

from svrm.ldm.vis_util import render
from infer.utils import seed_everything, timing_decorator

class GifRenderer():
    '''
        render frame(s) of mesh using pytorch3d
    '''
    def __init__(self, device="cuda:0"):
        self.device = device

    @timing_decorator("gif render")
    def __call__(
        self, 
        obj_filename, 
        elev=0, 
        azim=0, 
        resolution=512, 
        gif_dst_path='', 
        n_views=120, 
        fps=30, 
        rgb=True
    ):
        render(
            obj_filename,
            elev=elev, 
            azim=azim, 
            resolution=resolution, 
            gif_dst_path=gif_dst_path, 
            n_views=n_views, 
            fps=fps, 
            device=self.device, 
            rgb=rgb
        )

if __name__ == "__main__":
    import argparse
    
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--mesh_path", type=str, required=True)
        parser.add_argument("--output_gif_path", type=str, required=True)
        parser.add_argument("--device", default="cuda:0", type=str)
        return parser.parse_args()
        
    args = get_args()

    gif_renderer = GifRenderer(device=args.device)

    gif_renderer(
        args.mesh_path,
        gif_dst_path = args.output_gif_path
    )
