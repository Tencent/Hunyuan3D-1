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

import os
import warnings
import argparse
import gradio as gr
from glob import glob
import shutil
import torch
import numpy as np
from PIL import Image
from einops import rearrange

from infer import seed_everything, save_gif
from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer

warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--use_lite", default=False, action="store_true")
parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str)
parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str)
parser.add_argument("--text2image_path", default="weights/hunyuanDiT", type=str)
parser.add_argument("--save_memory", default=False, action="store_true")
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

################################################################
# initial setting
################################################################

CONST_PORT = 8080
CONST_MAX_QUEUE = 1
CONST_SERVER = '0.0.0.0'

CONST_HEADER = '''
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/tencent/Hunyuan3D-1' target='_blank'><b>Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D
Generationr</b></a></h2>
Code: <a href='https://github.com/tencent/Hunyuan3D-1' target='_blank'>GitHub</a>. Techenical report: <a href='https://arxiv.org/abs/placeholder' target='_blank'>ArXiv</a>.

❗️❗️❗️**Important Notes:**
- By default, our demo can export a .obj mesh with vertex colors or a .glb mesh.
- If you select "texture mapping," it will export a .obj mesh with a texture map or a .glb mesh.
- If you select "render GIF," it will export a GIF image rendering of the .glb file.
- If the result is unsatisfactory, please try a different seed value (Default: 0).
'''

CONST_CITATION = r"""
If HunYuan3D-1 is helpful, please help to ⭐ the <a href='https://github.com/tencent/Hunyuan3D-1' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/tencent/Hunyuan3D-1?style=social)](https://github.com/tencent/Hunyuan3D-1)
---
📝 **Citation**
If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@misc{yang2024tencent,
    title={Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation},
    author={Xianghui Yang and Huiwen Shi and Bowen Zhang and Fan Yang and Jiacheng Wang and Hongxu Zhao and Xinhai Liu and Xinzhou Wang and Qingxiang Lin and Jiaao Yu and Lifu Wang and Zhuo Chen and Sicong Liu and Yuhong Liu and Yong Yang and Di Wang and Jie Jiang and Chunchao Guo},
    year={2024},
    eprint={2411.02293},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
"""

################################################################
# prepare text examples and image examples
################################################################

def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./demos/example_*.png'))

def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list  = list()
    for line in open('./demos/example_list.txt'):
        txt_list.append(line.strip())
    return txt_list

example_is = get_example_img_list()
example_ts = get_example_txt_list()

################################################################
# initial models
################################################################

worker_xbg = Removebg()
print(f"loading {args.text2image_path}")
worker_t2i = Text2Image(
    pretrain = args.text2image_path, 
    device = args.device, 
    save_memory = args.save_memory
)
worker_i2v = Image2Views(
    use_lite = args.use_lite, 
    device = args.device,
    save_memory = args.save_memory
)
worker_v23 = Views2Mesh(
    args.mv23d_cfg_path, 
    args.mv23d_ckt_path, 
    use_lite = args.use_lite, 
    device = args.device,
    save_memory = args.save_memory
)
worker_gif = GifRenderer(args.device)

def stage_0_t2i(text, image, seed, step):
    os.makedirs('./outputs/app_output', exist_ok=True)
    exists = set(int(_) for _ in os.listdir('./outputs/app_output') if not _.startswith("."))
    if len(exists) == 30: shutil.rmtree(f"./outputs/app_output/0");cur_id = 0
    else:                 cur_id = min(set(range(30)) - exists)
    if os.path.exists(f"./outputs/app_output/{(cur_id + 1) % 30}"):
        shutil.rmtree(f"./outputs/app_output/{(cur_id + 1) % 30}")
    save_folder = f'./outputs/app_output/{cur_id}'
    os.makedirs(save_folder, exist_ok=True)

    dst = save_folder + '/img.png'
    
    if not text:
        if image is None: 
            return dst, save_folder
            raise gr.Error("Upload image or provide text ...")
        image.save(dst)
        return dst, save_folder
        
    image = worker_t2i(text, seed, step)
    image.save(dst)
    dst = worker_xbg(image, save_folder)
    return dst, save_folder

def stage_1_xbg(image, save_folder): 
    if isinstance(image, str):
        image = Image.open(image)
    dst =  save_folder + '/img_nobg.png'
    rgba = worker_xbg(image)
    rgba.save(dst)
    return dst
    
def stage_2_i2v(image, seed, step, save_folder):
    if isinstance(image, str):
        image = Image.open(image)
    gif_dst = save_folder + '/views.gif'
    res_img, pils = worker_i2v(image, seed, step)
    save_gif(pils, gif_dst)
    views_img, cond_img = res_img[0], res_img[1]
    img_array = np.asarray(views_img, dtype=np.uint8)
    show_img = rearrange(img_array, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_img = show_img[worker_i2v.order, ...]
    show_img = rearrange(show_img, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_img = Image.fromarray(show_img) 
    return views_img, cond_img, show_img

def stage_3_v23(
    views_pil, 
    cond_pil, 
    seed, 
    save_folder,
    target_face_count = 30000,
    do_texture_mapping = True,
    do_render =True
): 
    do_texture_mapping = do_texture_mapping or do_render
    obj_dst = save_folder + '/mesh_with_colors.obj'
    glb_dst = save_folder + '/mesh.glb'
    worker_v23(
        views_pil, 
        cond_pil, 
        seed = seed, 
        save_folder = save_folder,
        target_face_count = target_face_count,
        do_texture_mapping = do_texture_mapping
    )
    return obj_dst, glb_dst

def stage_4_gif(obj_dst, save_folder, do_render_gif=True):
    if not do_render_gif: return None
    gif_dst = save_folder + '/output.gif'
    worker_gif(
        save_folder + '/mesh.obj',
        gif_dst_path = gif_dst
    )
    return gif_dst
# ===============================================================
# gradio display
# ===============================================================
with gr.Blocks() as demo:
    gr.Markdown(CONST_HEADER)
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            with gr.Tab("Text to 3D"):
                with gr.Column():
                    text = gr.TextArea('一只黑白相间的熊猫在白色背景上居中坐着，呈现出卡通风格和可爱氛围。', lines=1, max_lines=10, label='Input text')
                    with gr.Row():
                        textgen_seed = gr.Number(value=0, label="T2I seed", precision=0)
                        textgen_step = gr.Number(value=25, label="T2I step", precision=0)
                        textgen_SEED = gr.Number(value=0, label="Gen seed", precision=0)
                        textgen_STEP = gr.Number(value=50, label="Gen step", precision=0)
                        textgen_max_faces = gr.Number(value=90000, label="max number of faces", precision=0)
                        
                    with gr.Row():
                        textgen_do_texture_mapping = gr.Checkbox(label="texture mapping", value=False, interactive=True)
                        textgen_do_render_gif = gr.Checkbox(label="Render gif", value=False, interactive=True)
                        textgen_submit = gr.Button("Generate", variant="primary")

                    with gr.Row():
                        gr.Examples(examples=example_ts, inputs=[text], label="Txt examples", examples_per_page=10)
                    
            with gr.Tab("Image to 3D"):
                with gr.Column():
                    input_image = gr.Image(label="Input image",
                                           width=256, height=256, type="pil",
                                           image_mode="RGBA", sources="upload",
                                           interactive=True)
                    with gr.Row(): 
                        imggen_SEED = gr.Number(value=0, label="Gen seed", precision=0)
                        imggen_STEP = gr.Number(value=50, label="Gen step", precision=0)
                        imggen_max_faces = gr.Number(value=90000, label="max number of faces", precision=0)

                    with gr.Row():
                        imggen_do_texture_mapping = gr.Checkbox(label="texture mapping", value=False, interactive=True)
                        imggen_do_render_gif = gr.Checkbox(label="Render gif", value=False, interactive=True)
                        imggen_submit = gr.Button("Generate", variant="primary")       
                    with gr.Row():
                        gr.Examples(
                            examples=example_is, 
                            inputs=[input_image], 
                            label="Img examples",
                            examples_per_page=10
                        )
           
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=2):
                    rem_bg_image = gr.Image(label="No backgraound image", type="pil",
                                           image_mode="RGBA", interactive=False)
                with gr.Column(scale=3):
                    result_image = gr.Image(label="Multi views", type="pil", interactive=False)
                
            with gr.Row():                
                result_3dobj = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="Output Obj",
                    show_label=True,
                    visible=True,
                    camera_position=[90, 90, None],
                    interactive=False
                )

                result_3dglb = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="Output Glb",
                    show_label=True,
                    visible=True,
                    camera_position=[90, 90, None],
                    interactive=False
                )
                result_gif = gr.Image(label="Rendered GIF", interactive=False)
                
            with gr.Row():    
                gr.Markdown("""
                We recommend downloading and opening Glb with 3D software, such as Blender, MeshLab, etc.
                
                Limited by gradio, Obj file here only be shown as vertex shading, but Glb can be texture shading.
                """)

#===============================================================
# gradio running code
#===============================================================

    none = gr.State(None)
    save_folder = gr.State()
    cond_image = gr.State()
    views_image = gr.State()
    text_image = gr.State()
    
    textgen_submit.click(
        fn=stage_0_t2i, inputs=[text, none, textgen_seed, textgen_step], 
        outputs=[rem_bg_image, save_folder],
    ).success(
        fn=stage_2_i2v, inputs=[rem_bg_image, textgen_SEED, textgen_STEP, save_folder], 
        outputs=[views_image, cond_image, result_image],
    ).success(
        fn=stage_3_v23, inputs=[views_image, cond_image, textgen_SEED, save_folder, 
                                textgen_max_faces, textgen_do_texture_mapping,
                                textgen_do_render_gif], 
        outputs=[result_3dobj, result_3dglb],
    ).success(
        fn=stage_4_gif, inputs=[result_3dglb, save_folder, textgen_do_render_gif], 
        outputs=[result_gif],
    ).success(lambda: print('Text_to_3D Done ...'))

    imggen_submit.click(
        fn=stage_0_t2i, inputs=[none, input_image, textgen_seed, textgen_step], 
        outputs=[text_image, save_folder],
    ).success(
        fn=stage_1_xbg, inputs=[text_image, save_folder], 
        outputs=[rem_bg_image],
    ).success(
        fn=stage_2_i2v, inputs=[rem_bg_image, imggen_SEED, imggen_STEP, save_folder], 
        outputs=[views_image, cond_image, result_image],
    ).success(
        fn=stage_3_v23, inputs=[views_image, cond_image, imggen_SEED, save_folder, 
                                imggen_max_faces, imggen_do_texture_mapping, 
                                imggen_do_render_gif], 
        outputs=[result_3dobj, result_3dglb],
    ).success(
        fn=stage_4_gif, inputs=[result_3dglb, save_folder, imggen_do_render_gif], 
        outputs=[result_gif],
    ).success(lambda: print('Image_to_3D Done ...'))
    
#===============================================================
# start gradio server
#===============================================================

    gr.Markdown(CONST_CITATION)
    demo.queue(max_size=CONST_MAX_QUEUE)
    demo.launch(server_name=CONST_SERVER, server_port=CONST_PORT)

