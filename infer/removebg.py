import os, sys
sys.path.insert(0, f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

import numpy as np
from PIL import Image
from rembg import remove, new_session
from infer.utils import timing_decorator

class Removebg():
    def __init__(self, name="u2net"):
        self.session = new_session(name)

    @timing_decorator("remove background")
    def __call__(self, rgb_maybe, force=True):
        '''
            args:
                rgb_maybe: PIL.Image, with RGB mode or RGBA mode
                force: bool, if input is RGBA mode, covert to RGB then remove bg
            return:
                rgba_img: PIL.Image, with RGBA mode
        '''
        if rgb_maybe.mode == "RGBA":
            if force:  
                rgb_maybe = rgb_maybe.convert("RGB")
                rgba_img = remove(rgb_maybe, session=self.session)
            else:
                rgba_img = rgb_maybe
        else:
            rgba_img = remove(rgb_maybe, session=self.session)
            
        rgba_img = white_out_background(rgba_img)
        
        rgba_img = preprocess(rgba_img)
        
        return rgba_img


def white_out_background(pil_img):
    data = pil_img.getdata()
    new_data = []
    for r, g, b, a in data:
        if a < 16:  # background
            new_data.append((255, 255, 255, 0))  # full white color
        else:
            is_white = (r>235) and (g>235) and (b>235)
            new_r = 235 if is_white else r
            new_g = 235 if is_white else g
            new_b = 235 if is_white else b
            new_data.append((new_r, new_g, new_b, a))
    pil_img.putdata(new_data)
    return pil_img
    
def preprocess(rgba_img, size=(512,512), ratio=1.15):
    image = np.asarray(rgba_img)
    rgb, alpha = image[:,:,:3] / 255., image[:,:,3:] / 255.

    # crop
    coords = np.nonzero(alpha > 0.1)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    rgb = (rgb[x_min:x_max, y_min:y_max, :] * 255).astype("uint8")
    alpha = (alpha[x_min:x_max, y_min:y_max, 0] * 255).astype("uint8")

    # padding
    h, w = rgb.shape[:2]
    resize_side = int(max(h, w) * ratio)
    pad_h, pad_w = resize_side - h, resize_side - w
    start_h, start_w = pad_h // 2, pad_w // 2
    new_rgb = np.ones((resize_side, resize_side, 3), dtype=np.uint8) * 255
    new_alpha = np.zeros((resize_side, resize_side), dtype=np.uint8)
    new_rgb[start_h:start_h + h, start_w:start_w + w] = rgb
    new_alpha[start_h:start_h + h, start_w:start_w + w] = alpha
    rgba_array = np.concatenate((new_rgb, new_alpha[:,:,None]), axis=-1)
    
    rgba_image = Image.fromarray(rgba_array, 'RGBA')
    rgba_image = rgba_image.resize(size)
    return rgba_image


if __name__ == "__main__":
    
    import argparse
    
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--rgb_path", type=str, required=True)
        parser.add_argument("--output_rgba_path", type=str, required=True)
        parser.add_argument("--force", default=False, action="store_true")
        return parser.parse_args()
        
    args = get_args()

    rgb_maybe = Image.open(args.rgb_path)
    
    model = Removebg()

    rgba_pil = model(rgb_maybe, args.force)

    rgba_pil.save(args.output_rgba_path)

    