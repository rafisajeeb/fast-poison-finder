from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms.functional as TF

def add_blended_square_pil(img_pil: Image.Image, box=(27,27,31,31), alpha=0.2):
    trig = Image.new('RGB', img_pil.size, (0,0,0))
    d = ImageDraw.Draw(trig)
    d.rectangle(box, fill=(255,255,255))
    return Image.blend(img_pil, trig, alpha)

def add_blended_square_tensor(img: torch.Tensor, box=(27,27,31,31), alpha=0.2):
    # img: [C,H,W], float in [0,1]
    c,h,w = img.shape
    x0,y0,x1,y1 = box
    overlay = img.clone()
    overlay[:, y0:y1+1, x0:x1+1] = 1.0
    return (1-alpha)*img + alpha*overlay

def default_trigger_box(h=32, w=32, size=5, margin=1):
    # bottom-right square
    x1 = w - margin - 1
    y1 = h - margin - 1
    x0 = max(0, x1 - size + 1)
    y0 = max(0, y1 - size + 1)
    return (x0, y0, x1, y1)
