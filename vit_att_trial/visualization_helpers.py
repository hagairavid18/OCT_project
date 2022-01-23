from attn_data import Kermany_DataSet
import timm
import wandb
import os
from timm.models.swin_transformer import SwinTransformer
from utils2 import *
from model_running import *
import numpy as np
import random
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
import cv2 as cv
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os
import io
from PIL import Image
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
from pytorch_grad_cam.ablation_layer import AblationLayerVit
# from res_models import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_visualization(original_image,attribution_generator, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).to(device),
                                                                 method="transformer_attribution",
                                                                 index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 31, 32)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(496, 512).to(device).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis, transformer_attribution

def reshape_transform(tensor, height=31, width=32):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap * 0.4 + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def create_vit_models():
    name = 'vit_base_patch16_224'
    # initialize ViT pretrained
    model_timm = timm.create_model(name, num_classes=4, img_size=(496, 512))
    model_timm.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
    model_timm = model_timm.to(device)

    model_attn = vit_LRP(num_classes=4, img_size=(496, 512))
    model_attn.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
    model_attn = model_attn.to(device)
    model_attn.eval()
    attribution_generator = LRP(model_attn)
    return  model_timm, model_attn,attribution_generator

def create_avg_img(init_avg,image_transformer_attribution,just_grads):
    avg = init_avg
    for j, grad in enumerate(just_grads):
        g = grad.copy()
        # plt.imshow(g)
        # plt.title(str(j))
        # plt.show()
        g = np.where(g < g.max() / 4, g / 7, g)
        g = np.exp(g)
        g = g - g.min()
        g = g / g.max()
        # plt.imshow(g)
        # plt.title(str(j))
        # plt.show()
        avg += g
        # print(avg.max())
        avg_cam =  avg / avg.max()
        vis = show_cam_on_image(image_transformer_attribution, avg_cam)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        # plt.imshow(vis)
        # plt.show()
        avg_cam = vis
        return  avg_cam

