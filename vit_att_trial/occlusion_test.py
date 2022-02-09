from PIL.ImageColor import colormap
from attn_data import Kermany_DataSet
import timm
import wandb
import os
from timm.models.swin_transformer import SwinTransformer
from utils2 import *

# from loging_gradcam import reshape_transform
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
from res_models import *
from convnext import convnext_xlarge, convnext_base
from visualization_helpers import generate_visualization, reshape_transform, show_cam_on_image, create_avg_img, \
    generate_cam_vis


# custom function to conduct occlusion experiments

def occlusion(model, image, label, occ_size=100, occ_stride=100, occ_pixel=0.5):
    # get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]

    # setting the output image width and height
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))

    # create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))
    # iterate all the pixels in each column
    max_prob = -100
    top_10_masks = [((0, 0, 0, 0), max_prob)]
    for h in range(0, height):
        for w in range(0, width):

            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if w_end >= width or h_end >= height:
                continue

            input_image = image.clone().detach()

            # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

            # run inference on modified image
            output = model(input_image)

            prob = output.tolist()[0][label]
            if prob > top_10_masks[-1][1]:

                top_10_masks.append(((w_start, w_end, h_start, h_end), prob))
                top_10_masks = sorted(top_10_masks, key=lambda x: x[1], reverse=True)
                if len(top_10_masks) > 20:
                    top_10_masks.pop(-1)

            # setting the heatmap location to probability value
            heatmap[h, w] = prob
    image_transformer_attribution = images.squeeze().permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    # vis = np.uint8(255 * vis)
    # vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    heatmap = 1 - heatmap
    heatmap = heatmap.permute(1, 0)

    heatmap = (heatmap - heatmap.min()) / (
            heatmap.max() - heatmap.min())
    heatmap = 1 - heatmap
    new_heatmap = cv2.resize(np.float32(heatmap), (512, 496))

    # heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    heatmap = heatmap / np.max(heatmap)
    heatmap_om_image = show_cam_on_image(image_transformer_attribution, new_heatmap)

    masked_image = image.clone().detach()

    for tup in top_10_masks:
        # print(tup)
        w_start, w_end, h_start, h_end = tup[0]
        # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
        masked_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel
    output = model(masked_image)
    return heatmap_om_image, heatmap, masked_image, output


seed = 25
torch.manual_seed(hash("by removing stochasticity") % seed)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

models = [convnext_base()]

config = {'convnext_xlarge': {'target_layers': [models[0].downsample_layers[0], models[0].downsample_layers[1],
                                                models[0].downsample_layers[-1]]},
          'use_wandb': True,
          'visualize_all_class': True,
          'seed': 25,
          'test_path': "../../data/kermany/test",
          # "test_path": "../../../Documents/GitHub/test",
          'label_names': ["NORMAL", "CNV", "DME", "DRUSEN"],
          'cam_algs': [GradCAM, GradCAMPlusPlus, XGradCAM, ScoreCAM],
          'cam_names': ['GradCAM', 'GradCAMPlusPlus', 'XGradCAM', 'ScoreCAM'],
          'layer_by_layer_cam': True
          }
# Iterate through test dataset
#  'ScoreCAM', 'GradCAMPlusPlus', 'XGradCAM', 'EigenCAM', 'EigenGradCAM',
# , ScoreCAM, EigenCAM, GradCAMPlusPlus, XGradCAM, EigenGradCAM


columns = ["Original Image", "Truth", "predicted", "Logits", "T/F", 'GradCAM', "occlusion", "occ_on_image", "best_mask",
           "Logits after", "new prediction"]
if config['visualize_all_class']:
    columns = ["Original Image", "Truth", "predicted", "Logits", "T/F", 'CAM', "OCC_NORMAL", "OCC_CNV", "OCC_DME",
               "OCC_DRUSEN"]

if config['use_wandb']:
    wandb.init(project="occlusion_test")

CLS2IDX = config['label_names']

test_dataset = Kermany_DataSet(config['test_path'])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=True)

count = 0
name = 'convnext_xlarge'
for i, (images, labels) in enumerate(test_loader):
    if count == 50:
        break

    images = Variable(images).to(device)
    labels = labels.to(device)

    model = models[4]
    model.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
    model = model.to(device)

    outputs = model(images)
    # Get predictions from the maximum value
    _, predictions = torch.max(outputs.data, 1)

    if torch.topk(k=2, input=outputs).values[0, 0] - torch.topk(k=2, input=outputs).values[0, 1] > 1.5:
        continue

    count += 1
    print(count)

    if config['use_wandb']:
        test_dt = wandb.Table(columns=columns)

    target_layers = [config[name]['target_layers'][-1]]
    # compute occlusion heatmap

    cam_and_occlusion = []
    new_ouputs = None
    target_categories = [labels.item()]
    targets = [ClassifierOutputTarget(category) for category in target_categories]

    cam_map, curr_grads, _ = generate_cam_vis(model, GradCAM, target_layers, name,
                                              images, labels, targets)
    cam_and_occlusion.append(cam_map)

    for k in range(4):
        print("curr label: {}".format(k))

        if not config['visualize_all_class']:
            k = labels.item()

        inter_heatmap, occlusion_map, best_mask, new_ouputs = occlusion(model, images, k, 50, 5)
        _, new_predictions = torch.max(new_ouputs.data, 1)

        cam_and_occlusion.append(inter_heatmap)
        if not config['visualize_all_class']:
            break

    plt.clf()
    plt.bar(config['label_names'], outputs.cpu().detach().numpy()[0])
    # plt.xlabel(label_names)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)

    plt.clf()
    plt.bar(config['label_names'], new_ouputs.cpu().detach().numpy()[0])
    img_buf_2 = io.BytesIO()
    plt.savefig(img_buf_2, format='png')
    im_2 = Image.open(img_buf_2)

    row = [wandb.Image(images), config['label_names'][labels.item()], config['label_names'][predictions.item()],
           wandb.Image(im), predictions.item() == labels.item(),
           ] + [wandb.Image(cam_and_occlusion[i]) for i in range(len(cam_and_occlusion))] + [wandb.Image(inter_heatmap),
                                                                                             wandb.Image(best_mask),
                                                                                             wandb.Image(im_2),
                                                                                             config['label_names'][
                                                                                                 new_predictions.item()]]

    if config['visualize_all_class']:
        row = [wandb.Image(images),
               config['label_names'][labels.item()], config['label_names'][predictions.item()], wandb.Image(im),
               predictions.item() == labels.item(),
               ] + [wandb.Image(cam_and_occlusion[i]) for i in range(len(cam_and_occlusion))]

    if config['use_wandb']:
        test_dt.add_data(*row)
        wandb.log({f"image_{count}_{config['label_names'][labels.item()]}": test_dt})
