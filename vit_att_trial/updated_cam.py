
from data import Kermany_DataSet
import timm
import wandb
import os
from timm.models.swin_transformer import SwinTransformer
from utils import *
from res_models import *
from model_running import *
from convnext import convnext_base, convnext_large, convnext_xlarge
import numpy as np
import random
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
import cv2 as cv
import cv2

from layer_cam import LayerCam


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


wandb.init(project="test_gradcam")

seed = 25
torch.manual_seed(hash("by removing stochasticity") % seed)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def_args = {
    "train": ["../../../data/kermany/train"],
    "val": ["../../../data/kermany/val"],
    # "test": ["../../../Documents/GitHub/test"],
    "test": ["../../data/kermany/test"],
}

label_names = [
    "NORMAL",
    "CNV",
    "DME",
    "DRUSEN",
]
test_dataset = Kermany_DataSet(def_args['test'][0])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)
names = ["res18"]

# names = ["res18", "res50", "res101", "res152"]
models = [Resnet18(4), Resnet50(4), Resnet101(4), Resnet152(4)]
models = [Resnet18(4)]

for name, model in zip(names, models):
    model.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
    model = model.to(device)
    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10
    predictions = None
    ground_truth = None
    # Iterate through test dataset
    # 'ScoreCAM','AblationCAM',
    columns = ["id", "Original Image", "Predicted", "Truth","curr_target","curr_target_score", "GradCAM",  'GradCAMPlusPlus',
               'XGradCAM', 'EigenCAM', 'FullGrad','ScoreCAM','AblationCAM']+ ['layer cam {}'.format(i) for i in
                                                      range(
                                                          8)]
    # for a in label_names:
    #     columns.append("score_" + a)
    test_dt = wandb.Table(columns=columns)

    for i, (images, labels) in enumerate(test_loader):
        if i % 10 == 0:
            print(f'image : {i}\n\n\n')
        images = Variable(images).to(device)
        labels = labels.to(device)
        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)
        correct += (predicted == labels).sum()

        for label in range(4):
            correct_arr[label] += (((predicted == labels) & (labels == label)).sum())
            total_arr[label] += (labels == label).sum()

        if i == 0:
            predictions = predicted
            ground_truth = labels
        else:
            predictions = torch.cat((predictions, predicted), 0)
            ground_truth = torch.cat((ground_truth, labels), 0)

        target_layers = [model.resnet.layer4[-1]]
        #, ScoreCAM, AblationCAM
        cams = [GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, FullGrad, ScoreCAM, AblationCAM]
        for j in range(4):
            res = []
            just_grads = []


            for cam_algo in cams:
                # print(images.shape)
                target_categories = [j]
                targets = [ClassifierOutputTarget(category) for category in target_categories]
                cam = cam_algo(model=model, target_layers=target_layers,
                               use_cuda=True if torch.cuda.is_available() else False,
                               )
                target_category = labels.item()
                grayscale_cam = cam(input_tensor=images, aug_smooth=True, eigen_smooth=True,targets=targets)
                just_grads.append(grayscale_cam[0, :])
                image_transformer_attribution = images.squeeze().permute(1, 2, 0).data.cpu().numpy()
                image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                        image_transformer_attribution.max() - image_transformer_attribution.min())
                vis = show_cam_on_image(image_transformer_attribution, grayscale_cam[0, :])
                vis = np.uint8(255 * vis)
                vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
                res.append(vis)  # superimposed_img / 255)

            for layer in range(8):
                layer_cam = LayerCam(model, target_layer=layer)
                target_category = labels.item()
                grayscale_cam=layer_cam.generate_cam(images, target_class=j)
                # print(grayscale_cam.shape)
                image_transformer_attribution = images.squeeze().permute(1, 2, 0).data.cpu().numpy()
                image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                        image_transformer_attribution.max() - image_transformer_attribution.min())
                vis = show_cam_on_image(image_transformer_attribution, grayscale_cam)
                vis = np.uint8(255 * vis)
                vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
                res.append(vis)  # superimposed_img / 255)

            gradcam = res
            row = [i, wandb.Image(images), label_names[predicted.item()], label_names[labels.item()],label_names[j],outputs[0,j]]+[
                    wandb.Image(gradcam[i]) for i in range(len(gradcam))]
            test_dt.add_data(*row)
        space_row = [0 for k in row]
        test_dt.add_data(*space_row)

        # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
        #                                                    y_true=ground_truth, preds=predictions,
        #                                                    class_names=label_names)})

    accuracy = correct / total
    metrics = {f'Test Accuracy_{name}': accuracy}
    for label in range(4):
        metrics[f'Test Accuracy_{name}' + label_names[label]] = correct_arr[label] / total_arr[label]
    wandb.log(metrics)
    wandb.log({f"Grads_{name}": test_dt})
