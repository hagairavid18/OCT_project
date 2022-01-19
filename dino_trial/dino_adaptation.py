import torch
from vit_pytorch import ViT, Dino
from vit_pytorch.recorder import Recorder
from dino_data import Kermany_DataSet
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


seed = 25
torch.manual_seed(hash("by removing stochasticity") % seed)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project="test_attn_plus_gradcam")

label_names = {
    0: "NORMAL",
    1: "CNV",
    2: "DME",
    3: "DRUSEN",
}

model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048
)

######################################################################### v = Recorder(model)

learner = Dino(
    model,
    image_size=256,
    hidden_layer='to_latent',  # hidden layer name or index, from which to extract the embedding
    projection_hidden_size=256,  # projector network hidden dimension
    projection_layers=4,  # number of layers in projection network
    num_classes_K=65336,  # output logits dimensions (referenced as K in paper)
    student_temp=0.9,  # student temperature
    teacher_temp=0.04,  # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale=0.4,  # upper bound for local crop - 0.4 was recommended in the paper
    global_lower_crop_scale=0.5,  # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay=0.9,  # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay=0.9,
    # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
)

pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
wandb.log({"Total Params": pytorch_total_params})
wandb.log({"Trainable Params": pytorch_total_params_train})

def_args = dot_dict({
    "train": ["../../../data/kermany/train"],
    "val": ["../../../data/kermany/val"],
    "test": ["../../../../data/kermany/test"],
})

label_names = [
    "NORMAL",
    "CNV",
    "DME",
    "DRUSEN",
]
test_dataset = Kermany_DataSet(def_args.test[0])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)
correct = 0.0
correct_arr = [0.0] * 10
total = 0.0
total_arr = [0.0] * 10
predictions = None
ground_truth = None
# Iterate through test dataset


name = "dino"

for i, (images, labels) in enumerate(test_loader):
    if i % 10 == 0:
        print(f'image : {i}\n\n\n')
    images = Variable(images).to(device)
    labels = labels.to(device)
    images = images.squeeze()
    # Forward pass only to get logits/output
    # print(images.shape)
    outputs_attn = model(images.unsqueeze(0))
    outputs_timm = model(images.unsqueeze(0))

    # Get predictions from the maximum value
    _, predicted = torch.max(outputs_attn.data, 1)

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

    # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
    #                                                    y_true=ground_truth, preds=predictions,
    #                                                    class_names=label_names)})

accuracy = correct / total
metrics = {f'Test Accuracy_{name}': accuracy}
for label in range(4):
    metrics[f'Test Accuracy_{name}' + label_names[label]] = correct_arr[label] / total_arr[label]
wandb.log(metrics)
