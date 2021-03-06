from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os

from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
import wandb
import random


from attn_data import Kermany_DataSet
import timm
import wandb
import os
from timm.models.swin_transformer import SwinTransformer
from utils2 import *
from model_running import *
import numpy as np
import random
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
import cv2 as cv



wandb.init(project="test_attention")

seed = 25
torch.manual_seed(hash("by removing stochasticity") % seed)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

def_args = dot_dict({
    "train": ["../../../../data/kermany/train"],
    "val": ["../../../../data/kermany/val"],
    "test": ["../../../../data/kermany/test"],
})

label_names_dict = {
    0: "NORMAL",
    1: "CNV",
    2: "DME",
    3: "DRUSEN",
}

label_names = [
    "NORMAL",
    "CNV",
    "DME",
    "DRUSEN",
]
CLS2IDX = label_names_dict
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


name = 'vit_base_patch16_224'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# initialize ViT pretrained
model = vit_LRP(num_classes=4, img_size=(496, 512))
model.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
model = model.to(device)
model.eval()
attribution_generator = LRP(model)


def generate_visualization(original_image, class_index=None):
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
    return vis


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(4, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)


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

columns = ["id", "Original Image", "Predicted", "Truth", "Attention_Map"]
# for a in label_names:
#     columns.append("score_" + a)
test_dt = wandb.Table(columns=columns)

for i, (images, labels) in enumerate(test_loader):

    if i % 10 == 0:
        print(f'image : {i}\n\n\n')
    images = Variable(images).to(device)
    labels = labels.to(device)
    images = images.squeeze()
    # Forward pass only to get logits/output
    # print(images.shape)
    outputs = model(images.unsqueeze(0))

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

    cat = generate_visualization(images)


    row = [i, wandb.Image(images), label_names[predicted.item()], label_names[labels.item()],
           wandb.Image(cat)]
    test_dt.add_data(*row)

    # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
    #                                                    y_true=ground_truth, preds=predictions,
    #                                                    class_names=label_names)})

accuracy = correct / total
metrics = {f'Test Accuracy_{name}': accuracy}
for label in range(4):
    metrics[f'Test Accuracy_{name}' + label_names[label]] = correct_arr[label] / total_arr[label]
wandb.log(metrics)
wandb.log({f"Grads_{name}": test_dt})





