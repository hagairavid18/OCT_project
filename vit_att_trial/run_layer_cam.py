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
from PIL import Image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import cv2 as cv
from torchvision import transforms

from misc_functions import preprocess_image
from layer_cam import LayerCam, save_class_activation_images
import os


wandb.init(project="layer_gradcam")
from vit_att_trial.misc_functions import convert_to_grayscale, get_positive_negative_saliency, save_gradient_images

seed = 25
torch.manual_seed(hash("by removing stochasticity") % seed)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def_args = {
    "train": ["../../../../data/kermany/train"],
    "val": ["../../../../data/kermany/val"],
    "test": ["../../../Documents/GitHub/test"],
    # "test": ["../../data/kermany/test"],
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
# names = ["res18", "res50", "res101", "res152"]
names = ["res18"]

# models = [Resnet18(4), Resnet50(4), Resnet101(4), Resnet152(4)]
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
    # , 'ScoreCAM', 'AblationCAM'
    columns = ["id", "Original Image", "Predicted", "Truth", "GradCAM", 'GradCAMPlusPlus',
               'XGradCAM', 'EigenCAM', 'FullGrad'] + ['layer cam {}'.format(i) for i in
                                                      range(
                                                          8)]  # + ['coloured_guided','grey_guided','pos_guided','neg_guided']
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
        # print(model.resnet)
        # , ScoreCAM, AblationCAM
        cams = [GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, FullGrad]

        grayscales = []

        # all grad-cam models
        for cam_algo in cams:
            # print(cam_algo)
            cam = cam_algo(model=model, target_layers=target_layers,
                           use_cuda=True if torch.cuda.is_available() else False)
            target_category = labels.item()
            grayscale_cam = cam(input_tensor=images)
            # print(grayscale_cam.shape)
            grayscales.append(grayscale_cam[0, :])

        # layer cam model
        for layer in range(8):
            layer_cam = LayerCam(model, target_layer=layer)
            target_category = labels.item()
            grayscales.append(layer_cam.generate_cam(images, labels.item()))

        # for each grayscale:
        res = []
        for grayscale_cam in grayscales:
            heatmap = np.uint8(255 * grayscale_cam)
            heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
            superimposed_img = heatmap * 0.01 + images.squeeze().permute(1, 2, 0).cpu().detach().numpy() * 5
            superimposed_img *= 255.0 / superimposed_img.max()
            res.append(superimposed_img / 255)

        row = [i, wandb.Image(images), label_names[predicted.item()], label_names[labels.item()]] + [
            wandb.Image(res[i]) for i in range(len(res))]

        # guided_res = []
        # # Guided backprop
        # GBP = GuidedBackprop(model)
        # # Get gradients
        # target_category = labels.item()
        # guided_grads = GBP.generate_gradients(images, labels.item())
        # print(guided_grads.dtype)
        # print(guided_grads.shape)
        # file_name_to_export = 'mamama'
        # save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
        # # Convert to grayscale
        # grayscale_guided_grads = convert_to_grayscale(guided_grads)
        # # Save grayscale gradients
        # save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
        # # Positive and negative saliency maps
        # pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        # save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
        # save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')

        # guided_res.append(guided_grads)
        # guided_res.append(convert_to_grayscale(guided_grads))
        # pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        # guided_res+=[pos_sal, neg_sal]
        # for guided in  guided_res:
        #     print(guided)
        #     guided = guided - guided.min()
        #     guided /= guided.max()
        #
        # row +=guided_res
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
