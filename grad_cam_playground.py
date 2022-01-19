from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
import cv2 as cv
from res_models import *
import numpy as np
import random
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


def grad_cam_model(res_model, input_tensor, label=0):
    res_model.eval()
    target_layers = [res_model.resnet.layer4[-1]]
    out = res_model(input_tensor)
    cams = [GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad]
    res = []
    for cam_algo in cams:
        cam = cam_algo(model=res_model, target_layers=target_layers,
                       use_cuda=True if torch.cuda.is_available() else False)
        target_category = label
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        res.append(grayscale_cam)
    return res, out


def vis(img, grayscale_cam, model_name):
    cams = ['GradCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad']

    fig = plt.figure()
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    fig.suptitle(f"CAM Algorithms Results ({model_name}):", fontsize=16)
    for i in range(1, 9):
        if i == 1:
            ax = plt.subplot(int(f'24{i}'))
            ax.set_title("Original Image")
            ax.imshow(img / 255)
        else:
            heatmap = np.uint8(255 * grayscale_cam[i - 2])
            heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img
            superimposed_img *= 255.0 / superimposed_img.max()
            ax = plt.subplot(int(f'24{i}'))
            ax.set_title(cams[i - 2], fontsize=10)
            ax.imshow(superimposed_img / 255)
    plt.show()


def get_img(path):
    t = transforms.Compose([transforms.ToTensor(), transforms.RandomResizedCrop((496, 512))])
    cv_im = cv.imread(path)
    cv_im = cv.resize(cv_im, (512, 496))
    input_tensor = t(cv_im)
    input_tensor = input_tensor.reshape(1, input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2])
    return cv_im, input_tensor


start = 'C:/Users/guylu/Academy/MSc_WIZ/BasriLab/kermany/OCT2017_/test/'
paths = ['DME/DME-70266-2.jpeg']
models_names = ['res18', 'res101', 'res152']
models = [Resnet18(4), Resnet101(4), Resnet152(4)]
for path in paths:
    for name, model in zip(models_names, models):
        model.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))

        cv_im, input_tensor = get_img(start + path)
        grad, out = grad_cam_model(model, input_tensor, 0)
        vis(cv_im, grad, name)

# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50
# import torch
# import matplotlib.pyplot as plt
# from torchvision import transforms as transforms
# import cv2 as cv
# from torchvision.models import resnet18, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2
# import torch.nn as nn
# import numpy as np
# seed = 25
# torch.manual_seed(hash("by removing stochasticity") % seed)
# torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed)
#
# model = Resnet101(4)
# model.load_state_dict(torch.load('res101.pt', map_location=torch.device('cpu')))
# model.eval()
# target_layers = [model.resnet.layer4[-1]]
# t = transforms.Compose([transforms.ToTensor(), transforms.RandomResizedCrop((496, 512))])
# cv_im = cv.imread('C:/Users/guylu/Academy/MSc_WIZ/BasriLab/kermany/OCT2017_/test/DME/DME-70266-2.jpeg')
# cv_im = cv.resize(cv_im, (512, 496))
# input_tensor = t(cv_im)
# input_tensor = input_tensor.reshape(1,input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2])
# out = model(input_tensor)
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True if torch.cuda.is_available() else False)
# target_category = torch.tensor(3)
# grayscale_cam = cam(input_tensor=input_tensor)
# grayscale_cam = grayscale_cam[0, :]
#
# # plt.imshow(grayscale_cam)
# # plt.show()
#
# heatmap = np.uint8(255 * grayscale_cam)
# heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
# superimposed_img = heatmap * 0.4 + cv_im
# superimposed_img *= 255.0/superimposed_img.max()
# plt.imshow(superimposed_img/255)
# plt.show()
