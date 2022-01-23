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
from visualization_helpers import generate_visualization,reshape_transform,show_cam_on_image,create_avg_img


seed = 25
torch.manual_seed(hash("by removing stochasticity") % seed)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'



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





# model_timm, model_attn,attribution_generator = create_vit_models()



models = [Resnet18(4),Resnet50(4),Resnet101(4),Resnet152(4),convnext_base(),model_timm ]


config = {'res18':{'target_layers':[models[0].resnet.layer4[-1]]},
          'res50':{'target_layers':[models[1].resnet.layer4[-1]]},
          'res101':{'target_layers':[models[2].resnet.layer4[-1]]},
          'res152':{'target_layers':[models[3].resnet.layer4[-1]]},
          'convnext_xlarge':{'target_layers':[models[4].downsample_layers[-1]]},
          'vit_base_patch16_224':{'target_layers':[models[5].blocks[-1].norm1]},
          'use_wandb': True,
          'visualize_all_class': False,
          'seed': 25,
          'test_path' :"../../data/kermany/test",
          'label_names':["NORMAL","CNV","DME","DRUSEN"],
          'cam_algs': [GradCAM],
          }
# Iterate through test dataset
#  'ScoreCAM', 'GradCAMPlusPlus', 'XGradCAM', 'EigenCAM', 'EigenGradCAM',

columns = ["model_name","id", "Original Image", "Predicted" ,"Logits","Truth", "Correct","curr_target","attention"]\
          +[ str(cam) for cam in config['cam_algs']]+['Avg']

    # "test": ["../../../Documents/GitHub/test"]


# use_wandb= True

if config['use_wandb']:
    wandb.init(project="test_attn_plus_gradcam")
#
# label_names = {
#     0: "NORMAL",
#     1: "CNV",
#     2: "DME",
#     3: "DRUSEN",
# }
CLS2IDX = config['label_names']










# pytorch_total_params = sum(p.numel() for p in model_timm.parameters())
# pytorch_total_params_train = sum(p.numel() for p in model_timm.parameters() if p.requires_grad)
# if config['use_wandb']:
#     wandb.log({"Total Params": pytorch_total_params})
#     wandb.log({"Trainable Params": pytorch_total_params_train})





# def_args = {
#     "train": ["../../../data/kermany/train"],
#     "val": ["../../../data/kermany/val"],
#     "test": ["../../data/kermany/test"],
#     # "test": ["../../../Documents/GitHub/test"],
# }


test_dataset = Kermany_DataSet(config['test_path'][0])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=True)
# , ScoreCAM, EigenCAM, GradCAMPlusPlus, XGradCAM, EigenGradCAM

# for a in label_names:
#     columns.append("score_" + a)
if config['use_wandb']:
    test_dt = wandb.Table(columns=columns)

#,"res101","res152","convnext_xlarge", 'vit_base_patch16_224'
names = ["res18","res50"]
for i, (images, labels) in enumerate(test_loader):

    for index, name in enumerate(names):
        # space_row = [None for _ in columns]
        # space_row[0] = "##### {} #####".format(name)
        # if config['use_wandb']:
        #     test_dt.add_data(*space_row)
        model = models[index]
        print(name)
        if name != 'vit_base_patch16_224':
            model.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
            model = model.to(device)


        predictions = None
        ground_truth = None
        # Iterate through test dataset
        if i % 10 == 0:
            print(f'image : {i}\n\n\n')
        images = Variable(images).to(device)
        labels = labels.to(device)
        # Forward pass only to get logits/output
        # print(images.shape)
        if name == 'vit_base_patch16_224':
            outputs_attn = model_attn(images)
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        # # total += labels.size(0)
        # # correct += (predicted == labels).sum()
        #
        # for label in range(4):
        #     correct_arr[label] += (((predicted == labels) & (labels == label)).sum())
        #     total_arr[label] += (labels == label).sum()

        if i == 0:
            predictions = predicted
            ground_truth = labels
        else:
            predictions = torch.cat((predictions, predicted), 0)
            ground_truth = torch.cat((ground_truth, labels), 0)
        target_layers = config[name]['target_layers']


        # image_transformer_attribution = None
        for k in range(4):
            print("curr label: {}".format(k))
            if not config['visualize_all_class']:
                k=labels[0]
            just_grads = []
            res = []
            attn_diff_cls = []

            target_categories = [k]
            targets = [ClassifierOutputTarget(category) for category in target_categories]

            for cam_algo in config['cam_algs']:
                # print(images.shape)

                cam = cam_algo(model=model, target_layers=target_layers,
                               use_cuda=True if torch.cuda.is_available() else False)#, reshape_transform=reshape_transform,
                               # )
                if name== 'vit_base_patch16_224':
                    cam = cam_algo(model=model, target_layers=target_layers,
                                   use_cuda=True if torch.cuda.is_available() else False, reshape_transform=reshape_transform,
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

            # # layer by layer grad cam
            # for level in range(0,len(model_timm.blocks),2):
            #     target_layers = [model_timm.blocks[level].norm1]
            #
            #     cam = GradCAM(model=model_timm, target_layers=target_layers,
            #                    use_cuda=True if torch.cuda.is_available() else False, reshape_transform=reshape_transform,
            #                    )
            #     target_category = labels.item()
            #     grayscale_cam = cam(input_tensor=images, aug_smooth=True, eigen_smooth=True, targets=targets)
            #     # just_grads.append(grayscale_cam[0, :])
            #     image_transformer_attribution = images.squeeze().permute(1, 2, 0).data.cpu().numpy()
            #     image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            #             image_transformer_attribution.max() - image_transformer_attribution.min())
            #     vis = show_cam_on_image(image_transformer_attribution, grayscale_cam[0, :])
            #     vis = np.uint8(255 * vis)
            #     vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
            #     res.append(vis)  # superimposed_img / 255)



            gradcam = res
            # images = images.squeeze()
            attention = None
            avg = just_grads[0].copy() * 0

            if name == 'vit_base_patch16_224':
                cat, attn_map = generate_visualization(images.squeeze(),attribution_generator=attribution_generator)
                attention = generate_visualization(images.squeeze(), attribution_generator=attribution_generator,class_index=k,)[0]
                avg = attn_map.copy() * 6
            avg_cam = create_avg_img(avg,image_transformer_attribution,just_grads)
            # # plt.imshow(avg)
            # # plt.show()
            # vis = show_cam_on_image(image_transformer_attribution, avg_cam)
            # vis = np.uint8(255 * vis)
            # vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
            # # plt.imshow(vis)
            # # plt.show()
            # avg_cam = vis
            T = predicted.item() == labels.item()
            out = outputs

            plt.bar(config['label_names'], out.cpu().detach().numpy()[0])
            # plt.xlabel(label_names)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            im = Image.open(img_buf)

            row = ["##### {} #####".format(name),str(i), wandb.Image(images), config['label_names'][predicted.item()], wandb.Image(im), config['label_names'][labels.item()], T,
                   config['label_names'][k]]+[ None] + [wandb.Image(cam) for cam in gradcam ] +[wandb.Image(avg_cam)]
            if name == 'vit_base_patch16_224':
                row[7] =wandb.Image(attention)
            print(row[7])
            if config['use_wandb']:
                test_dt.add_data(*row)
            if not config['visualize_all_class']:
                break



if config['use_wandb']:
    wandb.log({f"Grads_{name}": test_dt})
