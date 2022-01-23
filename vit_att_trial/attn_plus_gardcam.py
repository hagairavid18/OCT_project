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
from visualization_helpers import generate_visualization,reshape_transform,show_cam_on_image,create_avg_img,generate_cam_vis


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


config = {'res18':{'target_layers':[models[0].resnet.layer4[i] for i in range(len(models[0].resnet.layer4))]},
          'res50':{'target_layers':[models[1].resnet.layer4[i] for i in range(len(models[1].resnet.layer4))]},
          'res101':{'target_layers':[models[2].resnet.layer4[i] for i in range(len(models[2].resnet.layer4))]},
          'res152':{'target_layers':[models[3].resnet.layer4[i] for i in range(len(models[3].resnet.layer4))]},
          'convnext_xlarge':{'target_layers':[models[4].downsample_layers[-1]]},
          'vit_base_patch16_224':{'target_layers':[models[5].blocks[i].norm1 for i in range(0,len(model_timm.blocks),2)]},
          'use_wandb': True,
          'visualize_all_class': False,
          'seed': 25,
          'test_path' :"../../data/kermany/test",
          'label_names':["NORMAL","CNV","DME","DRUSEN"],
          'cam_algs': [GradCAM,GradCAMPlusPlus,XGradCAM],
          'cam_names':['GradCAM','GradCAMPlusPlus','XGradCAM'],
          'layer_by_layer_cam' :True
          }
# Iterate through test dataset
#  'ScoreCAM', 'GradCAMPlusPlus', 'XGradCAM', 'EigenCAM', 'EigenGradCAM',

columns = ["model_name","id", "Original Image", "Predicted" ,"Logits","Truth", "Correct","curr_target","attention"]\
          +[ cam for cam in config['cam_names']]+['Avg'] +["layer {}".format(i) for i in range(len(config['vit_base_patch16_224']['target_layers']))]

    # "test": ["../../../Documents/GitHub/test"]



if config['use_wandb']:
    wandb.init(project="test_attn_plus_gradcam")

CLS2IDX = config['label_names']

test_dataset = Kermany_DataSet(config['test_path'])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=True)
# , ScoreCAM, EigenCAM, GradCAMPlusPlus, XGradCAM, EigenGradCAM


if config['use_wandb']:
    test_dt = wandb.Table(columns=columns)


#,"res101","res152","convnext_xlarge", 'vit_base_patch16_224'
names = ["res18","res50","res101","res152","convnext_xlarge", 'vit_base_patch16_224']
print(len(test_dataset))
for i, (images, labels) in enumerate(test_loader):
    print('here')

    for index, name in enumerate(names):

        model = models[index]
        print(name)
        if name != 'vit_base_patch16_224':
            model.load_state_dict(torch.load(f'{name}.pt', map_location=torch.device(device)))
            model = model.to(device)

        correct = 0.0
        correct_arr = [0.0] * 10
        total = 0.0
        total_arr = [0.0] * 10
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
        target_layers = [config[name]['target_layers'][-1]]


        image_transformer_attribution = None
        for k in range(4):
            print("curr label: {}".format(k))
            if not config['visualize_all_class']:
                k=labels[0]
            just_grads,res,attn_diff_cls,layer_cam = [],[],[],[]
            target_categories = [k]
            targets = [ClassifierOutputTarget(category) for category in target_categories]

            for cam_algo in config['cam_algs']:

                vis, curr_grads,image_transformer_attribution = generate_cam_vis(model,cam_algo, target_layers,name,images,labels,targets)
                res.append(vis)  # superimposed_img / 255)
                just_grads.append(curr_grads)

            if config['layer_by_layer_cam']:

                # layer by layer grad cam
                for target_layer in config[name]['target_layers']:
                    # print(target_layer)
                    target_layer = [target_layer]
                    vis, curr_grads, image_transformer_attribution = generate_cam_vis(model, GradCAM, target_layer,
                                                                                      name, images, labels, targets)
                    layer_cam.append(vis)  # superimposed_img / 255)
                    # just_grads.append(curr_grads)


                    # cam = GradCAM(model=model, target_layers=target_layer,
                    #                use_cuda=True if torch.cuda.is_available() else False, reshape_transform=reshape_transform,
                    #                )
                    # target_category = labels.item()
                    # grayscale_cam = cam(input_tensor=images, aug_smooth=True, eigen_smooth=True, targets=targets)
                    # # just_grads.append(grayscale_cam[0, :])
                    # image_transformer_attribution = images.squeeze().permute(1, 2, 0).data.cpu().numpy()
                    # image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                    #         image_transformer_attribution.max() - image_transformer_attribution.min())
                    # vis = show_cam_on_image(image_transformer_attribution, grayscale_cam[0, :])
                    # vis = np.uint8(255 * vis)
                    # vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
                    # res.append(vis)  # superimposed_img / 255)



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

            row = ["# {} #".format(name),str(i), wandb.Image(images), config['label_names'][predicted.item()], wandb.Image(im), config['label_names'][labels.item()], T,
                   config['label_names'][k]]+[ None] +[wandb.Image(gradcam[i]) for i in range(len(gradcam))]+[wandb.Image(avg_cam)]
            row_2 = [  None for _ in range(len(config['vit_base_patch16_224']['target_layers']))]
            for pos in range(len(layer_cam)):
                row_2[pos] = wandb.Image(layer_cam[pos])
            row+=row_2

            if name == 'vit_base_patch16_224':
                row[8] =wandb.Image(attention)
            # print(row[7])
            if config['use_wandb']:
                test_dt.add_data(*row)
            if not config['visualize_all_class']:
                break



if config['use_wandb']:
    wandb.log({f"Grads_{name}": test_dt})
