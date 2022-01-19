from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os

from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP

label_names = {
    0: "NORMAL",
    1: "CNV",
    2: "DME",
    3: "DRUSEN",
}
CLS2IDX = label_names
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
    cam = heatmap * 0.4 + np.float32(img)
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


#
# image = Image.open('t.jpeg')
image = cv2.imread("t.jpeg")
transform = transforms.Compose([transforms.ToTensor(), transforms.RandomResizedCrop((496, 512))])
dog_cat_image = transform(image).to(device)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(image);
axs[0].axis('off');
print(dog_cat_image.unsqueeze(0).shape)
output = model(dog_cat_image.unsqueeze(0))
print_top_classes(output)

# cat - the predicted class
cat = generate_visualization(dog_cat_image)

# dog
# generate visualization for class 243: 'bull mastiff'
dog = generate_visualization(dog_cat_image, class_index=1)

axs[1].imshow(cat);
axs[1].axis('off');
axs[2].imshow(dog);
axs[2].axis('off');

plt.imshow(cat)
plt.savefig('foo2.png')
