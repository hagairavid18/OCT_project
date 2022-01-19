from __future__ import print_function
import torch.utils.data as data
import torch
from torchvision import transforms as transforms
from pathlib import Path
import cv2 as cv


class Kermany_DataSet(torch.utils.data.Dataset):
    def __init__(self, path, size=(496, 512)):
        # load your dataset (how every you want, this example has the dataset stored in a json file
        # path = "C:/Users/guylu/Desktop/prev_files/Weizmann/OCT/test/AIA 03346 OS 13.01.2020.E2E"
        self.t = transforms.Compose([transforms.ToTensor(), transforms.RandomResizedCrop(size)])
        self.dataset = []
        label = 0
        label_dict = {"NORMAL": 0,
                      "CNV": 1,
                      "DME": 2,
                      "DRUSEN": 3}
        f_1 = lambda x: 0 if "NORMAL" in x else 0
        f_2 = lambda x: 1 if "CNV" in x else 0
        f_3 = lambda x: 2 if "DME" in x else 0
        f_4 = lambda x: 3 if "DRUSEN" in x else 0
        i = 0
        self.labels = []
        for path2 in Path(path).rglob('*.jpeg'):
            # i += 1
            # if i > 200: break
            # path2 = str(path2)
            # image = self.t(cv.imread(path2))
            # label = f_1(path2) + f_2(path2) + f_3(path2) + f_4(path2)
            # self.dataset.append((image, label))

            path2 = str(path2)
            label = f_1(path2) + f_2(path2) + f_3(path2) + f_4(path2)
            self.dataset.append((path2, label))
            self.labels.append(label)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        sample = self.t(cv.imread(sample[0])), sample[1]

        return sample

    def __len__(self):
        return len(self.dataset)

    def get_labels(self):
        return self.labels
