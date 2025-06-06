import os
import csv
import re

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
#  Average frame count is 415 in one video

class Embryo_Dataset_Time_Lapse_I3D(Dataset):
    def __init__(self, folder_path, video_num: tuple, transform=None):  # video_num   The clip of video to train/test
        self.transform = transform
        self.phases = {"tPB2": 0, "tPNa": 1, "tPNf": 2, "t2": 3, "t3": 4, "t4": 5, "t5": 6, "t6": 7, "t7": 8, "t8": 9,"t9+": 10, "tM": 11, "tSB": 12, "tB": 13, "tEB": 14}
        self.all_image_path = []
        self.multi_focus_img_path = []
        self.label = []
        self.reset = []
        self.anno = os.path.join(folder_path, "embryo_dataset_annotations_revise")
        self.video_name = sorted(os.listdir(os.path.join(folder_path, "embryo_dataset_annotations_revise")))
        for _ in range(len(self.video_name)):
            self.video_name[_] = self.video_name[_].replace('_phases.csv', '')
        for i in range(video_num[1] - video_num[0] + 1):
            self.one_video = sorted(os.listdir(
                os.path.join(folder_path, "embryo_dataset_F0", self.video_name[i + video_num[0] - 1])), key=lambda x: int(re.search(r'RUN(\d+)(?=\.[^.]+$)', x).group(1)))
            with open(os.path.join(self.anno, self.video_name[i + video_num[0] - 1] + '_phases.csv'), 'r') as f:
                reader = csv.reader(f)
                for _, row in enumerate(reader):
                    if row[0] == 'tHB':
                        continue
                    for j in range(int(row[2]) - int(row[1]) + 1):
                        self.multi_focus_img_path = []
                        for k in range(7):
                            self.multi_focus_img_path.append(
                                os.path.join(folder_path, f"embryo_dataset_F{-45 + 15 * k}",self.video_name[i + video_num[0] - 1], self.one_video[int(row[1]) + j - 1]))
                        self.all_image_path.append(self.multi_focus_img_path)
                        self.label.append(self.phases[row[0]])
                        if _ == 0 and j == 0:
                            self.reset.append(True)
                        else:
                            self.reset.append(False)

    def get_class_distribution(self):
        class_num = [0] * 15
        for label in self.label:
            class_num[label] += 1
        return class_num

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = []
        for _ in range(7):
            image.append(self.transform(Image.open(self.all_image_path[idx][_])).repeat(3,1,1)) # 1,224,224 -> 3,224,224
        image = torch.stack(image, dim=1)  # 3,224,224 -> 3,7,224,224
        label = self.label[idx]
        reset = self.reset[idx]
        # return image, label, reset
        return image, label


# dataset for ResNet50_multi_focus
class Embryo_Dataset_Time_Lapse_ResNEt50_multi_focus(Dataset):
    def __init__(self, folder_path, video_num: tuple, transform=None):  # video_num   The clip of video to train/test
        self.transform = transform
        self.phases = {"tPB2": 0, "tPNa": 1, "tPNf": 2, "t2": 3, "t3": 4, "t4": 5, "t5": 6, "t6": 7, "t7": 8, "t8": 9,"t9+": 10, "tM": 11, "tSB": 12, "tB": 13, "tEB": 14}
        self.all_image_path = []
        self.multi_focus_img_path = []
        self.label = []
        self.reset = []
        self.anno = os.path.join(folder_path, "embryo_dataset_annotations_revise")
        self.video_name = sorted(os.listdir(os.path.join(folder_path, "embryo_dataset_annotations_revise")))
        for _ in range(len(self.video_name)):
            self.video_name[_] = self.video_name[_].replace('_phases.csv', '')
        for i in range(video_num[1] - video_num[0] + 1):
            self.one_video = sorted(os.listdir(
                os.path.join(folder_path, "embryo_dataset_F0", self.video_name[i + video_num[0] - 1])), key=lambda x: int(re.search(r'RUN(\d+)(?=\.[^.]+$)', x).group(1)))
            with open(os.path.join(self.anno, self.video_name[i + video_num[0] - 1] + '_phases.csv'), 'r') as f:
                reader = csv.reader(f)
                for _, row in enumerate(reader):
                    if row[0] == 'tHB':
                        continue
                    for j in range(int(row[2]) - int(row[1]) + 1):
                        self.multi_focus_img_path = []
                        for k in range(7):
                            self.multi_focus_img_path.append(
                                os.path.join(folder_path, f"embryo_dataset_F{-45 + 15 * k}",self.video_name[i + video_num[0] - 1], self.one_video[int(row[1]) + j - 1]))
                        self.all_image_path.append(self.multi_focus_img_path)
                        self.label.append(self.phases[row[0]])
                        if _ == 0 and j == 0:
                            self.reset.append(True)
                        else:
                            self.reset.append(False)

    def get_class_distribution(self):
        class_num = [0] * 15
        for label in self.label:
            class_num[label] += 1
        return class_num

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = []
        for _ in range(7):
            image.append(self.transform(Image.open(self.all_image_path[idx][_]))) # 1,224,224
        image = torch.stack(image, dim=0).squeeze(dim=1)  # 1,224,224 -> 7,1,224,224 -> 7,224,224
        label = self.label[idx]
        reset = self.reset[idx]
        # return image, label, reset
        return image, label

# dataset for I3D_F0
class Embryo_Dataset_Time_Lapse_I3D_F0(Dataset):
    def __init__(self, folder_path, video_num: tuple, transform=None):  # video_num   The clip of video to train/test
        self.transform = transform
        self.phases = {"tPB2": 0, "tPNa": 1, "tPNf": 2, "t2": 3, "t3": 4, "t4": 5, "t5": 6, "t6": 7, "t7": 8, "t8": 9,"t9+": 10, "tM": 11, "tSB": 12, "tB": 13, "tEB": 14}
        self.all_image_path = []
        self.multi_focus_img_path = []
        self.label = []
        self.reset = []
        self.anno = os.path.join(folder_path, "embryo_dataset_annotations_revise")
        self.video_name = sorted(os.listdir(os.path.join(folder_path, "embryo_dataset_annotations_revise")))
        for _ in range(len(self.video_name)):
            self.video_name[_] = self.video_name[_].replace('_phases.csv', '')
        for i in range(video_num[1] - video_num[0] + 1):
            self.one_video = sorted(os.listdir(
                os.path.join(folder_path, "embryo_dataset_F0", self.video_name[i + video_num[0] - 1])), key=lambda x: int(re.search(r'RUN(\d+)(?=\.[^.]+$)', x).group(1)))
            with open(os.path.join(self.anno, self.video_name[i + video_num[0] - 1] + '_phases.csv'), 'r') as f:
                reader = csv.reader(f)
                for _, row in enumerate(reader):
                    if row[0] == 'tHB':
                        continue
                    for j in range(int(row[2]) - int(row[1]) + 1):
                        self.all_image_path.append(os.path.join(folder_path, "embryo_dataset_F0",self.video_name[i + video_num[0] - 1], self.one_video[int(row[1]) + j - 1]))
                        self.label.append(self.phases[row[0]])

    def __len__(self):
        return len(self.label)

    def get_class_distribution(self):
        class_num = [0] * 15
        for label in self.label:
            class_num[label] += 1
        return class_num

    def __getitem__(self, idx):
        image = []
        for _ in range(7):
            image.append(self.transform(Image.open(self.all_image_path[idx])).repeat(3, 1, 1))  # 1,224,224 -> 3,224,224
        image = torch.stack(image, dim=1)  # 3,224,224 -> 3,7,224,224
        label = self.label[idx]
        return image, label


class Embryo_Dataset_Time_Lapse_F0(Dataset):
    def __init__(self, folder_path, video_num: tuple, transform=None):  # video_num   The clip of video to train/test
        self.transform = transform
        self.phases = {"tPB2": 0, "tPNa": 1, "tPNf": 2, "t2": 3, "t3": 4, "t4": 5, "t5": 6, "t6": 7, "t7": 8, "t8": 9,"t9+": 10, "tM": 11, "tSB": 12, "tB": 13, "tEB": 14}
        self.all_image_path = []
        self.multi_focus_img_path = []
        self.label = []
        self.reset = []
        self.anno = os.path.join(folder_path, "embryo_dataset_annotations_revise")
        self.video_name = sorted(os.listdir(os.path.join(folder_path, "embryo_dataset_annotations_revise")))
        for _ in range(len(self.video_name)):
            self.video_name[_] = self.video_name[_].replace('_phases.csv', '')
        for i in range(video_num[1] - video_num[0] + 1):
            self.one_video = sorted(os.listdir(
                os.path.join(folder_path, "embryo_dataset_F0", self.video_name[i + video_num[0] - 1])), key=lambda x: int(re.search(r'RUN(\d+)(?=\.[^.]+$)', x).group(1)))
            with open(os.path.join(self.anno, self.video_name[i + video_num[0] - 1] + '_phases.csv'), 'r') as f:
                reader = csv.reader(f)
                for _, row in enumerate(reader):
                    if row[0] == 'tHB':
                        continue
                    for j in range(int(row[2]) - int(row[1]) + 1):
                        self.all_image_path.append(os.path.join(folder_path, "embryo_dataset_F0",self.video_name[i + video_num[0] - 1], self.one_video[int(row[1]) + j - 1]))
                        self.label.append(self.phases[row[0]])


    def __len__(self):
        return len(self.label)

    def get_class_distribution(self):
        class_num = [0] * 15
        for label in self.label:
            class_num[label] += 1
        return class_num

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.all_image_path[idx])).repeat(3,1,1) # (1,500,500) -> (3,500,500)
        label = self.label[idx]
        return image, label

class Embryo_Dataset_Time_Lapse_F0_Multi_Task(Dataset):
    def __init__(self, folder_path, video_num: tuple, transform=None):  # video_num   The clip of video to train/test
        self.transform = transform
        self.phases = {"tPB2": 0, "tPNa": 1, "tPNf": 2, "t2": 3, "t3": 4, "t4": 5, "t5": 6, "t6": 7, "t7": 8, "t8": 9,"t9+": 10, "tM": 11, "tSB": 12, "tB": 13, "tEB": 14}
        self.all_image_path = []
        self.multi_focus_img_path = []
        self.label = []
        self.frame_id = []
        self.reset = []
        self.anno = os.path.join(folder_path, "embryo_dataset_annotations_revise")
        self.video_name = sorted(os.listdir(os.path.join(folder_path, "embryo_dataset_annotations_revise")))
        for _ in range(len(self.video_name)):
            self.video_name[_] = self.video_name[_].replace('_phases.csv', '')
        for i in range(video_num[1] - video_num[0] + 1):
            self.one_video = sorted(os.listdir(
                os.path.join(folder_path, "embryo_dataset_F0", self.video_name[i + video_num[0] - 1])), key=lambda x: int(re.search(r'RUN(\d+)(?=\.[^.]+$)', x).group(1)))
            with open(os.path.join(self.anno, self.video_name[i + video_num[0] - 1] + '_phases.csv'), 'r') as f:
                reader = csv.reader(f)
                for _, row in enumerate(reader):
                    if row[0] == 'tHB':
                        continue
                    for j in range(int(row[2]) - int(row[1]) + 1):
                        self.all_image_path.append(os.path.join(folder_path, "embryo_dataset_F0",self.video_name[i + video_num[0] - 1], self.one_video[int(row[1]) + j - 1]))
                        self.label.append(self.phases[row[0]])
                        self.frame_id.append(int(row[1]) + j)


    def __len__(self):
        return len(self.label)

    def get_class_distribution(self):
        class_num = [0] * 15
        for label in self.label:
            class_num[label] += 1
        return class_num

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.all_image_path[idx])).repeat(3,1,1) # (1,500,500) -> (3,500,500)
        label = self.label[idx]
        frame_id = torch.tensor([self.frame_id[idx] / 415.0])
        return image, label, frame_id


class Embryo_Dataset_Time_Lapse_I3D_ViT_Multi_task(Dataset):
    def __init__(self, folder_path, video_num: tuple, transform=None):  # video_num   The clip of video to train/test
        self.transform = transform
        self.phases = {"tPB2": 0, "tPNa": 1, "tPNf": 2, "t2": 3, "t3": 4, "t4": 5, "t5": 6, "t6": 7, "t7": 8, "t8": 9,"t9+": 10, "tM": 11, "tSB": 12, "tB": 13, "tEB": 14}
        self.all_image_path = []
        self.multi_focus_img_path = []
        self.label = []
        self.reset = []
        self.frame_id = []
        self.anno = os.path.join(folder_path, "embryo_dataset_annotations_revise")
        self.video_name = sorted(os.listdir(os.path.join(folder_path, "embryo_dataset_annotations_revise")))
        for _ in range(len(self.video_name)):
            self.video_name[_] = self.video_name[_].replace('_phases.csv', '')
        for i in range(video_num[1] - video_num[0] + 1):
            self.one_video = sorted(os.listdir(
                os.path.join(folder_path, "embryo_dataset_F0", self.video_name[i + video_num[0] - 1])), key=lambda x: int(re.search(r'RUN(\d+)(?=\.[^.]+$)', x).group(1)))
            with open(os.path.join(self.anno, self.video_name[i + video_num[0] - 1] + '_phases.csv'), 'r') as f:
                reader = csv.reader(f)
                for _, row in enumerate(reader):
                    if row[0] == 'tHB':
                        continue
                    for j in range(int(row[2]) - int(row[1]) + 1):
                        self.multi_focus_img_path = []
                        for k in range(7):  # 遍历7焦段
                            self.multi_focus_img_path.append(
                                os.path.join(folder_path, f"embryo_dataset_F{-45 + 15 * k}",self.video_name[i + video_num[0] - 1], self.one_video[int(row[1]) + j - 1]))
                        self.all_image_path.append(self.multi_focus_img_path)
                        self.label.append(self.phases[row[0]])
                        self.frame_id.append(int(row[1]) + j)
                        if _ == 0 and j == 0:
                            self.reset.append(True)
                        else:
                            self.reset.append(False)

    def get_class_distribution(self):
        class_num = [0] * 15
        for label in self.label:
            class_num[label] += 1
        return class_num

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = []
        for _ in range(7):
            image.append(self.transform(Image.open(self.all_image_path[idx][_])).repeat(3,1,1)) # 1,224,224 -> 3,224,224
        image = torch.stack(image, dim=1)  # 3,224,224 -> 3,7,224,224
        label = self.label[idx]
        frame_id = torch.tensor([self.frame_id[idx] / 415.0])
        return image, label, frame_id

