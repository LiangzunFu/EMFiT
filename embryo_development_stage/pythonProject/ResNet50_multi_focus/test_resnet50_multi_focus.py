import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from collections import OrderedDict
import torch
torch.set_printoptions(precision=3, sci_mode=False)
from torchvision import transforms
from tqdm import tqdm
import torchvision
from pythonProject.dataset import Embryo_Dataset_Time_Lapse_ResNEt50_multi_focus


def main():

    folder_path = "/path/to/your/dataset/embryo_time_lapse/"
    weight_path =  "/path/to/your/weights.pth"
    device = torch.device("cuda:0")
    print("using {} device.".format(device))

    batch_size = 64

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.RandomRotation(degrees=45),
                                     transforms.ColorJitter(brightness=0.2,contrast = 0.2),
                                     transforms.ToTensor(),
                                     transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.04, 0.0, 1.0)),
                                     transforms.Normalize(mean=[0.5], std=[0.5])
                                     ]),
        "test": transforms.Compose([transforms.Resize(size=224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                    ])}

    nw = min(os.cpu_count(), 8)
    print('Using {} dataloader workers every process'.format(nw))
    model = torchvision.models.resnet50()
    model.conv1 = torch.nn.Conv2d(
        in_channels=7,  # change to 7 channels input
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=in_features, out_features=15, bias=True)
    state_dict = torch.load(weight_path, map_location=device)
    model.to(device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.endswith('.scale'):
            k = k.replace('.scale', '.weight')
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v


    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    test_dataset = Embryo_Dataset_Time_Lapse_ResNEt50_multi_focus(folder_path, video_num=(425, 524),
                                                transform=data_transform["test"])
    # test_dataset = Embryo_Dataset_Time_Lapse_F0(folder_path, video_num=(425, 524), transform=data_transform["test"])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)
    # confusion_matrix = np.zeros((15, 15))
    test_num = len(test_dataset)
    test_bar = tqdm(test_loader, file=sys.stdout)
    acc_num = 0
    acc_rate = 0.0
    class_acc = [0] * 15
    class_acc_rate = [0.0] * 15
    confusion_matrix = torch.zeros(15, 15).to(device)
    class_num = test_dataset.get_class_distribution()
    with torch.no_grad():
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = model(test_images.to(device))
            predict_class = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(predict_class, test_labels.to(device)).sum().item()
            for i in range(predict_class.shape[0]):
                confusion_matrix[test_labels[i]][predict_class[i]] += 1
            for i in range(predict_class.shape[0]):
                if predict_class[i] == test_labels[i]:
                    class_acc[test_labels[i].item()] += 1
            test_bar.desc = "test"
        class_acc_rate = [class_acc[i]/class_num[i] for i in range(15)]
        acc_rate = acc_num / test_num
        for i in range(15):
            for j in range(15):
                if j == 14:
                    print(f"{int(confusion_matrix[i][j].item()):7d}\n")
                else:
                    print(f"{int(confusion_matrix[i][j].item()):7d}", end='')

        for i in range(15):
            print(f"class {i} acc: {class_acc_rate[i]:.4f}")
        print(f"total accuracy: {acc_rate:.4f}")
        print('Finished testing')


if __name__ == '__main__':
    main()
