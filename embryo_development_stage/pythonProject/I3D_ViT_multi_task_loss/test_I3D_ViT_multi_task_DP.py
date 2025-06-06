import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from collections import OrderedDict
import torch
torch.set_printoptions(precision=3, sci_mode=False)
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from pythonProject.dataset import Embryo_Dataset_Time_Lapse_I3D_ViT_Multi_task
from I3D_ViT_multi_task_loss_model import I3D_ViT_Multi_Task


def main():

    folder_path = "/path/to/your/dataset/embryo_time_lapse/"
    weight_path =  "/path/to/your/weights.pth"
    device = torch.device("cuda")
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

    model = I3D_ViT_Multi_Task(num_classes=16, pre_trained_weight=None).to(device)
    state_dict = torch.load(weight_path, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.endswith('.scale'):
            k = k.replace('.scale', '.weight')
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    # confusion_matrix = np.zeros((15, 15))

    all_test_num = 0
    all_acc = 0
    all_acc_dp = 0
    confusion_matrix = [[0 for i in range(15)] for j in range(15)]

    with torch.no_grad():
        for i in range(425, 524+1):
            print(f"Video:{i}")
            acc = 0.0
            acc_dp = 0.0
            predict_y = torch.tensor([]).to(device)
            lable_dp = torch.tensor([]).to(device)
            test_dataset = Embryo_Dataset_Time_Lapse_I3D_ViT_Multi_task(folder_path, video_num=(i, i),
                                                              transform=data_transform["test"])
            test_num = len(test_dataset)
            all_test_num += test_num
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                          batch_size=batch_size, shuffle=False,
                                                          num_workers=nw, drop_last=False)
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels, test_frame_id = test_data
                outputs = model(test_images.to(device), test_frame_id.to(device))
                predict_class = torch.max(outputs[:,:15], dim=1)[1]
                acc += torch.eq(predict_class, test_labels.to(device)).sum().item()
                predict_y = torch.cat((predict_y,outputs[:,:15]),dim=0)
                lable_dp = torch.cat((lable_dp,test_labels.to(device)),dim=0)
                test_bar.desc = "test"
            all_acc += acc
            test_accuracy = acc / test_num

            T, C = predict_y.shape[0], 15

            probs = F.softmax(predict_y, dim=1)
            log_probs = torch.log(probs)

            dp = torch.full((T, C), float('-inf'))
            backtrack = torch.zeros((T, C), dtype=torch.long)

            dp[0, :] = log_probs[0, :]

            for t in range(1, T):
                for c in range(C):
                    valid_prev = dp[t - 1, :c + 1]
                    best_prev = torch.max(valid_prev)
                    dp[t, c] = best_prev + log_probs[t, c]
                    backtrack[t, c] = torch.argmax(valid_prev)

            best_last = torch.argmax(dp[-1, :])
            predicted_sequence = [best_last.item()]
            for t in range(T - 1, 0, -1):
                best_last = backtrack[t, best_last]
                predicted_sequence.append(best_last.item())

            predicted_sequence.reverse()  # 翻转为正序
            # print("predicted_sequence",predicted_sequence)
            for _ in range(len(lable_dp)):
                confusion_matrix[int(lable_dp[_].item())][int(predicted_sequence[_])] += 1
                if predicted_sequence[_] == lable_dp[_].item():
                    acc_dp += 1
            all_acc_dp += acc_dp
            test_accurate_dp = acc_dp / test_num
            print("test_accuracy before DP:", test_accuracy)
            print("test_accuracy after DP:", test_accurate_dp)

        for i in range(15):
            for j in range(15):
                if j == 14:
                    print(f"{int(confusion_matrix[i][j]):7d}\n")
                else:
                    print(f"{int(confusion_matrix[i][j]):7d}", end='')
        all_test_accuracy = all_acc / all_test_num
        all_test_accuracy_dp = all_acc_dp / all_test_num
        print("all_test_accuracy", all_test_accuracy)
        print("all_test_accuracy_dp", all_test_accuracy_dp)
        print('Finished testing')




if __name__ == '__main__':
    main()
