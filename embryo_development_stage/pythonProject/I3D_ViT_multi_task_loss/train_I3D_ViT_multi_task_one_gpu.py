import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import ssl
import argparse
from pythonProject.multi_train_utils.distributed_utils import init_distributed_mode, cleanup, reduce_value, is_main_process
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from pythonProject.dataset import Embryo_Dataset_Time_Lapse_I3D_ViT_Multi_task
from I3D_ViT_multi_task_loss_model import I3D_ViT_Multi_Task

def main(args):

    txt_path = args.txt_path
    class_txt_path = args.class_txt_path
    save_path = args.save_path
    save_path_last = args.save_path_last
    folder_path = args.folder_path
    device = torch.device(args.device)
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    pre_trained_weight_path = '/path/to/your/i3d_baseline_32x2_IN_pretrain_400k.pth'
    checkpoint_path = ""

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

    print("begin loading dataset")
    train_dataset = Embryo_Dataset_Time_Lapse_I3D_ViT_Multi_task(folder_path, (1,324), transform=data_transform["train"])

    train_num = len(train_dataset)
    train_class_num = train_dataset.get_class_distribution()

    test_dataset = Embryo_Dataset_Time_Lapse_I3D_ViT_Multi_task(folder_path, video_num=(325,424), transform=data_transform["test"])

    test_num = len(test_dataset)
    test_class_num = test_dataset.get_class_distribution()
    print("using {} time-lapse images for training, every class num{}. {} time-lapse images for testing, every class num{}.".format(train_num, train_class_num,
                                                                           test_num, test_class_num))

    # number of workers
    nw = min(os.cpu_count(),16)
    if is_main_process():
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               shuffle=True,
                                               num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            pin_memory=True,
                                            shuffle=False,
                                            num_workers=nw)

    ssl._create_default_https_context = ssl._create_unverified_context
    net = I3D_ViT_Multi_Task(16, pre_trained_weight_path).to(device)
    if is_main_process():
        with open(txt_path, "w") as f:
            f.write("Epoch\tLoss\tClass_Loss\tFrame_Loss\tTrain_acc\tTest_acc\tLearning_rate\n")
            print("File write completed.")

    # define loss function
    loss_function1 = nn.CrossEntropyLoss()
    loss_function2 = nn.MSELoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=0.00005)
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        acc = torch.zeros(1).to(device)
        mean_loss1 = torch.zeros(1).to(device)
        mean_loss2 = torch.zeros(1).to(device)
        mean_loss = torch.zeros(1).to(device)
        if is_main_process():
            train_loader = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_loader):
            images, labels, frame_id = data
            optimizer.zero_grad()
            logits = net(images.to(device), frame_id.to(device))
            predict_y = torch.max(logits[:,:15], dim=1)[1]
            acc += torch.eq(predict_y, labels.to(device)).sum()
            loss1 = loss_function1(logits[:,:15], labels.to(device))
            loss2 = 5 * loss_function2(logits[:,15], frame_id.squeeze(-1).to(device))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            mean_loss1 = (mean_loss1 * step + loss1) / (step + 1)  # update mean losses
            mean_loss2 = (mean_loss2 * step + loss2) / (step + 1)  # update mean losses
            mean_loss = (mean_loss * step + loss) / (step + 1)  # update mean losses
            # print statistics
            if is_main_process():
                train_loader.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_acc = acc.item() / train_num
        if is_main_process():
            print(f"train acc{train_acc:.3f} in epoch{epoch+1} ")
        scheduler.step()
        # validate
        acc = torch.zeros(1).to(device)
        confusion_matrix = torch.zeros(15, 15).to(device)
        net.eval()
        with torch.no_grad():
            if is_main_process():
                test_loader = tqdm(test_loader, file=sys.stdout)
            for test_data in test_loader:
                test_images, test_labels, test_frame_id = test_data
                outputs = net(test_images.to(device), test_frame_id.to(device))
                predict_y = torch.max(outputs[:,:15], dim=1)[1]
                acc += torch.eq(predict_y, test_labels.to(device)).sum()
                for i in range(predict_y.shape[0]):
                    confusion_matrix[test_labels[i]][predict_y[i]] += 1
                if is_main_process():
                    test_loader.desc = "test epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        test_accurate = acc.item() / test_num
        if is_main_process():
            print(f'[epoch {epoch + 1}]   test_accuracy: {test_accurate:.4f}')
            with open(txt_path, "a") as file:
                file.write(
                    f"{epoch + 1}\t{mean_loss.item():.4f}\t{mean_loss1.item():.4f}\t{mean_loss2.item():.4f}\t{train_acc:.4f}\t{test_accurate:.4f}\t{optimizer.param_groups[0]['lr']}\n")
            with open(class_txt_path, "a") as file:
                file.write(
                    f"epoch: {epoch + 1}\nloss: {mean_loss.item():.4f}\t loss1: {mean_loss1.item():.4f}\t loss2: {mean_loss2.item():.4f}\t train_acc: {train_acc:.4f}\t test_acc: {test_accurate:.4f}\n")
                file.write(f"Confusion_Matrix\n")
                for i in range(15):
                    for j in range(15):
                        if j == 14:
                            file.write(f"{int(confusion_matrix[i][j].item()):7d}\n")
                        else:
                            file.write(f"{int(confusion_matrix[i][j].item()):7d}")
                file.write(f"Class Accuracy\n")
                for i in range(15):
                    file.write(f"C{i}: {confusion_matrix[i][i].item() / test_class_num[i]:.4f}\n")
            if test_accurate > best_acc:
                best_acc = test_accurate
                torch.save(net.state_dict(), save_path)
            if epoch == epochs - 1:
                torch.save(net.state_dict(), save_path_last)
    cleanup()
    print('Finished Training')



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # the path to save training and testing accuracy
    parse.add_argument('--txt_path', type=str, default="/path/to/output_file_I3D_ViT_multi_task_new_dataset.txt")
    # the path to save every class testing accuracy
    parse.add_argument('--class_txt_path', type=str,
                       default="/path/to/output_file_I3D_ViT_multi_task_confusion_matrix_new_dataset.txt")
    # the path to save the .pth weight file
    parse.add_argument('--save_path', type=str, default='/path/to/weights_I3D_ViT_multi_task_new_dataset.pth')
    # the path to save the last epoch .pth weight file
    parse.add_argument('--save_path_last', type=str,
                       default='/path/to/weights_I3D_ViT_multi_task_last_new_dataset.pth')
    # the path to the dataset
    parse.add_argument('--folder_path', type=str, default="/path/to/your/dataset/embryo_time_lapse/")
    parse.add_argument('--device', type=str, default='cuda:1')
    parse.add_argument('--lr', type=float, default=0.00016)
    parse.add_argument('--batch-size', type=int, default=64)
    parse.add_argument('--epochs', type=int, default=30)
    args = parse.parse_args()

    main(args)
