import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models import VisionTransformer


class FrozenBN(nn.Module):
    def __init__(self, num_channels, momentum=0.1, eps=1e-5):
        super(FrozenBN, self).__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.params_set = False

    def set_params(self, scale, bias, running_mean, running_var):
        self.register_buffer('scale', scale)
        self.register_buffer('bias', bias)
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)
        self.params_set = True

    def forward(self, x):
        assert self.params_set, 'model.set_params(...) must be called before the forward pass'
        return torch.batch_norm(x, self.scale, self.bias, self.running_mean, self.running_var, False, self.momentum,
                                self.eps, torch.backends.cudnn.enabled)

    def __repr__(self):
        return 'FrozenBN(%d)' % self.num_channels


def freeze_bn(m, name):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.BatchNorm3d:
            frozen_bn = FrozenBN(target_attr.num_features, target_attr.momentum, target_attr.eps)
            frozen_bn.set_params(target_attr.weight.data, target_attr.bias.data, target_attr.running_mean,
                                 target_attr.running_var)
            setattr(m, attr_str, frozen_bn)
    for n, ch in m.named_children():
        freeze_bn(ch, n)


# -----------------------------------------------------------------------------------------------#

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1),
                               padding=(temp_conv, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        outplanes = planes * 4
        self.nl = NonLocalBlock(outplanes, outplanes, outplanes // 2) if use_nl else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.nl is not None:
            out = self.nl(out)

        return out


class NonLocalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super(NonLocalBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

        self.theta = nn.Conv3d(dim_in, dim_inner, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.phi = nn.Conv3d(dim_in, dim_inner, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.g = nn.Conv3d(dim_in, dim_inner, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        self.out = nn.Conv3d(dim_inner, dim_out, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.bn = nn.BatchNorm3d(dim_out)

    def forward(self, x):
        residual = x

        batch_size = x.shape[0]
        mp = self.maxpool(x)
        theta = self.theta(x)
        phi = self.phi(mp)
        g = self.g(mp)

        theta_shape_5d = theta.shape
        theta, phi, g = theta.view(batch_size, self.dim_inner, -1), phi.view(batch_size, self.dim_inner, -1), g.view(
            batch_size, self.dim_inner, -1)

        theta_phi = torch.bmm(theta.transpose(1, 2), phi)  # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
        theta_phi_sc = theta_phi * (self.dim_inner ** -.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.view(theta_shape_5d)

        out = self.out(t)
        out = self.bn(out)

        out = out + residual
        return out


# -----------------------------------------------------------------------------------------------#

class I3Res50_revise(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, use_nl=False):
        super(I3Res50_revise, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.conv2 = nn.Conv2d(1024, 768, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(768)

        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0] != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1),
                          stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0], False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i],
                                i % nonlocal_mod == nonlocal_mod - 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # batch,3,7,224,224 -> batch,64,4,112,112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)  # batch,64,4,112,112 -> batch,64,2,55,55
        x = self.layer1(x)    # batch,64,2,55,55 -> batch,256,2,55,55
        x = self.maxpool2(x)    # batch,256,2,55,55 -> batch,256,1,55,55
        x = self.layer2(x)    # batch,256,1,55,55 -> batch,512,1,28,28
        x = self.layer3(x)    # batch,512,1,28,28 -> batch,1024,1,14,14
        x = x.squeeze(2)    # batch,1024,1,14,14 -> batch,1024,14,14
        x = self.conv2(x)   # batch,1024,14,14 -> batch,768,14,14
        x = self.bn2(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2) # batch,768,14,14 -> batch,768,196 -> batch,196,768
        return x

def i3_res50(num_classes, pre_trained_weight):
    net = I3Res50_revise(num_classes=num_classes, use_nl=False)
    if pre_trained_weight:
        pretrained_dict = torch.load(pre_trained_weight)

        model_dict = net.state_dict()

        # 过滤掉不匹配的权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

        # 更新当前模型的权重
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        freeze_bn(net, "net") # Only needed for finetuning. For validation, .eval() works.
    return net

class I3D_ViT(nn.Module):
    def __init__(self, num_classes, pre_trained_weight, use_nl=False):
        super(I3D_ViT, self).__init__()
        # self.I3Res50 = I3Res50(num_classes=num_classes, use_nl=use_nl)
        self.I3Res50 = i3_res50(num_classes=num_classes, pre_trained_weight=pre_trained_weight)
        self.ViT = torchvision.models.vit_b_16(weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.ViT.conv_proj = nn.Identity()

        def new_forward(self, x: torch.Tensor):
            # 去掉 x = self._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.encoder(x)

            # Classifier "token" as used by standard language architectures
            x = x[:, 0]

            x = self.heads(x)
            return x

        self.ViT.forward = new_forward.__get__(self.ViT, VisionTransformer)
        in_features = self.ViT.heads.head.in_features
        self.ViT.heads.head = torch.nn.Linear(in_features, num_classes)
        torch.nn.init.trunc_normal_(self.ViT.heads.head.weight, std=0.02)
        if self.ViT.heads.head.bias is not None:
            torch.nn.init.zeros_(self.ViT.heads.head.bias)

    def forward(self, x):
        x = self.I3Res50(x)
        x = self.ViT(x)
        return x

# -----------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    pre_trained_weight_path = 'D:\software\Pycharm\project\embryo_development_stage\pythonProject\I3D\i3d_baseline_32x2_IN_pretrain_400k.pth'
    model = I3D_ViT(num_classes=15, pre_trained_weight=pre_trained_weight_path)
    print("Checking I3D weights:")
    for name, param in model.I3Res50.named_parameters():
        print(name, param.requires_grad)

    # 检查 ViT 的权重是否加载成功
    print("Checking ViT weights:")
    for name, param in model.ViT.named_parameters():
        print(name, param.requires_grad)
    inp = torch.rand(4, 3, 7, 224, 224)
    pred = model(inp)
    print(pred.shape)
    # net = i3_res50(15, False)
    # # (Batch Size=4, Channels=3, Temporal Dimension=32, Height=224, Width=224)
    # inp = {'frames': torch.rand(4, 3, 7, 224, 224)}
    # pred = net(inp)


