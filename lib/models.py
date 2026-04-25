import torch
import torch.nn as nn
import torchvision
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torchvision.models import ViT_B_32_Weights


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_cluster, no_prototype=False):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        hidden_mlp, output_dim = 256, 128
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, output_dim),
        )
        self.prototype_layer = nn.Linear(output_dim, num_cluster, bias=False)
        self.fc = nn.Linear(output_dim, num_classes)


    def forward(self, x, feature_flag=False):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        out = self.projection_head(out)
        out = nn.functional.normalize(out, dim=1, p=2) # normalize the feature via f/|f|
        if feature_flag:
            return out, self.fc(out)
        # if prototype_flag:
        #     # out = self.projection_head(out)
        #     # out = nn.functional.normalize(out, dim=1, p=2) # normalize the feature via f/|f|
        #     return out, self.prototype_layer(out)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


class CNN(nn.Module):
    def __init__(self, num_classes, num_cluster=None, no_prototype=False):
        super(CNN, self).__init__()
        num_cluster = num_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(3, 3), padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding="same")
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.flatten = nn.Flatten()

        hidden_mlp, output_dim = 256, 128
        self.projection_head = nn.Sequential(
            nn.Linear(256 * 12 * 4, hidden_mlp),
            nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.BatchNorm1d(hidden_mlp),
            nn.Linear(hidden_mlp, output_dim),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )

        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, x, feature_flag=False, prototype_flag=False):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = self.flatten(x)
        self.act = x
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)  # normalize the feature via f/|f|
        if feature_flag:
            return x, self.fc(x)

        # Decode the hidden state of the last time step
        x = self.fc(x)

        return x

    def get_act(self):
        return self.act


class My_vit_b_32(nn.Module):  # Inherit from nn.Module
    def __init__(self, num_classes):
        super(My_vit_b_32, self).__init__()  # Initialize nn.Module
        self.backbone = torchvision.models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)

        # Define projection head
        project_indim, hidden_mlp, output_dim = 1000, 256, 128
        self.projection_head = nn.Sequential(
            nn.Linear(project_indim, hidden_mlp),
            nn.ReLU(),
            nn.Linear(hidden_mlp, output_dim),
        )

        # Define final classifier
        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, x, feature_flag=False):
        # Extract features using the backbone
        out = self.backbone(x)

        # Apply ReLU activation
        out = nn.ReLU()(out)

        # Save activation for later use
        self.act = out

        # Pass through projection head
        out = self.projection_head(out)

        # Normalize the output
        out = nn.functional.normalize(out, dim=1, p=2)  # normalize the feature via f/|f|

        if feature_flag:
            return out, self.fc(out)  # Return features and classification logits

        # Decode the hidden state of the last time step
        out = self.fc(out)

        return out

    def get_act(self):
        return self.act


def gen_model(poison_type, number_classes, num_cluster, model_name):
    if 'adaptivecifar10' in poison_type or 'freq' in poison_type or poison_type == 'pattern' \
            or 'wanet' in poison_type:
        model = ResNet18(number_classes)
    elif 'adaptiveattack' in poison_type:
        if model_name == 'cnn':
            model = ResNet18(number_classes)
        elif model_name == 'transformer':
            print(model_name)
            model = My_vit_b_32(number_classes)
        elif model_name == 'vgg19':
            print('vgg19')
            model = vgg19(number_classes)
        elif model_name == 'efficient':
            print('efficient')
            model = EfficientNet(number_classes)
        else:
            print('no implememtation')
            exit(-1)
    elif poison_type == 'corruptencoder' or poison_type == 'depud' or poison_type== 'adp_corrupt':
        model = ResNet18(number_classes) # EfficientNet(number_classes)
    elif poison_type== 'blto':
        if model_name == 'cnn':
            print('model is ResNet')
            model = ResNet18(num_classes=number_classes)#, project_indim=512)   # #EfficientNet(number_classes) #
        elif model_name == 'efficient':
            print('efficient')
            model = EfficientNet(number_classes)
        elif model_name == 'transformer':
            print(model_name)
            model = My_vit_b_32(number_classes)
        else:
            print('no implememtation')
            exit(-1)
    else:
        if model_name == 'rnn':
            input_size, hidden_size, num_layers = 40, 768, 3
            model = RNN(input_size, hidden_size, num_layers, number_classes, num_cluster)
        elif model_name == 'cnn':
            model = CNN(number_classes, num_cluster)
        else:
            raise ValueError("no model name {}".format(model_name))

    return model


"""Two contrastive encoders"""
class TFCModel(nn.Module):
    def __init__(self, TSlength_aligned=178):
        super(TFCModel, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(TSlength_aligned, dim_feedforward=2*TSlength_aligned, nhead=2, )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        encoder_layers_f = TransformerEncoderLayer(TSlength_aligned, dim_feedforward=2*TSlength_aligned,nhead=2,)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )


    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


"""Downstream classifier only used in finetuning"""
class target_classifier(nn.Module):
    def __init__(self, num_classes_target):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=16, project_indim=2048):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        hidden_mlp, output_dim = 256, 128
        self.projection_head = nn.Sequential(
            nn.Linear(project_indim, hidden_mlp),
            nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.BatchNorm1d(hidden_mlp),
            nn.Linear(hidden_mlp, output_dim),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.fc = nn.Linear(output_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature_flag=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.contiguous().view(out.size(0), -1)
        out = nn.ReLU()(out)
        self.act = out
        out = self.projection_head(out)
        out = nn.functional.normalize(out, dim=1, p=2)  # normalize the feature via f/|f|
        if feature_flag:
            return out, self.fc(out)
        # Decode the hidden state of the last time step
        out = self.fc(out)

        return out

    def first_conv1(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = out.view(x.shape[0], -1)
        out = nn.functional.normalize(out, dim=1, p=2)

        return out

    def get_act(self):
        return self.act

def ResNet18(num_classes, project_indim=2048):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, project_indim)


###########efficientNet###############

def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNet, self).__init__()
        cfg = {
            'num_blocks': [1, 2, 2, 3, 3, 4, 1],
            'expansion': [1, 6, 6, 6, 6, 6, 6],
            'out_channels': [16, 24, 40, 80, 112, 192, 320],
            'kernel_size': [3, 3, 5, 3, 5, 5, 3],
            'stride': [1, 2, 2, 2, 1, 2, 1],
            'dropout_rate': 0.2,
            'drop_connect_rate': 0.2,
        }
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        hidden_mlp, output_dim = 256, 128
        self.projection_head = nn.Sequential(
            nn.Linear(cfg['out_channels'][-1], hidden_mlp),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_mlp),
            nn.Linear(hidden_mlp, output_dim),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.fc = nn.Linear(128, num_classes)


    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, feature_flag=False):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        out = nn.ReLU()(out)
        self.act = out
        out = self.projection_head(out)
        out = nn.functional.normalize(out, dim=1, p=2)  # normalize the feature via f/|f|
        if feature_flag:
            return out, self.fc(out)
        # Decode the hidden state of the last time step
        out = self.fc(out)

        return out

    def get_act(self):
        return self.act
#################efficientNet#################

#################    VGG    ##################
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        hidden_mlp, output_dim = 256, 128
        self.projection_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_mlp),
            nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.BatchNorm1d(hidden_mlp),
            nn.Linear(hidden_mlp, output_dim),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.fc = nn.Linear(output_dim, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, feature_flag=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = x
        out = nn.ReLU()(out)
        out = self.projection_head(out)
        out = nn.functional.normalize(out, dim=1, p=2)  # normalize the feature via f/|f|
        if feature_flag:
            return out, self.fc(out)
        # Decode the hidden state of the last time step
        out = self.fc(out)

        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    return model

def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)
#################    VGG    ##################


################ VisTransformer ##################


################ VisTransformer ##################
if __name__ == '__main__':
    cnn_model = My_vit_b_32(num_classes=10) #ResNet18(num_classes=10, project_indim=512)
    input = torch.randn(128, 3, 224, 224)
    output = cnn_model(input)
    print(output.shape)