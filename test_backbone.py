import torch
from vietocr.model.backbone.cnn import CNN
import torch.nn as nn
import time
import torchvision
from torchsummary import summary


def _cnn_backbone(img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

device= 'cuda'
#cnn_config_vgg19 = {'ss': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]], 'ks': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]], 'hidden': 256}
#cnn = CNN('vgg19_bn', **cnn_config_vgg19)
#cnn_config_vgg11 = {'ss': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]], 'ks': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]], 'hidden': 256}
#cnn = CNN('vgg11_bn', **cnn_config_vgg11)
#cnn_config_resnet50 = {'ss': [[2, 2], [2, 1], [2, 1], [2, 1], [1, 1]], 'hidden': 256}    
#cnn = CNN('resnet50', **cnn_config_resnet50)
#cnn_config_resnet18 = {'ss': [[2, 2], [2, 1], [2, 1], [2, 1], [1, 1]], 'hidden': 256}  
#cnn = CNN('resnet18', **cnn_config_resnet18)
#cnn,_ = _cnn_backbone(img_channel=3, img_height=32, img_width=220, leaky_relu=True)
#cnn = CNN('mobilenetv1_0.25')
cnn_config_mobilenetv2 = {'ss': [[2, 2], [2, 1], [2, 1], [2, 1], [1, 1]], 'hidden': 256}  
cnn = CNN('mobilenetv2', **cnn_config_mobilenetv2)
#cnn = torchvision.models.mobilenet_v2()
#cnn = torch.nn.Sequential(*(list(cnn.children())[0][:7]))
#cnn = torchvision.models.shufflenet_v2_x1_5()
#cnn = torch.nn.Sequential(*(list(cnn.children())[0][:7]))
#cnn = torchvision.models.mnasnet1_0()
#cnn = torch.nn.Sequential(*(list(cnn.children())[0][:9]))
#print(cnn)
cnn = cnn.to(device)
if device == "cuda":
  summary(cnn, (3, 32, 1024))
torch.cuda.synchronize()
t1 = time.time()
x = torch.zeros(1, 3, 32, 1024).to(device)
y = cnn(x)
print(y.shape)
torch.cuda.synchronize()
print(time.time()-t1)

