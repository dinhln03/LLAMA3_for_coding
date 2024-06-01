import network
import torch

if __name__ == '__main__':
    net = network.modeling.__dict__['deeplabv3plus_resnet50']()
    print(net)
    
    input=torch.FloatTensor(2,3,512,512)
    output=net(input)
    print(output.shape)