#################################################
# Retrieve robust classifier from:
# https://github.com/MadryLab/robustness
#################################################

from robustness.datasets import CIFAR, RestrictedImageNet, ImageNet
from robustness.model_utils import make_and_restore_model

def get_robust_classifier(dataset, model_path, parallel=True):
    if dataset == "cifar10":
        model, _ = make_and_restore_model(arch='resnet50', dataset=CIFAR(), \
             resume_path=model_path, parallel=parallel)
    elif dataset == "RestrictedImageNet":
        model, _ = make_and_restore_model(arch='resnet50', dataset=RestrictedImageNet(''), \
             resume_path=model_path, parallel=parallel)
    elif dataset == "ImageNet":
        model, _ = make_and_restore_model(arch='resnet50', dataset=ImageNet(''), \
             resume_path=model_path, parallel=parallel)
    else:
        raise NotImplementedError("Model for {} is not implemented!".format(dataset))

    model.eval()
    return model

if __name__ == "__main__":
    netC = get_robust_classifier("cifar10", "pretrained/cifar_l2_0_5.pt")
    import torch, torchvision
    import numpy as np
    import  torchvision.transforms as transforms
    from torch.nn import functional as F

    with torch.no_grad():
        test_dir = "../output_imgs/cifar10_new9_cLoss10.0"
        transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor()#,
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
        dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=16, shuffle=False)
        for item, data in enumerate(data_loader):
            print(data[0].shape)
            output, _ = netC(data[0])
            output = F.softmax(output).data.cpu().numpy()
            print(output.shape)
            argmax = np.argmax(output, axis=-1)
            print(argmax.squeeze())
            maxp = np.amax(output, axis=-1)
            print(maxp.squeeze())
