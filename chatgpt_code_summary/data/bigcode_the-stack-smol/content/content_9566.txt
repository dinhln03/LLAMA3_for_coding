import pytorchresearch as ptr


import torch
import torchvision

if __name__ == "__main__":
    # transform for data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # dataloaders
    trainset = torchvision.datasets.CIFAR100(root='./data/datasets', train=True,
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data/datasets', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    # model specific stuff
    model = torchvision.models.MobileNetV2(num_classes=100)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=0.001,
        momentum=0.9
    )
    criterion = torch.nn.CrossEntropyLoss()

    # MAGIC GOES HERE

    research = ptr.ModelResearch(
        research_path='.temp',
        research_scheme=[
            ptr.ModelConfigurationItem(),
            ptr.CurrentIterationItem(print_end=' ', iteration_modulo=10),
            ptr.LossPrintItem(iteration_modulo=10),
            ptr.LossVisualizationItem(iteration_modulo=10)
        ],
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        watch_test=False
    )

    research.start_research_session(
        trainloader, testloader, epochs=1, iteration_modulo=20)
