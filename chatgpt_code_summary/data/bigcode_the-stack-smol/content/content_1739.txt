import torch
import torchvision
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix


def makeDataSet(IMAGE_SHAPE = 300,DATA_PATH = './data_after_splitting/'):

	image_transforms = {
	    "train": transforms.Compose([
	        transforms.Resize((IMAGE_SHAPE, IMAGE_SHAPE)),
	        transforms.ToTensor(),
	        transforms.Normalize([0.5, 0.5, 0.5],
	                             [0.5, 0.5, 0.5])
	    ]),
	    "val": transforms.Compose([
	        transforms.Resize((IMAGE_SHAPE, IMAGE_SHAPE)),
	        transforms.ToTensor(),
	        transforms.Normalize([0.5, 0.5, 0.5],
	                             [0.5, 0.5, 0.5])
	    ])
	}

	train_dataset = datasets.ImageFolder(root = DATA_PATH + "train",
	                                   transform = image_transforms["train"]
	                                  )

	val_dataset = datasets.ImageFolder(root = DATA_PATH + "val",
	                                   transform = image_transforms["val"]
	                                  )
	train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=2, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=2, shuffle=True)

	return train_dataloader,val_dataloader
