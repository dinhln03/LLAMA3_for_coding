import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import Image

import string
import torch
import torchvision
import torchvision.transforms as transforms
import coremltools as ct


from util import util
import numpy as np

opt = TrainOptions().gather_options()
opt.isTrain = True
opt.name = "siggraph_caffemodel"
opt.mask_cent = 0
# opt.name = "siggraph_retrained"
opt.gpu_ids = []
opt.load_model = True
opt.num_threads = 1   # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.display_id = -1  # no visdom display
opt.phase = 'val'
opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
opt.serial_batches = True
opt.aspect_ratio = 1.

# process opt.suffix
if opt.suffix:
    suffix = ('_' + opt.suffix.format(**vars(opt))
              ) if opt.suffix != '' else ''
    opt.name = opt.name + suffix

opt.A = 2 * opt.ab_max / opt.ab_quant + 1
opt.B = opt.A


class Colorization(torch.nn.Module):
    def __init__(self):
        super(Colorization, self).__init__()
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        self.model = model

    def forward(self, image, hint):
        data = {
            "A": image[:, 0:1, :, :],
            "B": image[:, 1:3, :, :],
            "hint_B": hint[:, 0:2, :, :],
            "mask_B": hint[:, 2:3, :, :]
        }
        # with torch.no_grad():
        self.model.set_input(data)
        self.model.forward()
        fake_reg = torch.cat((self.model.real_A, self.model.fake_B_reg), dim=1)
        return fake_reg


image_path = "./large.JPG"
image = Image.open(image_path)
image = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
])(image)
image = image.view(1, *image.shape)
image = util.crop_mult(image, mult=8, HWmax=[4032, 4032])
transforms.ToPILImage()(image[0]).show(command='fim')

data = util.get_colorization_data(
    [image], opt, ab_thresh=0., p=0.125)
img = torch.cat((data["A"], data["B"]), dim=1)
hint = torch.cat((data["hint_B"], data["mask_B"]), dim=1)

# print(data["mask_B"], data["hint_B"])
# data["hint_B"] = torch.zeros_like(data["hint_B"])
# data["mask_B"] = torch.zeros_like(data["mask_B"])
# model = Colorization()
with torch.no_grad():
    model = Colorization()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.model.set_requires_grad(model.model.netG)

# model(data)

# transforms.ToPILImage()(image[0]).show(command='fim')
# to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr',
#                 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]

# visuals = util.get_subset_dict(
#     model.model.get_current_visuals(), to_visualize)

# for key, value in visuals.items():
#     print(key)
#     transforms.ToPILImage()(value[0]).show(command='fim')
output = model(img, hint)
output = util.lab2rgb(output, opt=opt)
transforms.ToPILImage()(output[0]).show(command='fim')

traced_model = torch.jit.trace(
    model, (img, hint), check_trace=False)

mlmodel = ct.convert(model=traced_model, inputs=[
    ct.TensorType(name="image", shape=ct.Shape(
        shape=(1, 3, ct.RangeDim(1, 4096), ct.RangeDim(1, 4096)))),
    ct.TensorType(name="hint", shape=ct.Shape(
        shape=(1, 3, ct.RangeDim(1, 4096), ct.RangeDim(1, 4096)))),
])
mlmodel.save("~/color.mlmodel")
