import argparse
import torch
import torchvision.models as models

from PIL import Image
from torchvision.transforms import ToTensor


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# exporter settings
parser = argparse.ArgumentParser()
parser.add_argument('--model_in', type=str, default='model_best.pth.tar')
parser.add_argument('--model_out', type=str, default='resnet-18.onnx')
parser.add_argument('--image', type=str, required=True, help='input image to use')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

opt = parser.parse_args() 
print(opt)


# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))


# load the image
img = Image.open(opt.image)
img_to_tensor = ToTensor()
input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0]).to(device)

print('input image size {:d}x{:d}'.format(img.size[0], img.size[1]))


# load the model
#print(torch.load(opt.model_in))
print('using model: ' + opt.arch)
model = models.__dict__[opt.arch](pretrained=True)
print('loading checkpoint: ' + opt.model_in)
checkpoint = torch.load(opt.model_in)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()
print(model)


# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to ONNX...')
torch.onnx.export(model, input, opt.model_out, verbose=True, input_names=input_names, output_names=output_names)
print('model exported to {:s}'.format(opt.model_out))

