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
parser.add_argument('--model_in', type=str, default='model_best.pth.tar', help="name of input PyTorch model (default: model_best.pth.tar)")
parser.add_argument('--model_out', type=str, default='', help="name of output ONNX model (default: <ARCH>.onnx)")
parser.add_argument('--input_width', type=int, default=224, help="resolution of input image expected by the model (default: 224)")
parser.add_argument('--input_height', type=int, default=224, help="resolution of input image expected by the model (default: 224)")
#parser.add_argument('--image', type=str, required=True, help='example input image to use')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

opt = parser.parse_args() 
print(opt)

if not opt.model_out:
	opt.model_out = opt.arch + '.onnx'

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))


# create example image data
input = torch.ones((1, 3, opt.input_height, opt.input_width)).cuda()
#img = Image.open(opt.image)
#img_to_tensor = ToTensor()
#input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0]).to(device)

print('input image size {:d}x{:d}'.format(opt.input_width, opt.input_height))


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

