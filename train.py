#!/usr/bin/env python3
#
# Note -- this training script is tweaked from the original at:
#           https://github.com/pytorch/examples/tree/master/imagenet
#
# For a step-by-step guide to transfer learning with PyTorch, see:
#           https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#
import argparse
import os
import random

import time
import shutil
import warnings
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from voc import VOCDataset
from nuswide import NUSWideDataset
from reshape import reshape_model


# get the available network architectures
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# parse command-line arguments
parser = argparse.ArgumentParser(description='PyTorch Image Classifier Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-type', type=str, default='folder',
                    choices=['folder', 'nuswide', 'voc'],
                    help='specify the dataset type (default: folder)')
parser.add_argument('--multi-label', action='store_true',
                    help='multi-label model (aka image tagging)')
parser.add_argument('--multi-label-threshold', type=float, default=0.5,
                    help='confidence threshold for counting a prediction as correct')
parser.add_argument('--model-dir', type=str, default='models', 
                    help='path to desired output directory for saving model '
					'checkpoints (default: models/)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--resolution', default=224, type=int, metavar='N',
                    help='input NxN image resolution of model (default: 224x224) '
                         'note than Inception models should use 299x299')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=35, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                    help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU ID to use (default: 0)')

args = parser.parse_args()


# open tensorboard logger (to model_dir/tensorboard)
tensorboard = SummaryWriter(log_dir=os.path.join(args.model_dir, "tensorboard", f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
print(f"To start tensorboard run:  tensorboard --log-dir={os.path.join(args.model_dir, 'tensorboard')}")

# variable for storing the best model accuracy so far
best_accuracy = 0


def main(args):
    """
    Load dataset, setup model, and train for N epochs
    """
    global best_accuracy
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print(f"=> using GPU {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")

    # setup data transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
        
    val_transforms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        normalize,
    ])
        
    # load the dataset
    if args.dataset_type == 'folder':
        train_dataset = datasets.ImageFolder(os.path.join(args.data, 'train'), train_transforms)
        val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'), val_transforms)
    elif args.dataset_type == 'nuswide':
        train_dataset = NUSWideDataset(args.data, 'trainval', train_transforms)
        val_dataset = NUSWideDataset(args.data, 'test', val_transforms)
    elif args.dataset_type == 'voc':
        train_dataset = VOCDataset(args.data, 'trainval', train_transforms)
        val_dataset = VOCDataset(args.data, 'val', val_transforms)
    
    if (args.dataset_type == 'nuswide' or args.dataset_type == 'voc') and (not args.multi_label):
        raise ValueError("nuswide or voc datasets should be run with --multi-label")
        
    print(f"=> dataset classes:  {len(train_dataset.classes)}  {train_dataset.classes}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create or load the model if using pre-trained (the default)
    if args.pretrained:
        print(f"=> using pre-trained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()

    # reshape the model for the number of classes in the dataset
    model = reshape_model(model, args.arch, len(train_dataset.classes))

    # define loss function (criterion) and optimizer
    if args.multi_label:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        
    # transfer the model to the GPU that it should be run on
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint['best_accuracy']
            if args.gpu is not None:
                best_accuracy = best_accuracy.to(args.gpu)   # best_accuracy may be from a checkpoint from a different GPU
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    # if in evaluation mode, only run validation
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    # train for the specified number of epochs
    for epoch in range(args.start_epoch, args.epochs):
        # decay the learning rate
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion, epoch)

        # remember best acc@1 and save checkpoint
        is_best = val_acc > best_accuracy
        best_accuracy = max(val_acc, best_accuracy)

        print(f"=> Epoch {epoch}")
        print(f"  * Train Loss     {train_loss:.4e}")
        print(f"  * Train Accuracy {train_acc:.4f}")
        print(f"  * Val Loss       {val_loss:.4e}")
        print(f"  * Val Accuracy   {val_acc:.4f}{'*' if is_best else ''}")
        
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'resolution': args.resolution,
            'classes': train_dataset.classes,
            'num_classes': len(train_dataset.classes),
            'multi_label': args.multi_label,
            'state_dict': model.state_dict(),
            'accuracy': {'train': train_acc, 'val': val_acc},
            'loss' : {'train': train_loss, 'val': val_loss},
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Train one epoch over the dataset
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Accuracy', ':7.3f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc],
        prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    model.train()

    # get the start time
    epoch_start = time.time()
    end = epoch_start

    # train over each image batch from the dataset
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # record loss and measure accuracy
        losses.update(loss.item(), images.size(0))
        acc.update(accuracy(output, target), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader)-1:
            progress.display(i)
    
    print(f"Epoch: [{epoch}] completed, elapsed time {time.time() - epoch_start:6.3f} seconds")

    tensorboard.add_scalar('Loss/train', losses.avg, epoch)
    tensorboard.add_scalar('Accuracy/train', acc.avg, epoch)

    return losses.avg, acc.avg
    

def validate(val_loader, model, criterion, epoch):
    """
    Measure model performance across the val dataset
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Accuracy', ':7.3f')
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc],
        prefix='Val:   ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # record loss and measure accuracy
            losses.update(loss.item(), images.size(0))
            acc.update(accuracy(output, target), images.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader)-1:
                progress.display(i)

    tensorboard.add_scalar('Loss/val', losses.avg, epoch)
    tensorboard.add_scalar('Accuracy/val', acc.avg, epoch)
    
    return losses.avg, acc.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar', labels_filename='labels.txt'):
    """
    Save a model checkpoint file, along with the best-performing model if applicable
    """
    if args.model_dir:
        model_dir = os.path.expanduser(args.model_dir)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        filename = os.path.join(model_dir, filename)
        best_filename = os.path.join(model_dir, best_filename)
        labels_filename = os.path.join(model_dir, labels_filename)
        
    # save the checkpoint
    torch.save(state, filename)
            
    # earmark the best checkpoint
    if is_best:
        shutil.copyfile(filename, best_filename)
        print(f"saved best model to:  {best_filename}")
    else:
        print(f"saved checkpoint to:  {filename}")
        
    # save labels.txt on the first epoch
    if state['epoch'] == 0:
        with open(labels_filename, 'w') as file:
            for label in state['classes']:
                file.write(f"{label}\n")
        print(f"saved class labels to:  {labels_filename}")
            

def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """
    Computes the accuracy of predictions vs groundtruth
    """
    with torch.no_grad():
        if args.multi_label:
            output = F.sigmoid(output)
            preds = ((output >= args.multi_label_threshold) == target.bool())   # https://medium.com/@yrodriguezmd/tackling-the-accuracy-multi-metric-9e2356f62513
            
            # https://stackoverflow.com/a/61585551
            #output[output >= args.multi_label_threshold] = 1
            #output[output < args.multi_label_threshold] = 0
            #preds = (output == target)
        else:
            output = F.softmax(output, dim=-1)
            _, preds = torch.max(output, dim=-1)
            preds = (preds == target)
            
        return preds.float().mean().cpu().item() * 100.0
        
        
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Progress metering
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main(args)
