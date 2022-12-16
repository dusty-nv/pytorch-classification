#!/usr/bin/env python3
import os
import torch
import numpy as np
import xml.dom.minidom

from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    """
    Multi-label classification dataset for Pascal VOC (http://host.robots.ox.ac.uk/pascal/VOC/)
    This extracts objects (from the object detection benchmark) and uses them as image tags.
    """
    def __init__(self, root, set, transform=None, target_transform=None, use_difficult=False):
        """
        Load the dataset (tested on VOC2012)
        
        Parameters
            root (string) -- path to the VOC2007 or VOC2012 dataset, containing the following sub-directories:
                             Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject
                             
            set (string) -- the data subset, which corresponds to one of the files under ImageSets/Main/
                            like 'train', 'trainval', 'test', 'val', ect.
        """
        self.root = root
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # load the image ID's
        with open(os.path.join(self.root, f"ImageSets/Main/{set}.txt")) as file:
            self.id_list = file.read().splitlines() 
            
        # load the labels
        self.labels = []
        
        for id in self.id_list:
            xml_tree = xml.dom.minidom.parse(os.path.join(self.root, 'Annotations', f'{id}.xml'))
            xml_root = xml_tree.documentElement
            
            objects = xml_root.getElementsByTagName('object')
            labels = np.zeros(len(self.classes), dtype=np.float32)
            
            for obj in objects:
                if (not use_difficult) and (obj.getElementsByTagName('difficult')[0].firstChild.data == '1'):
                    continue
                    
                tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                labels[self.classes.index(tag)] = 1.0
                
            self.labels.append(torch.from_numpy(labels))
            
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, 'JPEGImages', f'{self.id_list[index]}.jpg')).convert('RGB')
        labels = self.labels[index]
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            labels = self.target_transform(labels)
            
        return image, labels

    def __len__(self):
        return len(self.id_list)
 
    def get_class_distribution(self):
        with torch.no_grad():
            distribution = torch.zeros(len(self.classes))
            for labels in self.labels:
                distribution += labels

        return distribution.to(dtype=torch.int64).tolist()
           
           
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, default='.')
    parser.add_argument('--set', type=str, default='trainval')
    parser.add_argument('--load-data', action='store_true')
    parser.add_argument('--distribution', action='store_true')
    
    args = parser.parse_args() 
    print(args)
    
    # load the dataset
    dataset = VOCDataset(args.data, args.set)
    print(f"=> loaded VOC dataset  set={dataset.set}  classes={len(dataset.classes)}  images={len(dataset)}")
    
    # verify that all images load
    if args.load_data:
        for idx, (img, target) in enumerate(dataset):
            print(f"loaded image {idx}  dims={img.size}  classes={[dataset.classes[n] for n in target.nonzero(as_tuple=True)[0]]}")
            #print(f"labels:  {target}")
    
    # get the class distributions
    if args.distribution:
        print("=> computing class distributions:")
        
        distribution = dataset.get_class_distribution()
        total_labels = 0
        
        for n, count in enumerate(distribution):
            print(f"  class {n} {dataset.classes[n]} - {count}")
            total_labels += count
            
        print(f"=> loaded VOC dataset  set={dataset.set}  classes={len(dataset.classes)}  images={len(dataset)}  labels={total_labels}")
        