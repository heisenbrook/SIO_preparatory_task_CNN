import os
import numpy as np
import yaml
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
from pathlib import Path

def get_data_yaml(output_dir, yaml_filename):
    """
    Creates and returns the .yaml file used for training YOLOv11 model.
    """
    transform = transforms.ToTensor()
    
    training_set = datasets.FashionMNIST(root = output_dir, 
                                         train= True, 
                                         download= True, 
                                         transform=transform)

    test_set = datasets.FashionMNIST(root = output_dir, 
                                     train= False, 
                                     download= True, 
                                     transform=transform)
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    

    train_data = []
    test_data = []


    for idx, (image, label) in enumerate(training_set):
        image = image.numpy()
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        subdir = train_dir / str(label)
        subdir.mkdir(exist_ok=True)
        path = str(subdir / f'{idx}.png')
        image.save(path, format='png')
        train_data.append({'image_path': path, 'label': int(label)})

    for idx, (image, label) in enumerate(test_set):
        image = image.numpy()
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        subdir = test_dir / str(label)
        subdir.mkdir(exist_ok=True)
        path = str(subdir / f'{idx}.png')
        image.save(path, format='png')
        test_data.append({'image_path': path, 'label': int(label)})

    with open(output_dir / yaml_filename, 'w') as file:
        yaml.dump({
            'train': train_data,
            'test': test_data,
            'nc': 10,
            'names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
         }, file)
    
    

    with open(output_dir / yaml_filename, 'r') as file:
        data = yaml.safe_load(file)

    # import pdb
    # pdb.set_trace()
    # x=yaml.safe_load(open(str(f), 'r'))
    class_names = data['names']
    
    return str(output_dir), class_names