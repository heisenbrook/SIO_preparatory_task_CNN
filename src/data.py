import os
import numpy as np
import yaml
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms


def get_data_yaml():
    """
    Creates and returns the .yaml file used for training YOLOv11 model.
    """
    transform = transforms.ToTensor()
    
    training_set = datasets.FashionMNIST(root = 'data', 
                                         train= True, 
                                         download= True, 
                                         transform=transform)

    test_set = datasets.FashionMNIST(root = 'data', 
                                     train= False, 
                                     download= True, 
                                     transform=transform)
    
    os.makedirs('fashionmnist/train', exist_ok=True)
    os.makedirs('fashionmnist/test', exist_ok=True)

    train_data = []
    test_data = []


    for idx, (image, label) in enumerate(training_set):
        image = image.numpy()
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        path = f'fashionmnist/train/{idx}'
        image.save(path, format='png')
        train_data.append({'image_path': path, 'label': int(label)})

    for idx, (image, label) in enumerate(test_set):
        image = image.numpy()
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        path = f'fashionmnist/test/{idx}'
        image.save(path, format='png')
        test_data.append({'image_path': path, 'label': int(label)})

    with open('data/fashionmnist_classification.yaml', 'w') as f:
        yaml.dump({
            'train': train_data,
            'test': test_data,
            'nc': 10,
            'names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
         }, f)
    
    yaml_path = 'data/fashionmnist_classification.yaml'
    return yaml_path