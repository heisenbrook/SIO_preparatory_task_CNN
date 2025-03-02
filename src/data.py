import numpy as np
import yaml
from torch.utils.data import random_split
from PIL import Image
from torchvision import datasets, transforms
from pathlib import Path

def lbl_to_class(label):
    """
    Returns the string associated from the numeric label
    """
    names = ['T-shirt or top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    lbl_dict = {}

    for idx, name in enumerate(names):
        lbl_dict[idx] = name
    
    return lbl_dict[label]


def train_test_val(output_dir):
    """
    Download and trasform the datasets to re-organize as .yaml file
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
    
    training_set, val_set = random_split(training_set, [0.8, 0.2])

    return training_set, test_set, val_set


def create_dir(data, set, dir):
    """
    Creates and returns the structured directories for the correct implementation
    of .yaml file
    """
    for idx, (image, label) in enumerate(set):
        image = image.numpy()
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        subdir = dir / lbl_to_class(label)
        subdir.mkdir(exist_ok=True)
        path = str(subdir / f'{idx}.png')
        image.save(path, format='png')
        data.append({'image_path': path, 'label': lbl_to_class(label)})       


def get_data_yaml():
    """
    Creates and returns the .yaml file used for training YOLOv11 model.
    """

    output_dir = Path('data/')
    yaml_filename = 'fashionmnist_classification.yaml'

    training_set, test_set, val_set = train_test_val(output_dir)

    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    val_dir = output_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    

    train_data = []
    test_data = []
    val_data = []

    create_dir(train_data, training_set, train_dir)
    create_dir(test_data, test_set, test_dir)
    create_dir(val_data, val_set, val_dir)


    with open(output_dir / yaml_filename, 'w') as file:
        yaml.dump({
            'train': train_data,
            'test': test_data,
            'val': val_data,
            'nc': 10,
            'names': ['T-shirt or top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
         }, file)
    

    with open(output_dir / yaml_filename, 'r') as file:
        data = yaml.safe_load(file)

    class_names = data['names']
    
    return str(output_dir), class_names