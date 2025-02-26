import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

def get_data():
    """
    Creates and returns the training loader, test loader and the number of 
    classes of the new choosen dataset.
    The transform applied to the dataset are minimal and they just involve resizing
    (to fit YOLOv11 input dimension) and normalization
    """
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)])
    
    training_set = datasets.CelebA(root = 'data', 
                                   split= 'train', 
                                   target_type= 'identity',
                                   download= True, 
                                   transform=transform)
    
    training_loader = DataLoader(training_set, 
                                 batch_size=64, 
                                 shuffle=True, 
                                 num_workers=4)

    test_set = datasets.CelebA(root = 'data', 
                               split= 'test', 
                               target_type= 'identity',
                               download= True, 
                               transform=transform)
    
    test_loader = DataLoader(test_set, 
                             batch_size=64, 
                             shuffle=False, 
                             num_workers=4)

    return (training_loader, test_loader, len(training_set.classes))



def model_transfer_l(classes, device):
    """
    Loads the model from /pretrained_models/, freezes the parameters and add a new head
    for the classification task of the new dataset and returns the new model for training
    """
    model = YOLO('yolo11n-cls.yaml').load('yolo11n-cls.pt')

    model.export(format='torchscript', device=device)

    for param in model.parameters():
        param.requires_grad = False

    in_feat = model.linear.in_features

    model.linear = nn.Linear(in_feat, classes)
    return model
