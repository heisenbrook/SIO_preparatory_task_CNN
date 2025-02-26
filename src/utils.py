import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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



def load_model_transfer_l(classes, device):
    """
    Loads the model from ultralytics and saves it as a torchscript file, freezes the parameters and add a new head
    for the classification task of the new dataset and returns the new model for training
    """
    model = YOLO('yolo11n-cls.yaml').load('yolo11n-cls.pt')

    model.export(format='torchscript', device=device)

    for param in model.parameters():
        param.requires_grad = False

    in_feat = model.linear.in_features

    model.linear = nn.Linear(in_feat, classes)
    return model



def training(model, n_epochs, training_loader, optimizer, loss_fn, device):
    """
    Training function for the imported and modified model for transfer learning
    :return: total_loss, total_accuracy, total_prec
    """
    
    total_train_loss, total_accuracy, total_prec = [], [], []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for imag, lbl in tqdm(enumerate(training_loader),
                              desc="Batch",
                              total=len(training_loader),
                              leave=False,
                              ncols=80,
                              ):
            imag, lbl = imag.to(device), lbl.to(device)

            optimizer.zero_grad()
            out = model(imag)
            loss = loss_fn(out, lbl)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(out, dim=1)
            train_loss += loss.item()
            epoch_loss = train_loss/len(training_loader)
            accuracy = accuracy_score(lbl, pred)
            prec = precision_score(lbl, pred, average='macro')
        print(f'Epoch {epoch + 1} | training accuracy:{accuracy:.2f}% | precision:{prec:.5f}% | training loss:{epoch_loss:.5f}% ')
        total_train_loss.append(epoch_loss)
        total_accuracy.append(accuracy)
        total_prec.append(prec)
    return total_train_loss, total_accuracy, total_prec

def testing(model, test_loader, loss_fn):
    """
    Testing function for the imported and modified model for transfer learning
    """
    model.eval()
    test_loss = 0.0

    for imag, lbl in tqdm(enumerate(test_loader),
                          desc="Batch",
                          total=len(test_loader),
                          leave=False,
                          ncols=80,
                          ):

        out = model(imag)
        loss = loss_fn(out, lbl)
            
        pred = torch.argmax(out, dim=1)
        test_loss += loss.item()
        accuracy = accuracy_score(lbl, pred)
        prec = precision_score(lbl, pred, average='macro')
        eval_loss = test_loss/len(test_loader)
    print(f'test accuracy:{accuracy:.2f}% | precision:{prec:.5f}% | training loss:{eval_loss:.5f}% ')


    
