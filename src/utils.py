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



def model_training(data_path, class_names):
    """
    Loads the model from ultralytics, freezes the parameters and add a new head
    for the classification task of the new dataset and returns training results
    """

    if torch.cuda.is_available():
        print('GPU available!')
    else:
        print('GPU not available!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = YOLO('yolo11n-cls.pt')
    model.to(device)

    model.train(
        data=data_path,
        epochs=10,
        batch=32,
        imgsz=28,
        workers=4,
        device=device,
        project='training_results',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        classes=class_names,
        freeze=9
    )




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


    
