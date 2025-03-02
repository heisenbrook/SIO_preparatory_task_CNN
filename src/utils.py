import torch
from ultralytics import YOLO


def model_training_transfer(data_path, class_names, att_epoch):
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
        epochs=100,
        patience=att_epoch,
        batch=64,
        device=device,
        workers=4,
        project='training_results',
        name=f'tl_run_{att_epoch}_epoch_attention',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        classes=class_names,
        freeze=10,
        val=True,
        plots=True
    )

def model_training(data_path, class_names, att_epoch):
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
        epochs=100,
        patience=att_epoch,
        batch=64,
        device=device,
        workers=4,
        project='training_results',
        name=f'normal_run_{att_epoch}_epoch_attention',
        exist_ok=True,
        pretrained=False,
        optimizer='Adam',
        classes=class_names,
        val=True,
        plots=True
    )



    
