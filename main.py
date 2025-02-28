from src.data import get_data_yaml
from src.utils import model_training
from ultralytics import YOLO
from pathlib import Path

class_names = []

data_dir = Path('/home/matteo-vannini/Scrivania/SIO_preparatory_task_CNN/data/')
yaml_filename = 'fashionmnist_classification.yaml'
data_path, class_names = get_data_yaml(data_dir, yaml_filename)

model_training(data_dir, class_names)

