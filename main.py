from src.data import get_data_yaml
from src.utils import model_training_transfer, model_training


class_names = []


data_path, class_names = get_data_yaml()
attention = 15

#training and evaluating model without transfer learning
model_training(data_path, class_names, attention)

#training and evaluating model with transfer learning
model_training_transfer(data_path, class_names, attention)



