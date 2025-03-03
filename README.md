# SIO_preparatory_task_CNN
## Preparatory task for technical interview at SIO s.p.a.

### General info about the task

The primary objective of this task is to evaluate the classification capabilities of the most advanced model from [Ultralytics](https://www.ultralytics.com/it "Ultralytics link") *YOLOv11*, using a dataset that the model has not been trained on.

The evaluation will be conducted in two ways:

1. **Transfer Learning**: Fine-tuning the model by importing weights and biases from a pre-trained model.
2. **Training from Scratch**: Training the model without any pre-trained weights.

The performance of both approaches will be compared using selected validation metrics to determine which method yields better results.
The `main.py` file starts the training and registers all the used validation metrics for both models.

A detailed description of each step, along with relevant results, will be provided.


#### 1. Identify an unrecognized class of objects from YOLOv11 model:
YOLO11's pre-trained classification models are all trained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml "Ultralytics yaml file for ImageNet") dataset. For this task, the chosen dataset is [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist "Fashion-MNIST"), which is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 0 to 9, representing a specific clothing category. 

The primary challenges that this dataset may pose to YOLOv11, which is pre-trained on ImageNet, include:

1. Differences in image size (28x28 grayscale vs. 224x224 RGB).
2. Domain shift (fashion items vs. general objects).
3. Lack of color information in Fashion-MNIST.

#### 2. Get the Data:
The dataset is downloaded using the `torchvision.datasets` module in python, which offers a wide range of built-in datasets as well as utility classes for creating custom datasets.

The datasets is automatically split into *train* and *test* sets by the module. For validation instead, a portion of the *train* dataset (20%) was randomly allocated.

All the code related is available in `src\data.py`

#### 3. Pre-process Data:
No transformations is applied to the dataset other than converting it to a tensor using the `torchvision.transforms` module.

However, to feed the dataset into the model, it needs to be converted into a `.yaml` file—a human-readable data serialization language that is often used for writing configuration files which points to the structured folders containing the dataset. While the `ultralytics` Python module provides tools to automatically download and convert the dataset, the challenge here was to perform the conversion manually.

To structure the dataset for Ultralytics YOLO classification tasks, a specific split-directory format should be followed. The dataset should be organized into separate directories for *train*, *test*, and optionally *val*. Each of these directories should contain subdirectories named after each class, with the corresponding images inside. This structure ensures a smooth training and evaluation process.

All the code for the conversion is located in `src\data.py`


#### 4. Code necessary for transfer learning on YOLOv11:
No implementation with `torch.hub` module is provided yet for this model, but Ultralytics provides all the necessary tools to use it in a python environment.
The code was straightforward: using `ultralytics` module, the model can be downloaded and trained with simple commands, deciding whether to to apply transfer learning (freezing layers and modify last layer to adapt to the new classification problem) or training from scratch.

A code snippet, also found [here](https://docs.ultralytics.com/modes/train/#key-features-of-train-mode) is provided to illustrate the simplicity of the implementation:
```python
from ultralytics import YOLO

    
model = YOLO('yolo11n-cls.pt')

model.train(
        data='path/to/data',
        epochs=100,
        ...
    )
```
Using the `model = YOLO(*model*)` command downloads the chosen model, while the `model.train()` function initiates the training process. The function accepts various hyperparameters, which can be configured using the [train settings](https://docs.ultralytics.com/modes/train/#train-settings "Ultralytics train settings") provided by Ultralytics.  These settings, which influence the model's performance, speed, and accuracy, require careful tuning to optimize results.

#### 5. Choice of frozen layers and stop criteria:
For simplicity, transfer learning was implemented by importing a pre-trained model and freezing all layers except the last one. The last layer was modified to adapt to the new dataset—ImageNet has 1,000 image classes, while Fashion-MNIST has only 10.

In contrast, the starting model was trained from scratch. Instead of using a pre-trained model, only the model's architecture was utilized, with the last layer adjusted to fit the new classification task.

The chosen stopping criterion was the `patience` parameter, which determines the number of epochs to wait without improvement in validation metrics before early stopping is triggered. This helps prevent overfitting. While further fine-tuning could certainly be done, the goal here was to keep things simple and efficient. As a result, only two experiments were conducted, using 10 and 15 epochs of patience, respectively, while training both models for 100 epochs. All other hyperparameters were left unchanged.

#### 6. Tests used to validate training with transfer learning vs normal training:
Validation is a critical step in the machine learning pipeline, as it allows for assessing the quality of trained models. The `.val()` method in Ultralytics provides all the necessary tools and metrics to evaluate model performance during training.

The most useful metrics for the evaluation of classification tasks are: 

1. **Accuracy**: 
```math
Accuracy = \frac{(TP + TN)}{(TP + TN + FP + FN)}
```
Measures the percentage of correct predictions (True Positive *TP* + True Negative *TN*) out of the total (True Positive *TP* + True Negative *TN* + False Positive *FP* + False Negative *FN*).

2. **Precision**:
```math
Precision = \frac{TP}{(TP + FP)}
```
Measures the percentage of correct positive predictions out of the total positive predictions.

3. **Recall**:
```math
Recall = \frac{TP}{(TP + FN)}
```
Measures the percentage of actual positive instances correctly identified out of the total actual positive instances.

4. **F1-Score**:
```math
F1-Score = 2 * \frac{(Precision * Recall)}{(Precision + Recall)}
```
It is the harmonic mean of precision and recall, useful when classes are imbalanced.

5. **Specifity**:
```math
Specifity = \frac{TN}{(TN + FP)}
```
Measures the percentage of actual negative instances correctly identified out of the total actual negative instances.

6. **Confusion Matrix**:
All of the above metrics are derived from the confusion matrix, which evaluates the performance of a classification model by comparing the model's predictions with the actual values (ground truth). It is particularly useful for analyzing model errors and understanding how data is classified.

In this task, the focus will primarily be on **accuracy**—specifically, top-1 and top-5 accuracy—and the confusion matrix. Additionally, the trends in training and validation losses over time will also be considered.



### Conclusions
The plots and matrices below show the trend for the loss for both training and validation as well with the accuracy—top 1 and top 5—of both models for all the tested patience parameter.

##### 10 epochs patience:
###### plots
<img src= 'https://github.com/heisenbrook/SIO_preparatory_task_CNN/blob/main/training_results/normal_run_10_epoch_patience/results.png' width="300" height="300" />   <img src= 'https://github.com/heisenbrook/SIO_preparatory_task_CNN/blob/main/training_results/tl_run_10_epoch_patience/results.png' width="300" 
height="300" />  

###### confusion matrix
<img src= 'https://github.com/heisenbrook/SIO_preparatory_task_CNN/blob/main/training_results/normal_run_10_epoch_patience/confusion_matrix_normalized.png' width="300" height="300" />   <img src= 'https://github.com/heisenbrook/SIO_preparatory_task_CNN/blob/main/training_results/tl_run_10_epoch_patience/confusion_matrix_normalized.png' width="300" 
height="300" />  

##### 15 epochs patience:
###### plots
<img src= 'https://github.com/heisenbrook/SIO_preparatory_task_CNN/blob/main/training_results/normal_run_15_epoch_patience/results.png' width="300" height="300" />    <img src= 'https://github.com/heisenbrook/SIO_preparatory_task_CNN/blob/main/training_results/tl_run_15_epoch_patience/results.png' width="300" height="300" />

###### confusion matrix
<img src= 'https://github.com/heisenbrook/SIO_preparatory_task_CNN/blob/main/training_results/normal_run_15_epoch_patience/confusion_matrix_normalized.png' width="300" height="300" />   <img src= 'https://github.com/heisenbrook/SIO_preparatory_task_CNN/blob/main/training_results/tl_run_15_epoch_patience/confusion_matrix_normalized.png' width="300" 
height="300" />  

Comparing the performance reveals that the model using transfer learning achieves higher accuracy (~87.5% vs. ~86.5%) and lower loss (~0.35 vs. ~0.36) compared to models trained from scratch. Transfer learning also enables faster learning, as demonstrated by the quicker decrease in training loss. Additionally, models utilizing transfer learning appear to generalize better, with more stable validation loss and higher accuracy.

When evaluating the impact of the patience parameter, both models (trained from scratch and with transfer learning) achieve slightly better performance with a patience of 15 epochs compared to 10 epochs. Higher patience allows the model to continue training even after reaching a plateau, further refining its performance.

In conclusion, transfer learning is the preferred approach for this task, as it achieves higher accuracy, lower loss, and faster convergence. It leverages pre-existing knowledge to learn more efficiently and generalize better to new data.

Training from scratch can be beneficial when the dataset is significantly different from those used for pre-training. However, in this case, it does not appear necessary, as it requires more time and computational resources to achieve performance comparable to transfer learning.

Finally, while a higher patience value (15) improves model performance, it also increases training time. If time is a critical factor, a patience of 10 epochs provides a good compromise between performance and efficiency.
