# Travel Image Classification using Transfer Learning with PyTorch

## Overview

This project aims to build an AI system that can automatically classify uploaded travel photos into one of three categories:

- **Exploring in the Forest**
- **Adventure in the Desert**
- **Relaxing on the Beach**

A travel planning company wants to analyze customers' uploaded vacation photos to understand their preferences and suggest suitable trips. However, the company has fewer than 500 categorized photos, which is insufficient to train a deep learning model from scratch. To overcome this challenge, we leverage **transfer learning** using a pre-trained VGG16 model to achieve accurate image classification with limited data.

## Project Structure

- **`main.py`**: The main script that orchestrates the transfer learning process, including data loading, model modification, training, and testing.
- **`train.py`**: Contains the `train_model` function that handles the training loop, including tracking the best model based on validation accuracy.
- **`test.py`**: Contains the `test_model` function that visualizes the model's predictions on the test dataset.
- **`helper.py`**: Includes utility functions, such as `to_device_precision`, to handle device-specific operations.

## Dataset

The dataset consists of categorized images stored in the following directory structure:

```
data/
├── imagedata-50/
    ├── train/
        ├── category1/
        ├── category2/
        ├── category3/
    ├── val/
        ├── category1/
        ├── category2/
        ├── category3/
    ├── test/
        ├── category1/
        ├── category2/
        ├── category3/
```

- **Categories**:
  - `category1`: Exploring in the Forest
  - `category2`: Adventure in the Desert
  - `category3`: Relaxing on the Beach

Each category folder contains images belonging to that class. The dataset is split into training, validation, and test sets.

## Instructions

### 1. Data Preparation

- Organize your images into the directory structure shown above.
- Ensure that each category folder contains the appropriate images.
- Since the dataset is small, you may consider augmenting the data or using more images if available.

### 2. Install Dependencies

Ensure you have Python 3.12 installed. Install the required Python libraries using:

```bash
pipenv install
```

### 3. Run the Main Script

Execute the main script to start the transfer learning process:

```bash
python main.py --type=transfer_learning
```

This script will:

- Load and preprocess the data using the specified transformations.
- Modify a pre-trained VGG16 model for our classification task.
- Train the model using the training data.
- Validate the model using the validation data.
- Save the best model weights.
- Test the model on the test dataset and display predictions.

### 4. View Test Results

The `test_model` function in `test.py` will display images from the test set along with the model's predicted category. This allows you to visually assess the model's performance.

## Code Details

### `main.py`

- **Data Transforms**: Defines image transformations for training, validation, and testing, including resizing, normalization, and data augmentation (if desired).
- **Data Loading**: Uses `torchvision.datasets.ImageFolder` to load images from the directory structure.
- **Model Modification**:
  - Loads the pre-trained VGG16 model using `models.vgg16(weights='DEFAULT')`.
  - Freezes all convolutional layers to retain learned features.
  - Replaces the classifier's final layer to match the number of classes (3 in this case).
- **Training Setup**:
  - Defines the loss function (`CrossEntropyLoss`) and optimizer (`SGD`).
  - Sets up a learning rate scheduler to adjust the learning rate during training.
- **Training Execution**: Calls `train_model` from `train.py` to train the model.
- **Model Saving**: Saves the best model weights to the `weights` directory.
- **Testing**: Calls `test_model` from `test.py` to evaluate and visualize the model's predictions.

### `train.py`

- **`train_model` Function**:
  - Handles the training loop over the specified number of epochs.
  - Tracks training and validation losses and accuracies.
  - Updates the model parameters based on the optimizer and scheduler.
  - Keeps a copy of the model with the highest validation accuracy.
  - Returns the best model after training completes.

### `test.py`

- **`test_model` Function**:

  - Loads the test data and passes it through the trained model.
  - Uses `matplotlib` to display images along with their predicted categories.
  - Allows for a visual assessment of the model's performance on unseen data.

- **`imshow` Function**:
  - A helper function to display images with proper normalization and scaling.

### `helper.py`

- **`to_device_precision` Function**:
  - Adjusts tensor precision based on the device being used (CPU, CUDA, or MPS).
  - Ensures compatibility and optimal performance across different hardware.

## Usage

### Command-Line Interface

You can modify `main.py` to adjust parameters such as the number of epochs, batch size, number of workers, and data paths.

Example:

```python
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transfer_learning(
        device=device,
        num_epochs=20,
        batch_size=32,
        num_workers=4,
        data_path="data",
        data_dir="imagedata-50"
    )
```

### Data Augmentation

To potentially improve model performance with limited data, you can uncomment or add data augmentation techniques in the `data_transforms` dictionary in `main.py`:

```python
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(10),
transforms.ColorJitter(),
```

### Adjusting the Model

If you wish to experiment with different pre-trained models, you can replace VGG16 with another model from `torchvision.models`, such as ResNet or EfficientNet.

Example:

```python
model = models.resnet50(weights='DEFAULT')
# Modify the model to fit the number of classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
```

## Requirements

- **Python 3.x**
- **Libraries**:
  - `torch`
  - `torchvision`
  - `matplotlib`
  - `numpy`

Install the required packages using:

```bash
pipenv install
```

- **Hardware**:
  - A GPU is recommended for faster training, but the code can run on a CPU.

## Potential Improvements

- **Data Augmentation**: Increase the variety of training data through augmentation to improve model generalization.
- **Fine-tuning**: Unfreeze some of the deeper layers in the pre-trained model to fine-tune the model on the new dataset.
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and optimizers.
- **Additional Data**: Collect more images for each category to enhance model performance.

## Conclusion

By utilizing transfer learning, this project demonstrates how to build an effective image classification model with a limited dataset. The pre-trained VGG16 model provides a solid foundation by leveraging learned features from a vast dataset like ImageNet. Modifying and training the model for our specific categories allows us to achieve accurate classification of travel photos, aiding the travel planning company's AI bot in suggesting suitable trips based on customer preferences.
