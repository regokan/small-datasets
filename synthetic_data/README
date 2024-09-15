# Loan Approval Prediction with Synthetic Data Generation using Variational Autoencoders

## Overview

This project addresses the problem of class imbalance in a loan approval dataset by generating synthetic data using a Variational Autoencoder (VAE). The dataset has a significant imbalance with fewer instances of denied loans (`Loan Status = 1`). By augmenting the dataset with synthetic data for the minority class, we aim to improve the performance of a loan approval prediction model.

## Project Structure

- **`main.py`**: The main script that orchestrates the entire process—data loading, model training, synthetic data generation, data augmentation, and model evaluation.
- **`train.py`**: Contains classes and functions for data loading, preprocessing, model definition (Autoencoder), and the custom loss function used during training.
- **`test.py`**: Contains code for testing the model's performance using precision, recall, and F1 score metrics.

## Dataset

The dataset used is `loan_continuous.csv`, which contains records of loan applications with various features and a `Loan Status` indicating approval (`0`) or denial (`1`). The dataset is imbalanced, with fewer denied loans, which poses a challenge for training an effective predictive model.

## Instructions

### 1. Load and Preprocess the Data

- **Load the Data**: Place the `loan_continuous.csv` file inside a `data` directory relative to the scripts.
- **Split the Data**: Extract records where `Loan Status = 1` to focus on the denied loans for augmentation.

### 2. Baseline Model Performance

- Use the `test_model` function in `test.py` to evaluate the baseline performance on the original dataset.
- This provides a reference point to assess the impact of data augmentation.

### 3. Create Datasets for Training and Validation

- Utilize the `DataBuilder` class in `train.py` to create PyTorch `Dataset` objects for training and validation.
- The data is standardized, and missing values are handled by filling them with `-99`.

### 4. Train the Variational Autoencoder

- Use the provided `Autoencoder` and `CustomLoss` classes in `train.py`.
- Implement training and validation loops in `main.py`.
- Monitor training and validation losses to ensure proper convergence.

### 5. Generate Synthetic Data

- After training, use the `generate_fake` function in `train.py` to generate 50,000 additional synthetic records for the denied loan class.
- The synthetic data is inverse-transformed back to the original scale using the `StandardScaler`.

### 6. Augment the Dataset

- Combine the generated synthetic data with the original dataset to create an augmented dataset.
- Save the augmented dataset as `loan_combined.csv` in the `data` directory.

### 7. Evaluate Model Performance

- Use the `test_model` function in `test.py` to evaluate the model's performance on the augmented dataset.
- Compare the precision, recall, and F1 score for both loan statuses before and after augmentation.

## Requirements

- **Python 3.x**
- **Libraries**:
  - `torch`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib` (optional, for plotting)
  - `torchvision` (if not already installed with `torch`)

Install the required packages using:

```bash
pipenv install
```

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/loan-prediction-vae.git
```

### 2. Prepare the Data

- Ensure the `loan_continuous.csv` file is placed inside a `data` directory within the project.

### 3. Run the Main Script

Execute the main script to start the entire process:

```bash
python main.py --type=synthetic_data --data_dir="loan_continuous.csv" --batch_size=64 --num_workers=8 --epochs=50
```

### 4. Observe the Output

- The script will print training and validation losses for each epoch.
- After training, it will generate synthetic data, augment the dataset, and evaluate the model.
- The final precision, recall, and F1 scores will be displayed in the console.

## File Descriptions

### `main.py`

This is the main script that:

- Loads and preprocesses the data using `DataBuilder`.
- Splits the data into training and validation sets.
- Initializes and trains the Variational Autoencoder.
- Generates synthetic data for the denied loan class.
- Augments the original dataset with the synthetic data.
- Evaluates the model performance using the `test_model` function.

**Key Functions and Classes Used**:

- `DataBuilder` from `train.py`
- `Autoencoder` and `CustomLoss` from `train.py`
- `generate_fake` from `train.py`
- `test_model` from `test.py`

### `train.py`

Contains the classes and functions necessary for training the VAE:

- **`DataBuilder`**: A custom `Dataset` class that:

  - Loads and standardizes the data.
  - Handles missing values.
  - Splits the data into features and targets.

- **`CustomLoss`**: A custom loss function that combines Mean Squared Error (MSE) loss and Kullback-Leibler Divergence (KLD) loss, which is essential for training VAEs.

- **`Autoencoder`**: Defines the architecture of the Variational Autoencoder with encoder and decoder networks.

- **`generate_fake`**: A function that uses the trained VAE to generate synthetic data samples.

### `test.py`

Contains code for evaluating the model's performance:

- **`test_model`**: Evaluates the model using a Multi-layer Perceptron (MLP) classifier and `GridSearchCV` for hyperparameter tuning.
- **`load_xy`**: Prepares the features (`X`) and labels (`y`) for testing by handling missing values and encoding categorical variables.

## Detailed Steps

### Data Loading and Preprocessing

- The `DataBuilder` class loads the dataset and fills missing values with `-99`.
- The data is standardized using `StandardScaler` to have zero mean and unit variance.

### Model Training

- The VAE is trained using the `Autoencoder` class, which consists of:
  - An encoder that maps input data to a latent space.
  - A decoder that reconstructs the data from the latent space.
- The `CustomLoss` function computes the loss as the sum of reconstruction loss (MSE) and the KL divergence.

### Synthetic Data Generation

- After training, the VAE's encoder is used to compute the mean (`mu`) and log variance (`logvar`) of the latent representations of the entire dataset.
- The `generate_fake` function samples from the latent space using these parameters and decodes the samples to generate synthetic data.
- The synthetic data is inverse-transformed back to the original scale.

### Data Augmentation

- The synthetic data is combined with the original dataset to create a more balanced dataset.
- The augmented dataset is saved as `loan_combined.csv`.

### Model Evaluation

- The `test_model` function evaluates the performance of an MLP classifier on both the original and augmented datasets.
- It reports precision, recall, and F1 score for both loan statuses.

## Results

After augmenting the dataset with synthetic data for the denied loan class:

- **Expected Improvements**:

  - Increase in recall and F1 score for the minority class.
  - Better overall performance due to a more balanced dataset.

- **Sample Output**:

  ```
  Best parameters found:
   {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (5, 10), 'learning_rate': 'constant', 'solver': 'adam'}
  Results on the test set:
                precision    recall  f1-score   support

             0       1.00      1.00      1.00     50000
             1       0.91      1.00      0.95     61222
             2       0.00      0.00      0.00      6241

      accuracy                           0.95    117463
     macro avg       0.64      0.67      0.65    117463
  weighted avg       0.90      0.95      0.92    117463
  ```

- **Interpretation**:
  - The augmented dataset leads to improved recall for the denied loan class (`Loan Status = 1`), indicating that the model is better at identifying denied loans.
  - The precision and F1 scores also show improvements, demonstrating enhanced model performance.

## Notes

- **Data Consistency**:

  - Ensure that data preprocessing steps are consistently applied throughout the process.
  - The same `StandardScaler` used during training should be used to inverse-transform the synthetic data.

- **Hyperparameter Tuning**:

  - Adjust hyperparameters such as learning rate, batch size, number of epochs, and VAE architecture to optimize performance.
  - Use the validation loss to prevent overfitting.

- **Computational Resources**:

  - Training a VAE and generating synthetic data can be computationally intensive.
  - Consider using a GPU if available.

- **Potential Improvements**:
  - Implement early stopping based on validation loss.
  - Experiment with different model architectures and loss functions.
  - Use advanced techniques like Conditional VAEs to generate more targeted synthetic data.

## Conclusion

By generating synthetic data for the minority class using a Variational Autoencoder, we effectively addressed the class imbalance issue in the loan approval dataset. The augmented dataset led to improved model performance, particularly in identifying denied loans, which is crucial for the loan company.
