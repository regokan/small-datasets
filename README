# Small Datasets in Machine Learning: Techniques and Applications

## Overview

This project demonstrates practical techniques to handle small datasets in machine learning:

1. **Synthetic Data Generation** using Variational Autoencoders (VAEs).
2. **Transfer Learning** with pre-trained convolutional neural networks.

The `main.py` script allows you to select and run either of these techniques, providing a hands-on experience with data augmentation and leveraging pre-trained models to improve performance on limited data.

## Project Structure

- **`main.py`**: The main script that serves as the entry point for the project. It allows you to choose which technique to run via command-line arguments.

- **`synthetic_data/`**: Directory containing modules and scripts related to synthetic data generation.

  - **`train.py`**: Contains code for training the VAE.
  - **`test.py`**: Contains code for testing the model's performance.
  - **`README.md`**: Detailed documentation for the synthetic data generation module.

- **`transfer_learning/`**: Directory containing modules and scripts for transfer learning.

  - **`train.py`**: Contains the training loop and functions for transfer learning.
  - **`test.py`**: Contains code for testing and visualizing model predictions.
  - **`helper.py`**: Utility functions for device-specific operations.
  - **`README.md`**: Detailed documentation for the transfer learning module.

- **`data/`**: Directory where the datasets should be placed. This is the default data path used by the scripts.

## Requirements

- **Python 3.x**

- **Libraries**:
  - `torch`
  - `torchvision`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `argparse`

Install the required packages using:

```bash
pipenv install
```

## Usage

### Running the Main Script

The `main.py` script allows you to run either the synthetic data generation or transfer learning technique by specifying the `--type` argument.

**Command Syntax**:

```bash
python main.py --type <technique> --data_dir <data_directory> [options]
```

**Arguments**:

- `--type`: The type of technique to run. Options are `synthetic_data` or `transfer_learning`.
- `--data_dir`: The name of the data directory or file name, depending on the technique.
- `--epochs`: (Optional) Number of epochs to train the model. Default is `20`.
- `--device`: (Optional) Device to use for training (`cuda`, `cpu`, or `mps`). Default is `mps`.
- `--data_path`: (Optional) Path to the data folder. Default is `data`.
- `--batch_size`: (Optional) Batch size for training. Default is `10`.
- `--num_workers`: (Optional) Number of workers for data loading. Default is `4`.

### Examples

#### 1. Synthetic Data Generation

To run the synthetic data generation technique:

```bash
python main.py --type synthetic_data --data_dir loan_continuous.csv --epochs 20 --device cpu
```

- **Data Directory**: Place the `loan_continuous.csv` file inside the `data` directory.
- **Description**: This will train a Variational Autoencoder on the loan dataset to generate synthetic data for the minority class and augment the dataset.

#### 2. Transfer Learning

To run the transfer learning technique:

```bash
python main.py --type transfer_learning --data_dir imagedata-50 --epochs 20 --device cuda --batch_size 32
```

- **Data Directory**: The `imagedata-50` directory should be inside the `data` directory and contain subdirectories for training, validation, and testing images organized by class.
- **Description**: This will perform transfer learning using a pre-trained VGG16 model on the provided image dataset to classify images into three categories.

### Notes

- Ensure that the data directories and files are correctly placed inside the `data` directory.
- The `device` argument should match your hardware capabilities. Use `cuda` for GPU acceleration if available.
- Adjust the `batch_size` and `num_workers` according to your system's capabilities to optimize performance.

## Additional Information

- For detailed information on each technique, refer to the README files inside the respective directories:

  - **Synthetic Data Generation**: `synthetic_data/README.md`
  - **Transfer Learning**: `transfer_learning/README.md`

- The individual READMEs provide comprehensive instructions, code explanations, and insights into how each technique works and how to adjust parameters for better results.

## Instructions

1. **Prepare the Data**:

   - For **synthetic data generation**, place the `loan_continuous.csv` file inside the `data` directory.
   - For **transfer learning**, ensure the `data` directory contains the image data organized into `train`, `val`, and `test` subdirectories, each containing class-specific folders.

2. **Install Dependencies**:

   Install the required Python libraries if you haven't already.

   ```bash
   pip install torch torchvision pandas numpy scikit-learn matplotlib argparse
   ```

3. **Run the Script**:

   Use the command-line examples provided above to run the desired technique.

4. **View Results**:

   - For synthetic data generation, the script will generate synthetic data, augment the dataset, and evaluate model performance.
   - For transfer learning, the script will train a model and display test images with predicted labels.

## Conclusion

This project provides practical solutions for working with small datasets in machine learning. By using synthetic data generation and transfer learning, we can enhance model performance even when data is limited. The modular structure allows you to explore both techniques and understand their applications.

Feel free to explore the code, adjust parameters, and experiment with the techniques to suit your specific needs.
