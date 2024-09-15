"""Model testing with PyTorch"""

import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np


def imshow(inp, title=None):
    """
    Helper function to Imshow for Tensor.

    Args:
    - inp: The input tensor
    - title: The title of the image
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2.0)


def test_model(test_loader, trained_model, class_names):
    """
    Test the trained model

    Args:
    - test_loader: The data loader for the test data
    - trained_model: The trained model
    - class_names: The names of the classes
    """
    print("##### TEST RESULTS #####")
    with torch.no_grad():
        # Get correct device
        device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")
        trained_model = trained_model.to(device)
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inp = torchvision.utils.make_grid(inputs)
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(len(inputs)):
                inp = inputs.data[i]
                imshow(inp, class_names[preds[i]])
