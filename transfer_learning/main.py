import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from .train import train_model
from .test import test_model
from torchvision import models


# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Set up Transforms (train, val, and test)
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),  # VGG16 expects the input size to be 224x224
            # transforms.RandomHorizontalFlip(),  # Augmentation for training
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # Ensure val/test images are resized to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Ensure test images are resized to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
}


def transfer_learning(
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    data_path: str,
    data_dir: str = "imagedata-50",
) -> None:
    """
    Main function for transfer learning
    """

    image_datasets = {
        "train": datasets.ImageFolder(
            f"{data_path}/{data_dir}/train", data_transforms["train"]
        ),
        "val": datasets.ImageFolder(
            f"{data_path}/{data_dir}/val", data_transforms["val"]
        ),
        "test": datasets.ImageFolder(
            f"{data_path}/{data_dir}/test", data_transforms["test"]
        ),
    }

    train_loader = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        image_datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        image_datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Class names
    class_names = image_datasets["train"].classes

    # Using the VGG16 model for transfer learning
    # 1. Get trained model weights
    # 2. Freeze layers so they won't all be trained again with our data
    # 3. Replace top layer classifier with a classifier for our 3 categories

    model = models.vgg16(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False  # Freeze all the convolutional layers

    # Replace the top layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, len(class_names))

    # Train model with these hyperparameters
    criterion = nn.CrossEntropyLoss()  # Use Cross Entropy for classification
    optimizer = optim.SGD(
        model.classifier.parameters(), lr=0.001, momentum=0.9
    )  # Only training classifier layer
    train_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )  # Decay LR by 0.1 every 7 epochs

    trained_model = train_model(
        model,
        criterion,
        optimizer,
        train_lr_scheduler,
        train_loader,
        val_loader,
        device=device,
        num_epochs=num_epochs,
    )

    # Save the model weights
    torch.save(
        trained_model.state_dict(),
        f"weights/transfer_learning_{model.__class__.__name__}_{num_epochs}.pth",
    )  # Save weights to a file

    test_model(test_loader, trained_model, class_names)
