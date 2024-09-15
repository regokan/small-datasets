"""Model training with PyTorch"""

import copy
import torch
from .helper import to_device_precision


def train_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 20,
) -> None:
    """
    Train a model with PyTorch.

    Args:
    - model: The PyTorch model to be trained.
    - criterion: The loss function to be used.
    - optimizer: The optimizer to be used.
    - scheduler: The learning rate scheduler to be used.
    - train_loader: The data loader for the training data.
    - val_loader: The data loader for the validation data.
    - num_epochs: The number of epochs to be trained.
    """
    model = model.to(device)

    highest_accuracy = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    def update_model(loader, training):
        current_loss = 0.0
        current_correct = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(training):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                if training:
                    loss.backward()
                    optimizer.step()

            current_loss += loss.item() * inputs.size(0)
            current_correct += torch.sum(preds == labels.data)
        return current_loss, current_correct

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")

        # train phase
        model.train()
        train_loss, train_correct = update_model(train_loader, True)
        scheduler.step()

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_accuracy = to_device_precision(train_correct, device) / (
            len(train_loader) * train_loader.batch_size
        )
        print(
            f"Phase: Train  Loss: {epoch_train_loss:.4f} Accuracy: {epoch_train_accuracy:.4f}"
        )

        # val phase
        model.eval()
        val_loss, val_correct = update_model(val_loader, False)

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = to_device_precision(val_correct, device) / (
            len(val_loader) * val_loader.batch_size
        )
        print(
            f"Phase: Validation  Loss: {epoch_val_loss:.4f} Accuracy: {epoch_val_accuracy:.4f}"
        )

        if epoch_val_accuracy > highest_accuracy:
            highest_accuracy = epoch_val_accuracy
            best_model_weights = copy.deepcopy(model.state_dict())

    print(f"Training finished. Highest accuracy: {highest_accuracy:.4f}")
    model.load_state_dict(best_model_weights)
    return model
