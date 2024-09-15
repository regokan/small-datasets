"""Model training with PyTorch"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from synthetic_data.test import test_model
from synthetic_data.train import Autoencoder, DataBuilder, CustomLoss, generate_fake


def synthetic_data(
    device: torch.device,
    num_epochs: int,
    batch_size: int = 64,
    data_path: str = "data",
    data_dir: str = "loan_continuous.csv",
):
    """
    Main function
    """

    # Load the dataset and split into train/test
    dataset = DataBuilder(f"{data_path}/{data_dir}", train=True)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders for training and validation
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the autoencoder model
    d_in = dataset.x.shape[1]  # Input dimension
    autoencoder = Autoencoder(d_in).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = CustomLoss()

    # Train the model
    autoencoder.train()
    for epoch in range(num_epochs):
        total_train_loss = 0
        for batch in trainloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = autoencoder(batch)
            loss = criterion(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(trainloader)

         # Validation phase
        autoencoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in valloader:
                val_batch = val_batch.to(device)
                recon_batch, mu, logvar = autoencoder(val_batch)
                val_loss = criterion(recon_batch, val_batch, mu, logvar)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(valloader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )

    # Generate fake data using the trained model
    autoencoder.eval()
    with torch.no_grad():
        # Get indices of the training data from the Subset object
        train_indices = trainloader.dataset.indices
        # Access the original dataset and get the training data
        train_data = dataset.x[train_indices].to(device)  # dataset.x is already a tensor
        # Pass the batch of data to the encode method
        mu, logvar = autoencoder.encode(train_data)
        # Access the standardizer from the original dataset
        scaler = dataset.standardizer
        # Generate fake data
        fake_data = generate_fake(mu, logvar, 50000, scaler, autoencoder)


    # Save the fake data to a CSV
    fake_data_df = pd.DataFrame(
        fake_data, columns=[f"Feature_{i}" for i in range(d_in)]
    )
    fake_data_df.to_csv("data/fake_loan_data.csv", index=False)

    # Combine the new fake data with the original dataset
    df_original = pd.read_csv(f"{data_path}/{data_dir}")
    df_combined = pd.concat([df_original, fake_data_df], axis=0)
    df_combined.to_csv(f"{data_path}/loan_combined.csv", index=False)

    # Run test on the combined dataset
    test_model(f"{data_path}/loan_combined.csv")
