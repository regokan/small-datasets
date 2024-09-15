"""Main script for small dataset in machine learning"""

from transfer_learning import transfer_learning
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="transfer_learning",
        help="Type of technique to test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use for training",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to the data folder",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="imagedata-50",
        help="Name of the data folder",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.type == "transfer_learning":
        transfer_learning(
            device=args.device,
            num_epochs=args.epochs,
            data_path=args.data_path,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
