"""Helper functions"""


def to_device_precision(tensor, device):
    """
    Function to adjust precision for device

    args:
        tensor: torch tensor
        device: torch device
    """
    if device == "mps":
        return tensor.float()  # MPS requires float32
    return tensor.double()  # Use float64 for CPU and CUDA
