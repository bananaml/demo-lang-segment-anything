from lang_sam import LangSAM
import torch
import logging

def download_model():
    model = LangSAM()

    # Print PyTorch and CUDA versions
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
     # Log PyTorch and CUDA versions
    logging.info("PyTorch version: %s", torch.__version__)
    logging.info("CUDA version: %s", torch.version.cuda)

    # Check GPU availability
    logging.info("Number of GPUs available: %d", torch.cuda.device_count())
    if torch.cuda.is_available():
        logging.info("GPU device name: %s", torch.cuda.get_device_name(0))
    else:
        logging.info("No GPU available.")
    # Check GPU availability
    print("Number of GPUs available:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU device name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available.")

if __name__ == "__main__":
    download_model()