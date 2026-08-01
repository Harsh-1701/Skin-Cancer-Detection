import torch

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Training Parameters
NUM_EPOCHS = 20

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_CLASSES = 7

MODEL_NAME = "efficientnet_b0"

RANDOM_SEED = 42