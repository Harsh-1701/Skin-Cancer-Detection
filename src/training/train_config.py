import torch


# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------
# TRAINING PARAMETERS
# ---------------------------------------------------

NUM_EPOCHS = 20

# B4 is considerably larger than B0.
# Start conservatively to avoid GPU out-of-memory errors.
BATCH_SIZE = 4

LEARNING_RATE = 5e-5

WEIGHT_DECAY = 1e-4


# ---------------------------------------------------
# V2 CLASS CONFIGURATION
# ---------------------------------------------------

NUM_CLASSES = 9

MODEL_NAME = "efficientnet_b4"


# ---------------------------------------------------
# RANDOM SEED
# ---------------------------------------------------

RANDOM_SEED = 42