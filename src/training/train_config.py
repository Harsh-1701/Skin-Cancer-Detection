import torch


# -------------------------------------------------------
# DEVICE
# -------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# -------------------------------------------------------
# TRAINING PARAMETERS
# -------------------------------------------------------

NUM_EPOCHS = 30


# Physical batch size.
# GTX 1650 4 GB cannot safely use a large B4 batch.
BATCH_SIZE = 4


# Gradient accumulation:
# 4 physical batch x 2 accumulation = effective batch 8
GRADIENT_ACCUMULATION_STEPS = 2


# -------------------------------------------------------
# DIFFERENTIAL LEARNING RATES
# -------------------------------------------------------

# Small learning rate for pretrained EfficientNet backbone.
BACKBONE_LR = 1e-5


# Larger learning rate for the new classification head.
CLASSIFIER_LR = 2e-4


# -------------------------------------------------------
# OPTIMIZATION
# -------------------------------------------------------

WEIGHT_DECAY = 1e-4


# -------------------------------------------------------
# LR SCHEDULER
# -------------------------------------------------------

WARMUP_EPOCHS = 3

MIN_LR = 1e-6


# -------------------------------------------------------
# LOSS
# -------------------------------------------------------

LABEL_SMOOTHING = 0.05

CLASS_WEIGHT_POWER = 0.5


# -------------------------------------------------------
# MODEL
# -------------------------------------------------------

NUM_CLASSES = 9

MODEL_NAME = "efficientnet_b4"


# -------------------------------------------------------
# RANDOM SEED
# -------------------------------------------------------

RANDOM_SEED = 42