import cv2
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMGenerator:

    def __init__(self, model):

        self.model = model

        self.target_layers = [
            model.conv_head
        ]

        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers,
        )

    def generate(
        self,
        image_tensor,
        original_image,
    ):

        grayscale_cam = self.cam(
            input_tensor=image_tensor,
        )[0]

        resized = cv2.resize(
            original_image,
            (224, 224),
        )

        rgb_image = (
            resized.astype(np.float32)
            / 255.0
        )

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True,
        )

        return visualization