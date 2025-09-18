import os

import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from PIL import Image


class FineTunedCropper(torch.nn.Module):
    def __init__(self, model_name_or_path="./pretrain/dinov3-vits16-pretrain", num_classes=10):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            device_map=self.device,
        )
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_classes).to(self.device)

    def forward(self, x):
        if isinstance(x, Image.Image):
            inputs = self.processor(images=x, return_tensors="pt").to(self.device)
            x = inputs['pixel_values']
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)

        outputs = self.model(pixel_values=x)
        logits = self.classifier(outputs.last_hidden_state)
        return logits


if __name__ == "__main__":
    from utils import count_parameters

    model = FineTunedCropper(model_name_or_path="./pretrain/dinov3-vits16-pretrain", num_classes=10)
    count_parameters(model, input_size=[(1, 3, 224, 224)], cpu=True)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(url)

    with torch.inference_mode():
        outputs = model(image)
        print(outputs)
