import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from PIL import Image


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

pretrained_model_name = "./dinov3-convnext-small-pretrain"
pretrained_model_name = "./dinov3-vits16-pretrain"

processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="cuda:0" if torch.cuda.is_available() else "cpu", 
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)

# pooled_output shape: (1, 768)