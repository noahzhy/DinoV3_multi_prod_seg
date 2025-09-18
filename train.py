import os, random, sys, time, glob
from pathlib import Path
from datetime import datetime
from argparse import Namespace

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Trainer, TrainingArguments
from transformers.image_utils import load_image
from PIL import Image

from model import FineTunedCropper
from utils import *


cfg = Namespace(**yaml.safe_load(open("config.yaml")))
set_seed(cfg.seed)

model = FineTunedCropper(
    model_name_or_path=cfg.pretrain_dir,
    num_classes=cfg.num_classes,
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(**cfg.TrainingArguments),
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,
)

trainer.train()
