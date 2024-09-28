# CONTAINS CODE LIKE DATASETS WHICH WILL BE USED BY ALL DIFFERENT APPROACHES
import torchvision
import torch
import wandb

def get_dataset_train():
    return ...

def get_dataset_val():
    return ...


wandb.init(project="computer-vision")
...