import torch
import torchvision
from torch import nn
from timm import create_model #type: ignore
from torchvision import transforms

def create_swin_transformer():
    # Load the pretrained Swin Transformer model
    model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=101)
    
    train_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(224),             
                    transforms.RandomHorizontalFlip(),              
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
                    transforms.RandomRotation(15),                  
                    transforms.RandomAffine(degrees=15, scale=(0.8, 1.2)),  
                    transforms.ToTensor(),                         
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])

    return model,train_transforms