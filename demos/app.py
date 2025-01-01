import gradio as gr 
import os 
import torch
import json 

from model import create_swin_transformer
from timeit import default_timer as timer

with open("class_names.json", "r") as f:
    class_names = json.load(f)

swin_model, swin_transforms = create_swin_transformer()

swin_model.load_state_dict(torch.load('swin.pth', weights_only=True ))

def predict_image(img):
    
    start_time = timer()
    
    img = swin_transforms(img).unsqueeze(0)
    
    swin_model.eval()
    
    with torch.inference_mode():
        pred_probs = torch.softmax(swin_model(img), dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    pred_time = round(timer() - start_time, 5)
    
    return pred_labels_and_probs, pred_time

title = "FoodVision Big"
description = "A Swin Transformer model trained on the Food101 dataset. The model has been trained for 10 epochs and has an accuracy over 80%."

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict_image, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description)

# Launch the demo!
demo.launch()
    