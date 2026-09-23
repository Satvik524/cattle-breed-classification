from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import tensorflow as tf

def load_yolo_model():
    return YOLO('yolo26m.pt')

def load_cattle_model():
    model_path = hf_hub_download(
                repo_id="Shogun007/cleaned_cattle_50_breed",
                filename="best_model.keras"
            )
        
    return tf.keras.models.load_model(model_path, compile=False)