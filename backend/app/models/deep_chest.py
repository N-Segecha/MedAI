from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

class DeepChestModel:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "deep_chest", "pretrained_model.h5")
        self.model = load_model(model_path)
        self.labels = ["Edema", "Pneumonia", "Nodule", "Effusion", "Atelectasis"]

    def preprocess(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path):
        input_tensor = self.preprocess(image_path)
        predictions = self.model.predict(input_tensor)[0]
        return {label: round(float(score), 3) for label, score in zip(self.labels, predictions)}
