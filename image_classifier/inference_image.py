import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model_path = "animal_classifier_model.keras" 
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog", "Elephant", 
    "Gorilla", "Hippo", "Lizard", "Monkey", "Mouse", 
    "Panda", "Spider", "Tiger", "Zebra"
]

# Preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Predict the class of the image
def predict_animal(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_labels[predicted_class], confidence

# Run inference
if __name__ == "__main__":
    test_image_path = "./demo_images/Tiger-1.jpg"  # Add your image path
    if not os.path.exists(test_image_path):
        print(f"Error: The file {test_image_path} does not exist.")
    else:
        predicted_label, confidence_score = predict_animal(test_image_path)
        print(f"Predicted Animal: {predicted_label} ({confidence_score:.2f} confidence)")
