"""Image classification using MobileNet and MobileNetV2"""
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils

IMAGE_FILE = "image.jpg"

def load_and_preprocess_image(image_path):
    """Load and preprocess an image."""
    img = image.load_img(image_path, target_size=(224, 224))
    image_array = image.img_to_array(img)
    # Add a dimension to transform the array into a batch
    image_batch = np.expand_dims(image_array, axis=0)
    image_final = tf.keras.applications.mobilenet.preprocess_input(image_batch)
    return image_final


def predict_image(img, model):
    """Predict the image using the given model."""
    predictions = model.predict(img)
    results = imagenet_utils.decode_predictions(predictions)

    print(model.name)
    for result in results[0]:
        print(f"{result[1]:<20} {result[2]}")
    print("\n")


def main():
    img = load_and_preprocess_image(IMAGE_FILE)
    model = tf.keras.applications.MobileNet()
    predict_image(img, model)
    model = tf.keras.applications.MobileNetV2()
    predict_image(img, model)


if __name__ == "__main__":
    # In case it was started from the debugger (runs from the top-level directory)
    # The code in the script assumes it is run from the assignment1-nltk directory
    asignment_dir = "assignment3-image-classification"
    if Path.cwd().name != asignment_dir:
        os.chdir(asignment_dir)

    main()
