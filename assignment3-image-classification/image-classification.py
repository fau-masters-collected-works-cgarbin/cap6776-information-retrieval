"""Image classification using MobileNet and MobileNetV2.

TensorFlow documentation:
https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet
"""
import os
from pathlib import Path
import tensorflow as tf

IMAGE_FILE = "image.jpg"


def load_and_preprocess_image(image_path):
    """Load and preprocess an image.

    Code based on the example in the TensorFlow documentation:
    https://www.tensorflow.org/guide/saved_model
    """
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)

    # There is also a mobilenet_v2.preprocess_input, but the results are the same
    x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis, ...])
    return x


def predict_image(img, model):
    """Predict the image using the given model."""
    predictions = model.predict(img)
    results = tf.keras.applications.imagenet_utils.decode_predictions(predictions)

    print(model.name)
    for result in results[0]:
        print(f"{result[1]:<20} {result[2]:>6.2%}")
    print("\n")


def main():
    img = load_and_preprocess_image(IMAGE_FILE)

    models = [
        tf.keras.applications.MobileNet(),
        tf.keras.applications.mobilenet_v2.MobileNetV2(),
    ]

    for model in models:
        predict_image(img, model)


if __name__ == "__main__":
    # In case it was started from the debugger (runs from the top-level directory)
    # The code in the script assumes it is run from the assignment1-nltk directory
    asignment_dir = "assignment3-image-classification"
    if Path.cwd().name != asignment_dir:
        os.chdir(asignment_dir)

    main()
