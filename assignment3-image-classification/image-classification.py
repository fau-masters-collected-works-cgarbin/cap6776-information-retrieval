import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils

IMAGE_FILE = "image.jpg"


def main():
    img = image.load_img(IMAGE_FILE, target_size=(224, 224))
    print(img)

    # Preprocess the image
    image_array = image.img_to_array(img)
    # Add a dimension to transform the array into a batch
    image_batch = np.expand_dims(image_array, axis=0)
    image_final = tf.keras.applications.mobilenet.preprocess_input(image_batch)

    mnet = tf.keras.applications.mobilenet.MobileNet()
    predictions = mnet.predict(image_final)
    results = imagenet_utils.decode_predictions(predictions)
    print(results)


if __name__ == "__main__":
    # In case it was started from the debugger (runs from the top-level directory)
    # The code in the script assumes it is run from the assignment1-nltk directory
    asignment_dir = "assignment3-image-classification"
    if Path.cwd().name != asignment_dir:
        os.chdir(asignment_dir)

    main()
