# style_transfer.py

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the model only once
@tf.keras.utils.register_keras_serializable()
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Convert PIL image to Tensor
def preprocess_image(pil_image, max_dim=512):
    image = np.array(pil_image.convert("RGB"))
    image = tf.convert_to_tensor(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    return image[tf.newaxis, :]

# Convert Tensor to PIL
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.clip_by_value(tensor, 0, 255)
    tensor = tf.cast(tensor, tf.uint8)
    return Image.fromarray(tensor.numpy()[0])

# Main function to perform style transfer
def perform_style_transfer(pil_content, pil_style):
    model = load_model()
    content_image = preprocess_image(pil_content)
    style_image = preprocess_image(pil_style)
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    result_image = tensor_to_image(stylized_image)
    return result_image
