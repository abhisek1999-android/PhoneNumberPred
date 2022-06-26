from http.client import responses
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
from flask import Flask, render_template
from flask import request
from flask import jsonify
from flask import Flask, render_template
from flask import request
from flask import jsonify
from random import seed
import urllib.request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

seed(100)
tf.random.set_seed(100)
np.random.seed(100)

img_width = 200
img_height = 50
max_length = 10

print(tf.__version__)

prediction_model1 = tf.keras.models.load_model("model/model17052022.h5")
# Mapping characters to integers
characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
char_to_num = layers.StringLookup(
    vocabulary=characters, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def encode_single_sample(img_path):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    # label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return img


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(
            num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


@app.route('/', methods=['GET', 'POST'])
def home():
    title = 'Home'
    data = request.get_json()
    print(data['url'])
    return data['url']


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    data = request.get_json()
    # just pass the url
    urllib.request.urlretrieve(
        data['url'], "local-filename.jpg")

    file = "local-filename.jpg"
    # img = cv2.imread(file)
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_image = np.invert(gray_image)
    # cv2.imwrite(file,gray_image)
    img = encode_single_sample(file)
    # img = img["image"]
    # Convert to Tensor of type float32 for example
    image_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    # Add dimension to match with input mode
    image_tensor = tf.expand_dims(image_tensor, 0)
    preds = prediction_model1.predict(image_tensor)
    pred_texts = decode_batch_predictions(preds)
    print(pred_texts)
    print(file)
    response = {'phone_number': pred_texts}
    return response


if __name__ == '__main__':

    app.run(debug=True)
