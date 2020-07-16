"""Flask web server serving text_recognizer predictions."""
import os
import json
from io import BytesIO
import base64
from importlib.util import find_spec
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow.keras.backend as K

from api import preprocess

if find_spec("text_recognizer") is None:
    import sys

    sys.path.append(".")

from text_recognizer.datasets import IamLinesDataset
from text_recognizer.line_predictor import LinePredictor  # pylint: disable=wrong-import-position
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
from text_recognizer.character_predictor import CharacterPredictor
import text_recognizer.util as util  # pylint: disable=wrong-import-position

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU

app = Flask(__name__)  # pylint: disable=invalid-name


def convert_b64(image_file):
    buffered = BytesIO()
    image_file.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_base64 = (bytes("data:image/png;base64,", encoding="utf-8") + img_str).decode("ascii")
    return img_base64


@app.route("/", methods=["GET","POST"])
def upload_predict():
    K.clear_session()
    character_model = CharacterPredictor()
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image = Image.open(image_file)
            npimg = np.asarray(image, dtype=np.uint8)
            prediction, confidence = character_model.predict(npimg)
            image_b64 = convert_b64(image)
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=str(np.round(confidence * 100, 2)) + "%",
                image_b64=image_b64,
            )
    return render_template("index.html", prediction=None, confidence=None, image_b64=None)


@app.route("/_get_character", methods=["POST"])
def character_predict():
    """Decodes image and uses it to make prediction."""
    K.clear_session()
    character_model = CharacterPredictor()
    if request.method == "POST":
        char_image = request.data
        if len(char_image) > 20:
            image = preprocess.b64_str_to_np(char_image)
            image = preprocess.crop_img(image)
            # image = preprocess.center_img(image)
            image = preprocess.resize_img(image)
            image = preprocess.min_max_scaler(image, final_range=(0, 1))
            image = preprocess.reshape_array(image)
            prediction, confidence = character_model.predict(image)
            result = {"prediction": prediction, "confidence": str(np.round(confidence * 100, 2)) + "%", "image_b64": 1}
            return json.dumps(result)
        else:
            image_file = request.files["image"]
            if image_file:
                pass
    return render_template("index.html")


@app.route("/healthcheck")
def index():
    """Provide simple health check route."""
    return "Hello, world"


def main():
    """Run the app."""
    app.run(host="0.0.0.0", debug=True, port=5000)  # nosec


if __name__ == "__main__":
    main()
