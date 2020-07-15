"""Flask web server serving text_recognizer predictions."""
import os
from io import BytesIO
import base64
from importlib.util import find_spec
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow.keras.backend as K

if find_spec("text_recognizer") is None:
    import sys

    sys.path.append(".")

from text_recognizer.datasets import IamLinesDataset
from text_recognizer.line_predictor import LinePredictor  # pylint: disable=wrong-import-position
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util  # pylint: disable=wrong-import-position

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU

app = Flask(__name__)  # pylint: disable=invalid-name


def convert_b64(image_file):
    buffered = BytesIO()
    image_file.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_base64 = (bytes("data:image/png;base64,", encoding='utf-8') + img_str).decode('ascii')
    return img_base64


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    # K.clear_session()
    # predictor = LinePredictor(IamLinesDataset)
    if request.method == "POST":
        image_file = request.files["image"]

        if image_file:
            image = Image.open(image_file)
            npimg = np.asarray(image, dtype=np.uint8)
            pred, conf = predictor.predict(npimg)
            b64_string = convert_b64(image)
            return render_template("index.html", prediction=pred, confidence=conf, image_filename=b64_string)
            # return render_template("index.html", prediction=pred, confidence=conf, image_filename=jpg_as_text)
    return render_template("index.html", prediction=None, confidence=None, image_filename=None)


@app.route("/healthcheck")
def index():
    """Provide simple health check route."""
    return "Hello, world!"


@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    K.clear_session()
    predictor = LinePredictor()
    image = _load_image()
    pred, conf = predictor.predict(image)
    print(pred, conf)
    print("METRIC confidence {}".format(conf))
    print("METRIC mean_intensity {}".format(image.mean()))
    print("INFO pred {}".format(pred))
    return jsonify({"pred": str(pred), "conf": float(conf)})


def _load_image():
    if request.method == "POST":
        data = request.get_json()
        if data is None:
            return "no json received"
        return util.read_b64_image(data['image'], grayscale=True)
    if request.method == "GET":
        image_url = request.args.get("image_url")
        if image_url is None:
            return "no image_url defined in query string"
        print("INFO url {}".format(image_url))
        return util.read_image(image_url, grayscale=True)
    raise ValueError("Unsupported HTTP method")


def main():
    """Run the app."""
    app.run(host="0.0.0.0", debug=True, port=30000)  # nosec


if __name__ == "__main__":
    K.clear_session()
    predictor = ParagraphTextRecognizer()
    main()
