"""Flask web server serving text_recognizer predictions."""
import os
from importlib.util import find_spec
from flask import Flask, request, jsonify, render_template
import tensorflow.keras.backend as K

if find_spec("text_recognizer") is None:
    import sys

    sys.path.append(".")

from text_recognizer.line_predictor import LinePredictor  # pylint: disable=wrong-import-position
import text_recognizer.util as util  # pylint: disable=wrong-import-position

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU

app = Flask(__name__)  # pylint: disable=invalid-name


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    K.clear_session()
    predictor = LinePredictor()
    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            # image_location = os.path.join("api/static/", image_file.filename)
            # image_file.save(image_location)
            image = util.read_image(image_location, grayscale=True)
            pred, conf = predictor.predict(image)
            print(pred, conf)
            print("METRIC confidence {}".format(conf))
            print("METRIC mean_intensity {}".format(image.mean()))
            print("INFO pred {}".format(pred))
            return render_template("index.html", prediction=1)

    return render_template("index.html", prediction=0)


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
    return render_template("index.html", prediction=pred, confidence=conf)
    # return jsonify({"pred": str(pred), "conf": float(conf)})


def _load_image():
    if request.method == "POST":
        data = request.get_json()
        if data is None:
            image = request.files["image"]
            # return "no json received"
        return util.read_b64_image(data, grayscale=True)
    if request.method == "GET":
        image_url = request.args.get("image_url")
        if image_url is None:
            return "no image_url defined in query string"
        print("INFO url {}".format(image_url))
        return util.read_image(image_url, grayscale=True)
    raise ValueError("Unsupported HTTP method")


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=20000, debug=True)  # nosec


if __name__ == "__main__":
    main()
