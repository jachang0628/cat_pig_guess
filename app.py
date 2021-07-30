import flask
from flask import Flask, render_template, url_for, request
import numpy as np
import json
import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
import os
import os.path
import base64
from collections import OrderedDict
from PIL import Image
import io

os.chdir(os.path.dirname(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


classes = ["cat", "pig"]
densenet = models.densenet121(pretrained=True)
# define extra layer for densenet
extra_layer = nn.Sequential(
    OrderedDict(
        [
            ("layer1", nn.Linear(1024, 500)),
            ("relu1", nn.ReLU()),
            ("layer2", nn.Linear(500, 1)),
        ]
    )
)
densenet.classifier = extra_layer
densenet.to(device)
# check if model is trained already, if yes then load the model and if not train the model
if os.path.exists("saved_model/full_trained.pt"):
    densenet.load_state_dict(
        torch.load("saved_model/full_trained.pt", map_location=device)
    )
    densenet.eval()


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img64 = request.values["imageBase64"]
        img_enc = img64.split(",")[1]
        img = base64.decodebytes(img_enc.encode("utf-8"))
        image = Image.open(io.BytesIO(img))
        image_np = np.array(image)

    print(image_np)
    print(image_np.shape)

    return "hello"


if __name__ == "__main__":
    app.run(debug=True)
