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
import io
from PIL import Image
import PIL.ImageOps
import cv2

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
if os.path.exists("saved_model/full_trained_v2.pt"):
    densenet.load_state_dict(
        torch.load("saved_model/full_trained_v2.pt", map_location=device)
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
        print(img64)
        img_enc = img64.split(",")[1]
        print("\n")
        print("\n")
        print(img_enc)
        img = base64.b64decode(img_enc)
        im_frame = Image.open(io.BytesIO(img))
        im_frame = PIL.ImageOps.invert(im_frame)
        im_frame = transforms.Resize((64, 64))(im_frame)
        im_frame.show()
        im_frame = transforms.ToTensor()(im_frame)
        print(im_frame.shape)
        pauline2 = im_frame.unsqueeze(0).to(device)
        prediction = int(torch.sigmoid(densenet(pauline2)) > 0.5)
        print(classes[prediction])
        print(
            f"The prediction is {classes[prediction]} with probability of {float(torch.sigmoid(densenet(pauline2))) if classes[prediction] == 'pig' else float(1-torch.sigmoid(densenet(pauline2)))}"
        )

    return "hello"


if __name__ == "__main__":
    app.run(debug=True)
