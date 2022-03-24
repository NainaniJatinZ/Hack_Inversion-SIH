

# import time
# from flask import Flask,request,jsonify
# # import cv2
# # from numpy.core.fromnumeric import argmax
# from werkzeug.datastructures import ImmutableMultiDict
# import numpy as np
# # from test_img import detect
# import os
# import io
# import base64
# import PIL.Image as Image

# from PIL import ImageFile


# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag

# from fuzzywuzzy import fuzz as fz
# from fuzzywuzzy import process as pr
# import csv
# import re

# # import cv2
# import numpy as np

# # import easyocr
# from flask_ngrok import run_with_ngrok

# from flask_ngrok import run_with_ngrok
# from flask import Flask,request,jsonify
# from werkzeug.datastructures import ImmutableMultiDict
# import os



# app = Flask(__name__)

# @app.route('/',methods=['POST'])
# def test():
#     f = request.files['myFiles']
#     print(f)
#     f.save("/content/drive/MyDrive/Mini_Project_Sem_VI/FullBackend/received_images"+f.filename)
#     return 'OK'


# if __name__ == '__main__':
#     # run_with_ngrok(app)
#     app.run()






from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
@app.route('/')
def get_message():
    # if request.method == "GET":
    print("Got request in main function")

    values = [12, 19, 3, 5, 2, 3]
    labels = ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange']
    colors = ['#ff0000','#0000ff','#ffffe0','#008000','#800080','#FFA500']

    return render_template("index.html",values=values, labels=labels, colors=colors)


@app.route('/upload_static_file', methods=['POST'])
def upload_static_file():
    print("Got request in static files")
    print(request.files)
    f = request.files['static_file']
    f.save('receivedCSVs/'+f.filename)
    resp = {"success": True, "response": "file saved!"}
    return jsonify(resp), 200

@app.route('/viz')
def viz():
    
    return render_template('viz.html')


if __name__ == "__main__":
    app.run()