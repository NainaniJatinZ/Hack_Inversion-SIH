from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
app = Flask(__name__)
CORS(app)

filePath = ''

@app.route('/', methods= ['GET', 'POST'])
def lander():
    # if request.method == "GET":
    return render_template("home.html")


@app.route('/visualize', methods= ['GET', 'POST'])
def get_message():
    # if request.method == "GET":
    print("Got request in main function")

    x = pd.read_csv(filePath)
    dates = x['Date'].values.tolist()
    closed = x['Volume'].values.tolist()
    return render_template("index.html",values=closed, labels=dates)





@app.route('/upload_static_file', methods=['POST'])
def upload_static_file():
    global filePath
    print("Got request in static files")
    print(request.files)
    f = request.files['static_file']
    f.save('receivedCSV/'+f.filename)

    filePath = 'receivedCSV/'+f.filename

    resp = {"success": True, "response": "file saved!"}
    return jsonify(resp), 200


if __name__ == "__main__":



    app.run()