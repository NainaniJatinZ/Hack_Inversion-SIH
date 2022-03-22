from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
app = Flask(__name__)
CORS(app)

@app.route('/', methods= ['GET', 'POST'])
def lander():
    # if request.method == "GET":
    return render_template("home.html")


@app.route('/visualize', methods= ['GET', 'POST'])
def get_message():
    # if request.method == "GET":
    print("Got request in main function")

    values = [2.269,2.299,2.284,2.279,2.326]
    labels = ['2012-03-12','2012-03-13','2012-03-14','2012-03-15','2012-03-16']


    colors = ['#ff0000','#0000ff','#ffffe0','#008000','#800080','#FFA500']

    return render_template("index.html",values=closed, labels=dates, colors=colors)





@app.route('/upload_static_file', methods=['POST'])
def upload_static_file():
    print("Got request in static files")
    print(request.files)
    f = request.files['static_file']
    f.save('receivedCSV/'+f.filename)
    resp = {"success": True, "response": "file saved!"}
    return jsonify(resp), 200


if __name__ == "__main__":
    x = pd.read_csv('/Users/anishpawar/dev/SIH/Repo/Hack_Inversion-SIH/backend/local/22.3.22-9pm/nymex_4ind.csv')
    dates = x['Date'].values.tolist()
    # dates = dates[:1000]


    closed = x['Close'].values.tolist()
    # closed = closed[:1000]


    app.run()