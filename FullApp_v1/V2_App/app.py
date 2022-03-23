from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd

from helpers import calculate_sma,calculate_ema,calculate_MACD
from preds import LSTMPred


app = Flask(__name__)
CORS(app)

filePath = ''
isUploaded = False
indexVal = ['1']

@app.route('/', methods= ['GET', 'POST'])
def lander():
    global indexVal
    global isUploaded
    
    isUploaded = request.args.get('isUploaded')

        
    if len(filePath)!=0:
        
        x = pd.read_csv(filePath)

        closed = x['Close'].values.tolist()
        high = x['High'].values.tolist()
        low = x['Low'].values.tolist()
        volume = x['Volume'].values.tolist()

        CurrentVals = [int(closed[-1]),int(high[-1]),int(low[-1]),int(volume[-1])]


        
        
        # print("I work")
        dates = x['Date'].values.tolist()
        closed = x['Close'].values.tolist()
        


        weekly = x.iloc[::5, :]
        monthly = x.iloc[::21, :]
        six_monthly = x.iloc[::42, :]
        yearly = x.iloc[::260, :]

        
        if request.form:
            indexVal= request.form.getlist('foox')
        
        # print(indexVal[0])
        try:
            if indexVal[0] == '1':
                return render_template("landing.html",values= x['Close'].values.tolist(), labels= x['Date'].values.tolist())
            elif indexVal[0] == '2':
                return render_template("landing.html",values=weekly['Close'].values.tolist(), labels= weekly['Date'].values.tolist())
            elif indexVal[0] == '3':
                return render_template("landing.html",values=monthly['Close'].values.tolist(), labels= monthly['Date'].values.tolist())
            elif indexVal[0] == '4':
                return render_template("landing.html",values=six_monthly['Close'].values.tolist(), labels= six_monthly['Date'].values.tolist())
            elif indexVal[0] == '5':
                return render_template("landing.html",values=yearly['Close'].values.tolist(), labels= yearly['Date'].values.tolist())
        except:
            return render_template("landing.html",values=closed, labels=dates)
        
    
    return render_template("landing.html",values=[1,2,3], labels=['Jan','Feb','March'])



@app.route('/preds', methods= ['GET', 'POST'])
def makePreds():
    
    LSTM = [['Jan','Feb','March'],[0,6,1]]
    threeModel = [['Jan','Feb','March'],[1,2,3]]
    SVR = [['Jan','Feb','March'],[3,2,1]]


    if filePath:
        x = pd.read_csv(filePath)
        input1,pred_god,test_index = LSTMPred(x)
        return render_template("prediction.html",LSTMx = test_index,LSTMy = pred_god,threeModelx = input1,threeModely = threeModel[1],SVRx = SVR[0],SVRy = SVR[1])


    return render_template("prediction.html",LSTMx = LSTM[0],LSTMy = LSTM[1],threeModelx = threeModel[0],threeModely = threeModel[1],SVRx = SVR[0],SVRy = SVR[1])


@app.route('/indicators', methods= ['GET', 'POST'])
def indicators():
    global filePath
    LSTM = [['Jan','Feb','March'],[1,2,3]]
    threeModel = [['Jan','Feb','March'],[1,2,3]]
    SVR = [['Jan','Feb','March'],[3,2,1]]

    
    


    if filePath:
        x = pd.read_csv(filePath)
        input1,pred_god,test_index = LSTMPred(x)
        
        
        dates = x['Date'].values.tolist()
        closed = x['Close'].values.tolist()
        dataset = x

        weekly = x.iloc[::5, :]
        monthly = x.iloc[::21, :]
        six_monthly = x.iloc[::42, :]
        yearly = x.iloc[::260, :]


        if request.form:
            indexVal= request.form.getlist('foox')
        
        # print(indexVal[0])
        try:
            if indexVal[0] == '1':
                dates = x['Date'].values.tolist()
                closed = x['Close'].values.tolist()
                dataset = x
            elif indexVal[0] == '2':
                dates = weekly['Date'].values.tolist()
                closed = weekly['Close'].values.tolist()
                dataset = weekly
            elif indexVal[0] == '3':
                dates = monthly['Date'].values.tolist()
                closed = monthly['Close'].values.tolist()
                dataset = monthly
            elif indexVal[0] == '4':
                dates = six_monthly['Date'].values.tolist()
                closed = six_monthly['Close'].values.tolist()
                dataset = six_monthly
            elif indexVal[0] == '5':
                dates = yearly['Date'].values.tolist()
                closed = yearly['Close'].values.tolist()
                dataset = yearly
        except:
            dates = x['Date'].values.tolist()
            closed = x['Close'].values.tolist()
            dataset = x


        sma = calculate_sma(data_series=dataset['Close'], window_size=21*7)
        ema = calculate_ema(dataset['Close'], 20*7)
        macd = calculate_MACD(dataset)
        # closed = x['Close'].values.tolist()
        
        
        return render_template("indicators.html",xPlot = dates,y1 = sma,y2 = ema,y3 = closed,y4 = macd)    

    

    return render_template("indicators.html",xPlot = [0,0,0,0],y1 = [1,2,3,4],y2 = [4,3,2,1],y3 = [0,0,0,0],y4 = [1,2,3,4])    


@app.route('/models', methods= ['GET', 'POST'])
def models():
    LSTM = [10,20,30]
    threeModels = [10,20,40]
    SVR = [60,10,50]

    return render_template("models.html",LSTM = LSTM, threeModels= threeModels, SVR = SVR)


@app.route('/upload_static_file', methods=['POST'])
def upload_static_file():
    global filePath
    print("Got request in static files")
    print(request.files)
    f = request.files['static_file']
    f.save('receivedCSV/'+f.filename)

    filePath = 'receivedCSV/'+f.filename

    resp = {"success": True, "response": "file saved!"}
    # return jsonify(resp), 200

    x = pd.read_csv(filePath)
    dates = x['Date'].values.tolist()
    closed = x['Volume'].values.tolist()
    return render_template("landing.html",values=closed, labels=dates)


if __name__ == "__main__":



    app.run()