from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd

from helpers import calculate_sma,calculate_ema,calculate_MACD
from pred import prep,create_indicators,train_test,lstm,arima_es,svr,getMAE
import datetime

app = Flask(__name__)
CORS(app)

filePath = ''
isUploaded = False
indexVal = ['-1']
n_future = 2

@app.route('/', methods= ['GET', 'POST'])
def lander():
    global indexVal
    global isUploaded
    global n_future


    gaugeVal = -10
    isUploaded = request.args.get('isUploaded')
    cardVals = [0,0,0,0]
        
    if len(filePath)!=0:
        
        x = pd.read_csv(filePath)

        data = prep(x)
        data = create_indicators(data)
        gaugeVal = data['Custom'].values.tolist()[-1]

        closed = x['Close'].values.tolist()
        high = x['High'].values.tolist()
        low = x['Low'].values.tolist()
        volume = x['Volume'].values.tolist()

        CurrentVals = [int(closed[-1]),int(high[-1]),int(low[-1]),int(volume[-1])]


        
        
        # print("I work")
        dates = x['Date'].values.tolist()
        closed = x['Close'].values.tolist()
        
        cardVals = [closed[-1],high[-1],low[-1],volume[-1]]


        weekly = x.iloc[::5, :]
        monthly = x.iloc[::21, :]
        six_monthly = x.iloc[::42, :]
        yearly = x.iloc[::260, :]

        
        if request.form:
            
            if request.form.getlist('foox'):
                indexVal= request.form.getlist('foox')

            if request.form.getlist('Num'):
                n_future = int(request.form.getlist('Num')[0])
            print(f"Gooo: {n_future}")
        
        # try:
        
        # except:
        #     print("Error Couldnt Print")
        try:
            if indexVal[0] == '1':
                return render_template("landing.html",values= x['Close'].values.tolist()[-5:], labels= x['Date'].values.tolist()[-5:],gaugeVal=gaugeVal,cardVals=cardVals)
            elif indexVal[0] == '2':
                return render_template("landing.html",values=x['Close'].values.tolist()[-7:], labels= x['Date'].values.tolist()[-7:],gaugeVal=gaugeVal,cardVals=cardVals)
            elif indexVal[0] == '3':
                return render_template("landing.html",values=x['Close'].values.tolist()[-30:], labels= x['Date'].values.tolist()[-21:],gaugeVal=gaugeVal,cardVals=cardVals)
            elif indexVal[0] == '4':
                return render_template("landing.html",values=x['Close'].values.tolist()[-183:], labels= x['Date'].values.tolist()[-126:],gaugeVal=gaugeVal,cardVals=cardVals)
            elif indexVal[0] == '5':
                return render_template("landing.html",values=x['Close'].values.tolist()[-365:], labels= x['Date'].values.tolist()[-260:],gaugeVal=gaugeVal,cardVals=cardVals)
            else:
                return render_template("landing.html",values=x['Close'].values.tolist(), labels= x['Date'].values.tolist(),gaugeVal=gaugeVal,cardVals=cardVals)
        except:
            return render_template("landing.html",values=x['Close'].values.tolist(), labels= x['Date'].values.tolist(),gaugeVal=gaugeVal,cardVals=cardVals)
        
    
    return render_template("landing.html",values=[1,2,3], labels=['Jan','Feb','March'],gaugeVal=gaugeVal,cardVals=cardVals)



@app.route('/preds', methods= ['GET', 'POST'])
def makePreds():
    global n_future
    LSTM = [['Jan','Feb','March'],[0,6,1]]
    threeModel = [['Jan','Feb','March'],[1,2,3]]
    SVR = [['Jan','Feb','March'],[3,2,1]]


    

    if filePath:
        x = pd.read_csv(filePath)
        data = prep(x)
        
        dates = data['Date']

        data = create_indicators(data)
        train, test, data = train_test(data)
        print("Me is")
        print(data['Custom'].values.tolist()[-1])
        lstm_out, test_index, prev = lstm(train, test, data,n_future)
        lstm_out = lstm_out.tolist()[:n_future]

        lstmx = [x for x in range(len(prev)+len(lstm_out))]
        
        lstmy = prev + lstm_out 

        # 

        current_date = dates.tolist()[-1]
        current_date_temp = datetime.datetime.strptime(current_date, "%Y-%m-%d")
        newdate = current_date_temp + datetime.timedelta(days=n_future-1)


        start_date, end_date = dates.tolist()[-1], newdate
        daterange = pd.date_range(start_date, end_date)
        date_list = list()
        for single_date in daterange:
            date_list.append(single_date.strftime("%Y-%m-%d"))



        print("Dates Are:")
        print(len(dates.tolist()),len(date_list))

        XFinalVals = dates.tolist()+ date_list

        stats_out, test_index1, prev = arima_es(train, test, data,n_future)
        stats_out = stats_out[:n_future]
        statsx = [x for x in range(len(prev)+len(stats_out))]
        
        statsy = prev + stats_out 
        # stats_out = stats_out.tolist()
        # print(type(stats_out))

        svr_out, test_index_svr, prev = svr(train, test, data,n_future)
        svr_out = svr_out.tolist()[:n_future]

        svrx = [x for x in range(len(prev)+len(svr_out))]
        
        svry = prev + svr_out 
        # print(type(svr_out))
        
        print("Lens Are:")
        print(len(lstm_out),len(stats_out),len(svr_out))



        # CSV
        # print("CSV LEN")
        # print(len(XFinalVals[:len(lstmy)]),len(lstmy))

        LSTMCSV = pd.DataFrame({'Dates':XFinalVals[:len(lstmy)],'Price':lstmy})
        LSTMCSV.to_csv("static/outputs/lstmOutput.csv")

        statsCSV = pd.DataFrame({'Dates':XFinalVals[:len(statsy)],'Price':statsy})
        statsCSV.to_csv("static/outputs/statsOutput.csv")

        svrCSV = pd.DataFrame({'Dates':XFinalVals[:len(svry)],'Price':svry})
        svrCSV.to_csv("static/outputs/svrOutput.csv")

        # LSTMCSV = pd.DataFrame({'Dates':XFinalVals,'Price':lstmy})
        # LSTMCSV = pd.DataFrame({'Dates':XFinalVals,'Price':lstmy})


        # input1,pred_god,test_index = LSTMPred(x)
        return render_template("prediction.html",LSTMx = XFinalVals,LSTMy = lstmy,LSTMprev = prev ,threeModelx = XFinalVals,threeModely = statsy,SVRx = XFinalVals,SVRy = svry)


    return render_template("prediction.html",LSTMx = LSTM[0],LSTMy = LSTM[1],threeModelx = threeModel[0],threeModely = threeModel[1],SVRx = SVR[0],SVRy = SVR[1])


@app.route('/indicators', methods= ['GET', 'POST'])
def indicators():
    global filePath
    LSTM = [['Jan','Feb','March'],[1,2,3]]
    threeModel = [['Jan','Feb','March'],[1,2,3]]
    SVR = [['Jan','Feb','March'],[3,2,1]]

    
    


    if filePath:
        x = pd.read_csv(filePath)
        # pred_god,test_index,input1 = lstm(x)
        
        
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
        ema1 = calculate_ema(dataset['Close'], 12*7)
        ema2 = calculate_ema(dataset['Close'], 26*7)
        macd = list()
        for i in range(len(ema1)):
            macd.append(ema1[i] - ema2[i])
        # macd = calculate_MACD(dataset)
        closed = x['Close'].values.tolist()
        
        
        return render_template("indicators.html",xPlot = dates[182:],y1 = sma[182:],y2 = ema[182:],y3 = closed[182:],y4 = macd[182:])    

    

    return render_template("indicators.html",xPlot = [0,0,0,0],y1 = [1,2,3,4],y2 = [4,3,2,1],y3 = [0,0,0,0],y4 = [1,2,3,4])    


@app.route('/models', methods= ['GET', 'POST'])
def models():
    LSTM = [1.0,20.0,30.0]
    threeModels = [1.0,20.0,40.0]
    SVR = [1.0,10.0,50.0]
    global filePath
    if filePath:
        x = pd.read_csv(filePath)

        data = prep(x)
        data = create_indicators(data)
        train, test, data = train_test(data)

        bgColors = ['white','white','white']
        # best = 0

        MAE_lstm,MAE_stat,MAE_svm, mapeLSTM, mapestat, mapesvr = getMAE(train,test,data)
        LSTM = [round(mapeLSTM, 2)] + LSTM 
        LSTM = [round(MAE_lstm, 2)] + LSTM 
        
        threeModels = [round(mapestat, 2)] + threeModels 
        threeModels = [round(MAE_stat, 2)] + threeModels 

        SVR = [round(mapesvr, 2)] + SVR 
        SVR = [round(MAE_svm, 2)] + SVR 

        if MAE_lstm < MAE_svm and MAE_lstm < MAE_stat:
            bgColors[0] = '#40db5a'
        elif MAE_svm < MAE_lstm and MAE_svm < MAE_stat:
            bgColors[1] = '#40db5a'
        elif MAE_stat < MAE_lstm and MAE_stat < MAE_svm:
            bgColors[2] = '#40db5a'


    return render_template("models.html",LSTM = LSTM, threeModels= threeModels, SVR = SVR,bgColors=bgColors)


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