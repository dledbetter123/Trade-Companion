import flask
from flask import request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

app = flask.Flask(__name__)

# allows anyone to make a URI request
from flask_cors import CORS
CORS(app)

def generate_prediction(X, model, days_in_future=30, days_in_past=100):
    from numpy import array

    if (days_in_future > 30):
        days_in_future = 30



    if (days_in_past > 200):
        days_in_past = 200

    x_input=X[-days_in_past:].reshape(1,-1)
    x_input.shape
    print(x_input)
    param_input=list(x_input)
    param_input=param_input[0].tolist()
\
    print(len(param_input))

    lst_output=[]
    n_steps = days_in_past
    i=0
    days = days_in_future

    while(i<days):
        
        if(len(param_input)>100):
            #print(param_input)
            x_input=np.array(param_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            param_input.extend(yhat[0].tolist())
            param_input=param_input[1:]
            #print(param_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            param_input.extend(yhat[0].tolist())
            print(len(param_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        

    print("lst output: {}".format(lst_output))
    lst_output = np.array(lst_output)
    return lst_output

@app.route('/')
def home():
    return '<h1> API server is Working </h1>'

@app.route('/predict')
def predict_close():
    
    df=pd.read_csv('AAPL.csv')
    close_prices=df.reset_index()['close']
    scaler=MinMaxScaler(feature_range=(0,1))
    close=scaler.fit_transform(np.array(close_prices).reshape(-1,1))

    model = joblib.load("stock_price_prediction_model.ml")
    lst = generate_prediction(close, model)
    lst = scaler.inverse_transform(lst)
    print(lst[:5])
    return str(lst[:5])

app.run()