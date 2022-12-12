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

    if (days_in_future > 90):
        days_in_future = 90

    if (days_in_past > 500):
        days_in_past = 500

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
    lst = generate_prediction(close, model, days_in_future=90)
    lst = scaler.inverse_transform(lst)
    lst = lst.reshape(-1,).tolist()
    # print(close_prices)

    close=scaler.inverse_transform(close)
    X_base = [_ for _ in range(len(close_prices))]
    X_pred = [i + len(close_prices) for i in range(len(lst))]
    # X_pred = [0 for _ in range(len(close_prices))] + X_pred
    print(X_pred)
    # print(lst[:5])
    vals = {
        "base": np.array(close_prices).tolist(),
        "prediction": lst,
        "bplot_x": X_base,
        "pplot_x": X_pred,
        "all": np.array(close_prices).tolist() + lst
    }
    return vals

@app.route('/profile')
def my_profile():
    response_body = {
        "name": "David",
        "about" :"Hello! I'm a full stack developer that loves python and javascript"
    }

    return response_body


# app.run(host="localhost", port=8080, debug=True)
app.run()