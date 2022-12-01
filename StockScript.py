
# %%
### Data Collection
import pandas_datareader as pdr
import numpy as np
import pandas as pd

# %%
df=pd.read_csv('AAPL.csv')

# %%
df.head()

# %%
close_prices=df.reset_index()['close']

# %%
close_prices

# %%
import matplotlib.pyplot as plt
plt.plot(close_prices)

# %% [markdown]
# # Scaling and Normalization

# %%
close_prices

# %%
# def mean_norm(df, labels=[]):
#   scale_columns = None

#   if (len(labels) > 1):
#     scale_columns = df[labels]
#     scale_columns = (scale_columns-scale_columns.mean())/scale_columns.std()
#   else:
#     scale_columns = df
#     scale_columns = (scale_columns-scale_columns.mean())/scale_columns.std()
#     # print(scale_columns.head())
#   return (scale_columns)

# # def invert_mean(df, labels=[]):
# #     scale_columns = None

# #   if (len(labels) > 1):
# #     scale_columns = df[labels]
# #     scale_columns = (scale_columns-scale_columns.mean())/scale_columns.std()
# #   else:
# #     scale_columns = df
# #     scale_columns = (scale_columns-scale_columns.mean())/scale_columns.std()
# #     # print(scale_columns.head())
# #   return (scale_columns)

# def scale(df, labels=[]):
#   scale_columns = None

#   if (len(labels) > 1):
#     scale_columns = df[labels]
#     scale_columns = (scale_columns-scale_columns.min())/(scale_columns.max()-scale_columns.min())
#   else:
#     scale_columns = df
#     scale_columns = (scale_columns-scale_columns.min())/(scale_columns.max()-scale_columns.min())
#     # print(scale_columns.head())
#   return (scale_columns)  

# # def invert_scale(df, maxp, minp):
# #   scale_columns = None

# #   if (len(labels) > 1):
# #     scale_columns = df[labels]
# #     scale_columns = scale_columns * (maxp-minp) + minp
# #   else:
# #     scale_columns = df
# #     # print(scale_columns.head())
# #   return (scale_columns)  

# maxp = close_prices.max()
# minp = close_prices.min()


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
close=scaler.fit_transform(np.array(close_prices).reshape(-1,1))
close.shape

# %%
print(close)

# %%
# close = np.array(close).reshape(len(close), 1)
# print(close)

# %% [markdown]
# ## creating train and test sets

# %%

split=int(len(close)*0.7)
print(close[split-5:split,:])
train, test = close[:split,:], close[split:,:1]
# print(test)


# %%
len(train),len(test)

# %% [markdown]
# ### allows us to create a backtesting dataset
# ### according to how long we want to look into
# ### the past to predict a specific day

# %%
import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# %%

time_step = 100
X_train, y_train = create_dataset(train, time_step)
X_test, ytest = create_dataset(test, time_step)

# %%
print(X_train.shape), print(y_train.shape)

# %%
print(X_test.shape), print(ytest.shape)

# %%
# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

# %%
### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# %%
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# %%
model.summary()

# %%
model.summary()

# %%


# %%
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

# %%
import tensorflow as tf

# %%
tf.__version__

# %%
### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

# %%
##Transformback to original form
train_predict= scaler.inverse_transform(train_predict)
test_predict= scaler.inverse_transform(test_predict)


# %%
### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

# %%
### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

# %%
### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(close)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(close)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(close)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(close))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# %%
len(test)
x_input=test[-100:].reshape(1,-1)
x_input.shape

temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input
print(len(temp_input))

# demonstrate prediction for next 30 days
from numpy import array

lst_output=[]
n_steps=100
i=0
days = 30
while(i<days):
    
    if(len(temp_input)>n_steps):
        print("i = ", str(i), "\n")
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        print(len(temp_input), "\n")
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print("i = ", str(i))
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input), "\n")
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)
lst_output = np.array(lst_output)

# %%
day_new=np.arange(1,n_steps+1)
day_pred=np.arange(101,101+days)
import matplotlib.pyplot as plt

plt.plot(day_new, scaler.inverse_transform(close[1158:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))


# %%
df3=close.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# %%
df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)

# %% [markdown]
# # Export to Flask

# %%
import joblib
joblib.dump(model, "stock_price_prediction_model.ml")

# %%



