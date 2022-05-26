#!/usr/bin/env python
# coding: utf-8

# # New York Stock Exhange Stock Price Predictions RNN-LSTM

# 
# ### Dataset consists of following files:
# 
# ###### prices.csv: raw, as-is daily prices. Most of data spans from 2010 to the end 2016, for companies new on stock market date range is shorter. There have been approx. 140 stock splits in that time, this set doesn't account for that.
# 
# ###### prices-split-adjusted.csv: same as prices, but there have been added adjustments for splits.
# 
# ###### securities.csv: general description of each company with division on sectors
# 
# ######  fundamentals.csv: metrics extracted from annual SEC 10K fillings (2012-2016), should be enough to derive most of popular fundamental indicators.

# In[2]:


#importing important libraries
import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings('ignore')


# In[3]:


#calling the file in nyse named prices.csv
df =pd.read_csv("prices.csv", header=0)
df.head()


# In[4]:


#size of the dataset in 851264 rows and 7 columns
print(df.shape)


# In[5]:


# takes 20 different and unique from symbol
df.symbol.unique()[0:20]


# In[6]:


# finind the length of column named symbol
print(len(df.symbol.values))


# In[7]:


#gives the details of each columns of the dataset like mean, max etc
df.describe()


# In[8]:


#checking whether their is any null value in the dataset
# .sum() will give the total no. of null value column vise 
df.isnull().sum()


# In[9]:


# taking all the unique or one time value in the date column 
df.date.unique()


# In[23]:


#calling the file in nyse named securities.csv
comp_info = pd.read_csv('securities.csv')
comp_info.head()


# In[11]:


# taking total no. of unique values in column Ticket symbol
comp_info["Ticker symbol"].nunique()


# In[12]:


# for locating specific data here.... in security column of string that starts with "Face"
comp_info.loc[comp_info.Security.str.startswith('Face') , :]


# In[18]:


# here we locate Ticker symbol of company with security like Yahoo, Xerox, Adobe etc 
comp_plot = comp_info.loc[(comp_info["Security"] == 'Yahoo Inc.') | (comp_info["Security"] == 'Xerox Corp.') | (comp_info["Security"] == 'Adobe Systems Inc')
              | (comp_info["Security"] == 'Microsoft Corp.') | (comp_info["Security"] == 'Facebook') | (comp_info["Security"] == 'Goldman Sachs Group') , ["Ticker symbol"] ]["Ticker symbol"]
print(comp_plot)


# In[28]:


def plotter(code):
    
    global closing_stock ,opening_stock
    
    f, axs = plt.subplots(2,2,figsize=(8,8))
    
    plt.subplot(212)
    
    company = df[df['symbol']==code]
    
    company = company.open.values.astype('float32')
    
    company = company.reshape(-1, 1)
     
    opening_stock = company
    
    
    plt.grid(True)
    plt.xlabel('Time') 
    plt.ylabel(code + " open stock prices")
    plt.title('prices Vs Time')
    plt.plot(company , 'g') 
    
    
    plt.subplot(211)
    
    company_close = df[df['symbol']==code]
    
    company_close = company_close.close.values.astype('float32')
    
    company_close = company_close.reshape(-1, 1)
   
    closing_stock = company_close
    
    
    plt.xlabel('Time') 
    plt.ylabel(code + " close stock prices")
    plt.title('prices Vs Time') 
    plt.grid(True)
    plt.plot(company_close , 'b')
    plt.show() 
company = df[df['symbol']=='ADBE']
company = company.open.values.astype('float32')
company = company.reshape(-1, 1)
print(company)
# calling the graphs through the function    
#for i in comp_plot:
#    plotter(i)


# **Lets take a single stock as a sample to forecast further stock prices.**

# In[19]:


#taking the values of closing_stock in a single list
closing_stock[:,0]


# In[20]:


#taking the values of closing_stock in a single list called stocks
stocks = closing_stock[: , 0]
print(stocks)
#reshaping the stocks in 1D array form
stocks = stocks.reshape(len(stocks) , 1)


# **Feature scaling the vector for better model performance.**

# In[21]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1)) #scaling features between 0 and 1
stocks = scaler.fit_transform(stocks) # it will start learning algo and then provide scaled and dimension reduced output


# In[22]:


# Fit transform image
from IPython.display import Image
Image(filename="fittrans.jpg")


# In[23]:


train = int(len(stocks) * 0.80) #creating sizes of train and taking 80% percentage of the part
test = len(stocks) - train #creating sizes of test as total minus train


# In[24]:


#sizes of train and test
print(train , test)


# In[25]:


#divinding the values of stocks data to train from 0 to 1409 i.e 80% data
train = stocks[0:train]
print(train)


# In[26]:


#divinding the values of stocks data to test from train ending to stock data ending i.e rest 20% data
test = stocks[len(train) : ]


# In[27]:


#reshaping train data in 1D array form
train = train.reshape(len(train) , 1)
#reshaping test data in 1D array form
test = test.reshape(len(test) , 1)


# In[28]:


#new train and test array shape
print(train.shape , test.shape)


# In[29]:


#creating function to create trainX,testX and target(trainY, testY)
def process_data(data , n_features):
    dataX, dataY = [], [] 
    for i in range(len(data)-n_features-1):
        
        a = data[i:(i+n_features), 0]
        
        dataX.append(a) 
        
        dataY.append(data[i + n_features, 0])
        
    return np.array(dataX), np.array(dataY)

#so the stucture of trainX and trainY is somehow like this
# trainX=[[i1 , i2,...., i n_features ]] and trainY=[i + n_features]  
# trainY will show the future value of trainX values


# In[30]:


n_features = 2

trainX, trainY = process_data(train, n_features)

testX, testY = process_data(test, n_features)


# In[31]:


# printing the structure of train X,Y and test X,Y
print(trainX.shape , trainY.shape , testX.shape , testY.shape)


# In[32]:


# reshaping trainX and testX to use in deeplearning model
trainX = trainX.reshape(trainX.shape[0] , 1 ,trainX.shape[1])
testX = testX.reshape(testX.shape[0] , 1 ,testX.shape[1])


# In[33]:


# Image of LSTM RNN
from IPython.display import Image
Image(filename="Lstm.png")


# In[34]:


# Image of GRU RNN
from IPython.display import Image
Image(filename="gru.png")


# In[35]:


# image of Activation
from IPython.display import Image
Image(filename="activation.png")


# In[36]:


import math 
from keras.models import Sequential
from keras.layers import Dense , BatchNormalization , Dropout , Activation # types of layers
from keras.layers import LSTM , GRU 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error 
from keras.optimizers import Adam , SGD , RMSprop 


# In[37]:


from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint


# In[38]:


filepath="stock_weights1.hdf5"


lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.0001, patience=1, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')


# In[39]:


# creating model for training data using sequential to give series wise output between layers
model = Sequential()

model.add(GRU(256 , input_shape = (1 , n_features) , return_sequences=True))

model.add(Dropout(0.4))

model.add(LSTM(256))

model.add(Dropout(0.4))

model.add(Dense(64 ,  activation = 'relu'))
model.add(Dense(1))

print(model.summary())


# In[40]:


# selecting the loss measurement metrics and optimizer for our model , to find out mean square error
model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.0005) , metrics = ['mean_squared_error'])


# In[41]:


# fitting the data i.e training the trainX, to relate to trainY
history = model.fit(trainX, trainY, epochs=100 , batch_size = 128 , 
          callbacks = [checkpoint , lr_reduce] , validation_data = (testX,testY))    
#callbacks are proper


# In[42]:


# image of mean square error
from IPython.display import Image
Image(filename="meansquareerror.png")


# In[43]:


#predicting the value for testX
pred = model.predict(testX)

pred = scaler.inverse_transform(pred)

pred[:10]
# taking pred from 1 to 10


# In[44]:


# reshaping testY in single array
testY = testY.reshape(testY.shape[0] , 1)

testY = scaler.inverse_transform(testY)

testY[:10]


# In[45]:


# ploting the graph of stock prices with time
print("Red - Predicted Stock Prices  ,  Blue - Actual Stock Prices")
plt.rcParams["figure.figsize"] = (15,7)

plt.plot(testY , 'b')

plt.plot(pred , 'r')

plt.xlabel('Time')

plt.ylabel('Stock Prices')

plt.title('Check the accuracy of the model with time')

plt.grid(True)

plt.show()


# In[ ]:




