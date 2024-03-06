# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

In this model we will discuss with a neural network with 3 layers of neurons excluding input . First hidden layer with 3 neurons , Second hidden layer with 17 neurons and final Output layer with 1 neuron to predict the regression case scenario.

we use the relationship between input and output which is 
output = input * 17
and using epoch of about 1000 to train and test the model and finnaly predicting the  output for unseen test case.

## Neural Network Model
![262085896-0a19cc22-3c90-4158-9373-4f8fde080468](https://github.com/SASIRAJ27/basic-nn-model/assets/113497176/5dfaea7f-c76e-4f79-98e5-306ef49ee119)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SASIRAJKUMAR T J
### register number 212222230137
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('dl_ex1').sheet1
data = worksheet.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()

X = df[['Input']].values
y = df[['Output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
X_train1

ai=Sequential([
    Dense(3,activation='relu'),
    Dense(17,activation='relu'),
    Dense(1)
])
ai.compile(optimizer='rmsprop',loss='mse')
ai.fit(X_train1,y_train,epochs=2000)


## Plot the loss
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

## Evaluate the model
X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)

# Prediction
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)

z = [[19]]

z1=Scaler.transform(z)

ai.predict(z1)          #expected output is 323.
```

## Dataset Information
![262087283-821991ae-3771-4ad0-8593-5ffaaa2b2e11](https://github.com/SASIRAJ27/basic-nn-model/assets/113497176/67999f4d-57bf-4926-a6e1-db23b2b0ed32)


## OUTPUT

### Training Loss Vs Iteration Plot
![262087520-9477054d-139e-43c1-aae8-6aa0ea38b8ac](https://github.com/SASIRAJ27/basic-nn-model/assets/113497176/9f8c323e-1da1-4674-851d-ee4274a03cb7)

### Test Data Root Mean Squared Error
![262087622-32c07abc-1d2b-43c9-b268-aebcea2d0398](https://github.com/SASIRAJ27/basic-nn-model/assets/113497176/b67830ea-20ce-4ba5-b3dd-1efc2c2388c0)
![262087714-2946be5a-5d37-4e36-84b9-a1de1cf4ea90](https://github.com/SASIRAJ27/basic-nn-model/assets/113497176/aaa09115-283e-40b9-8ba6-6dbd320ada20)


### New Sample Data Prediction
![262087941-d933ec5a-cdc2-46ee-8b5e-325a401093a5](https://github.com/SASIRAJ27/basic-nn-model/assets/113497176/cca69b8f-a71b-40b9-90a3-ea7ce4a2f5dc)

## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
