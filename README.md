# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
STEP 1 : Load the customer segmentation dataset and preprocess the data by removing unnecessary columns and handling missing values.

STEP 2 : Encode categorical features and convert the target classes into numerical labels.

STEP 3 : Split the dataset into training and testing sets and normalize the feature values.

STEP 4 : Build a neural network classification model using fully connected layers and ReLU activation functions.

STEP 5 : Train the neural network using the Cross Entropy Loss function and Adam optimizer.

STEP 6 : Evaluate the trained model using accuracy, confusion matrix, classification report, and predict the class of a new sample.


## PROGRAM

### Name: Kamlesh Y

### Register Number: 212224100029

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("/customers.csv")

df=df.drop(columns=["ID"])

df.columns

df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)

cat_columns=['Gender','Ever_Married', 'Graduated', 'Profession',
    'Spending_Score', 'Var_1']
for col in cat_columns:
    df[col]=LabelEncoder().fit_transform(df[col])

lbe=LabelEncoder()
df["Segmentation"]=lbe.fit_transform(df["Segmentation"])

x=df.drop(columns="Segmentation")
y=df["Segmentation"].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
xt=scaler.fit_transform(xt)
xst=scaler.transform(xst)

xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)

tr=TensorDataset(xt,yt)
tst=TensorDataset(xst,yst)
trl=DataLoader(tr,batch_size=16,shuffle=True)
tstl=DataLoader(tst,batch_size=16)

class classifier1(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,32)
        self.l2=nn.Linear(32,16)
        self.l3=nn.Linear(16,8)
        self.l4=nn.Linear(8,4)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x

model=classifier1(input_size=xt.shape[1])
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)

epochs=100
plt.show()

```

### Dataset Information


<img width="1061" height="268" alt="Screenshot 2026-10-28 043034" src="https://github.com/user-attachments/assets/43dcd53a-5da4-4081-a4a7-ec3fce262d38" />

### OUTPUT

## Confusion Matrix

Include confusion matrix here

## Classification Report
Include classification report here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
Include your result here
