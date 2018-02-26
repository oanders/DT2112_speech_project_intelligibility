import pandas
import os
import numpy as np
import matplotlib as plt
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error

def readData():
    currentDir = os.getcwd()
    path = os.path.join(currentDir, "data" + os.sep +"formant-data.xlsx")
    df = pandas.read_excel(path)
    return df


#Train model to fit x -> y
def model1(df_X, df_Y):

    df_X_1to3 = df_X.head(86)
    df_Y_1to3 = df_Y.head(86)
    X = df_X_1to3[["F1", "F2"]].values
    Y = df_Y_1to3[["F1", "F2"]].values

    model = linear_model.LinearRegression()
    
    # Train the model
    model.fit(X, Y)

    # Predict (Should do on only test set)
    prediction = model.predict(X)

    # Get error score
    print("Mean squared error is", mean_squared_error(Y, prediction))

def saveDFs(df_B, df_J, df_O):
    writer = pandas.ExcelWriter("test.xlsx")
    df_B.to_excel(writer, 'B')
    df_J.to_excel(writer, 'J')
    df_O.to_excel(writer, 'O')
    writer.save()

# Imput mean for missing F2s
def imput(df):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    df["F2"]=imputer.fit_transform(df[["F2"]]).ravel()

def main():
    df = readData()
    
    # Sepparate the data sets
    df_B = df[df['speaker'] == "B"]
    df_J = df[df['speaker'] == "J"]
    df_O = df[df['speaker'] == "O"]

    # Imput missing values for F2 with mean
    imput(df_B)
    imput(df_J)
    imput(df_O)

    print("Fitting O to O")
    model1(df_O, df_O)
    print("--------------------------\nFitting J to O")
    model1(df_J, df_O)
    print("--------------------------\nFitting B to O")
    model1(df_B, df_O)

    saveDFs(df_B, df_J, df_O)


main()