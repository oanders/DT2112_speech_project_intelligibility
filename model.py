import pandas
import os
import numpy as np
import matplotlib as plt
from sklearn import linear_model

def readData():
    currentDir = os.getcwd()
    path = os.path.join(currentDir, "data-extraction/formant-values.xlsx")
    df = pandas.read_excel(path)
    return df

def getMeanValues(df):
    f1_list = ["F1_1","F1_2","F1_3","F1_4","F1_5"]
    f2_list = ["F2_1","F2_2","F2_3","F2_4","F2_5"]
    df = df.assign(F1_mean=df[f1_list].mean(axis=1))
    df = df.assign(F2_mean=df[f2_list].mean(axis=1))
    return df

#Train model to fit x -> y
def model1(df_X, df_Y):
    X = df_X[["F1_mean", "F2_mean"]].values
    Y = df_Y[["F1_mean", "F2_mean"]].values

    model = linear_model.LinearRegression()
    model.fit(X, Y)

    print(model.coef_)

def saveDFs(df_Be, df_JP, df_O):
    writer = pandas.ExcelWriter("test.xlsx")
    df_Be.to_excel(writer, 'Be')
    df_JP.to_excel(writer, 'JP')
    df_O.to_excel(writer, 'O')
    writer.save()

def main():
    df = readData()
    df = df.replace("--undefined--", np.NaN) #Seems reasonable to just set undefined to 0 for averaging
    
    df_Be = df[df.ID_rec.str.startswith("Be")]
    df_JP = df[df.ID_rec.str.startswith("JP")]
    df_O = df[df.ID_rec.str.startswith("O")]

    df_Be = getMeanValues(df_Be)
    df_JP = getMeanValues(df_JP)
    df_O = getMeanValues(df_O)

    #model1(df_JP, df_O)

    saveDFs(df_Be, df_JP, df_O)


main()