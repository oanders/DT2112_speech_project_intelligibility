import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

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

def cluster(df_B, df_J, df_O):
    df_B_clean = df_B[df_B["include"] != 0]
    df_J_clean = df_J[df_J["include"] != 0]
    df_O_clean = df_O[df_O["include"] != 0]

    X_O = df_O_clean[["F1", "F2"]].values
    X_B = df_B_clean[["F1", "F2"]].values
    X_J = df_J_clean[["F1", "F2"]].values

    # print("running tsne")
    # tsne = TSNE(n_components=2, random_state=0, perplexity=35)
    # xo = tsne.fit_transform(X_O)
    # xb = tsne.fit_transform(X_B)
    # xj = tsne.fit_transform(X_J)

    xo = X_O
    xj = X_J
    xb = X_B

    print("clustering")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=7)
    lo = clusterer.fit_predict(xo)
    lj = clusterer.fit_predict(xj)
    lb = clusterer.fit_predict(xb)



    print("Plotting data")
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(1,3,1) 
    ax1.set_xlabel('F1', fontsize = 15)
    ax1.set_ylabel('F2', fontsize = 15)
    ax1.set_title("O", fontsize = 20)

    ax1.scatter(xo[:,0]
                , xo[:,1]
                , c = lo
                , s = 10)
    ax1.grid()
    ax2 = fig.add_subplot(1,3,2) 
    ax2.set_xlabel('F1', fontsize = 15)
    ax2.set_ylabel('F2', fontsize = 15)
    ax2.set_title("J", fontsize = 20)

    ax2.scatter(xj[:,0]
                , xj[:,1]
                , c = lj
                , s = 10)
    ax2.grid()

    ax3 = fig.add_subplot(1,3,3) 
    ax3.set_xlabel('F1', fontsize = 15)
    ax3.set_ylabel('F2', fontsize = 15)
    ax3.set_title("B", fontsize = 20)

    ax3.scatter(xb[:,0]
                , xb[:,1]
                , c = lb
                , s = 10)
    ax3.grid()
    plt.show()
    



def main():
    df = readData()
    
    # Sepparate the data sets
    df_B = df[df['speaker'] == "B"]
    df_J = df[df['speaker'] == "J"]
    df_O = df[df['speaker'] == "O"]

    print("Size B", df_B.shape)
    print("Size J", df_J.shape)
    print("Size O", df_O.shape)

    print("Size B without zeros", df_B[df_B["include"] != 0].shape)
    print("Size J without zeros", df_J[df_J["include"] != 0].shape)
    print("Size O without zeros", df_O[df_O["include"] != 0].shape)


    # Imput missing values for F2 with mean
    imput(df_B)
    imput(df_J)
    imput(df_O)

    cluster(df_B, df_J, df_O)

    print("Fitting O to O")
    model1(df_O, df_O)
    print("--------------------------\nFitting J to O")
    model1(df_J, df_O)
    print("--------------------------\nFitting B to O")
    model1(df_B, df_O)

    #saveDFs(df_B, df_J, df_O)


main()