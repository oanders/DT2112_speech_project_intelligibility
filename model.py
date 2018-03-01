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
from sklearn.model_selection import train_test_split
import seaborn

def readData():
    currentDir = os.getcwd()
    path = os.path.join(currentDir, "data" + os.sep +"formant-data.xlsx")
    df = pandas.read_excel(path)
    return df


#Train model to fit x -> y
def model1(df_X, df_Y):

    # Get the readings that are good in both
    good_x = df_X["include"] == 1
    good_y = df_X["include"] == 1
    good = good_x & good_y 

    df_X_good = df_X[good]
    df_Y_good = df_Y[good]

    X = df_X_good[["F1", "F2"]].values
    y = df_Y_good[["F1", "F2"]].values

    model = linear_model.LinearRegression()
    
    # Split into train and test set (Dont see why we would need validation since we have no hyperparameters)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict (Should do on only test set)
    prediction = model.predict(X_test)

    # Get error score
    print("Mean squared error before regression was", mean_squared_error(y, X))
    print("Mean squared error after regression is", mean_squared_error(y_test, prediction))

    return model

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
    df_B_clean = df_B[df_B["include"] == 1]
    df_J_clean = df_J[df_J["include"] == 1]
    df_O_clean = df_O[df_O["include"] == 1]

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

    # print("clustering")
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    # lo = clusterer.fit_predict(xo)
    # lj = clusterer.fit_predict(xj)
    # lb = clusterer.fit_predict(xb)

    # clusterer = KMeans(n_clusters=4)
    # lo = clusterer.fit_predict(xo)
    # lj = clusterer.fit_predict(xj)
    # lb = clusterer.fit_predict(xb)



    print("Plotting data")
    fig1 = plt.figure(None)
    ax1 = fig1.add_subplot(1,3,1) 
    ax1.set_xlabel('F1', fontsize = 15)
    ax1.set_ylabel('F2', fontsize = 15)
    ax1.set_title("O", fontsize = 20)

    ax1.scatter(xo[:,0]
                , xo[:,1]
                #, c = lo
                , s = 10)
    ax1.grid()
    ax1.set_ylim([1000, 3000])
    ax1.set_xlim([0, 2200])
    ax2 = fig1.add_subplot(1,3,2) 
    ax2.set_xlabel('F1', fontsize = 15)
    ax2.set_ylabel('F2', fontsize = 15)
    ax2.set_title("J", fontsize = 20)

    ax2.scatter(xj[:,0]
                , xj[:,1]
                #, c = lj
                , s = 10)
    ax2.grid()
    ax2.set_ylim([1000, 3000])
    ax2.set_xlim([0, 2200])
    ax3 = fig1.add_subplot(1,3,3) 
    ax3.set_xlabel('F1', fontsize = 15)
    ax3.set_ylabel('F2', fontsize = 15)
    ax3.set_title("B", fontsize = 20)

    ax3.scatter(xb[:,0]
                , xb[:,1]
                #, c = lb
                , s = 10)
    ax3.grid()
    ax3.set_ylim([1000, 3000])
    ax3.set_xlim([0, 2200])

    fig2 = plt.figure(None)
    plt.title("All three in one plot")
    plt.xlabel("F1")
    plt.ylabel("F2")
    po = plt.scatter(xo[:,0]
            , xo[:,1]
            , s = 10)
    pj = plt.scatter(xj[:,0]
            , xj[:,1]
            , s = 10)
    pb = plt.scatter(xb[:,0]
            , xb[:,1]
            , s = 10)
    
    plt.legend((po, pj, pb), ("O", "J", "B"))

def plotCompare(bj, bo, df_B, df_J, df_O):
    X_B = df_B[["F1", "F2"]].values
    X_J = df_J[["F1", "F2"]].values
    X_O = df_O[["F1", "F2"]].values


    bj_x = bj.predict(X_B)
    bo_x = bo.predict(X_B)



    print("Plotting data")
    fig1 = plt.figure(None)
    ax1 = fig1.add_subplot(1,2,1) 
    ax1.set_xlabel('F1', fontsize = 15)
    ax1.set_ylabel('F2', fontsize = 15)
    ax1.set_title("BJ", fontsize = 20)

    bj_p = ax1.scatter(bj_x[:,0]
                , bj_x[:,1]
                , c = 'r'
                , s = 10)

    j_p = ax1.scatter(X_J[:,0]
                , X_J[:,1]
                , c = 'b'
                , s = 10)
    ax1.grid()

    ax2 = fig1.add_subplot(1,2,2) 
    ax2.set_xlabel('F1', fontsize = 15)
    ax2.set_ylabel('F2', fontsize = 15)
    ax2.set_title("BO", fontsize = 20)

    bo_p = ax2.scatter(bo_x[:,0]
                , bo_x[:,1]
                , c = 'r'
                , s = 10)

    o_p = ax2.scatter(X_O[:,0]
                , X_O[:,1]
                , c = 'b'
                , s = 10)
    ax2.grid()
    
    plt.legend((bj_p, j_p, bo_p, o_p), ("B->J", "J", "B->O", "O"))



def main():
    df = readData()
    
    # Sepparate the data sets
    df_B = df[df['speaker'] == "B"].reset_index()
    df_J = df[df['speaker'] == "J"].reset_index()
    df_O = df[df['speaker'] == "O"].reset_index()

    print("Size B without zeros", df_B[df_B["include"] != 0].shape)
    print("Size J without zeros", df_J[df_J["include"] != 0].shape)
    print("Size O without zeros", df_O[df_O["include"] != 0].shape)


    # Imput missing values for F2 with mean
    imput(df_B)
    imput(df_J)
    imput(df_O)

    cluster(df_B, df_J, df_O)
    print("\n\n#####################\nWith O as goal\n##################\n")
    print("Fitting O to O")
    oo = model1(df_O, df_O)
    print("--------------------------\nFitting J to O")
    jo = model1(df_J, df_O)
    print("--------------------------\nFitting B to O")
    bo = model1(df_B, df_O)

    print("\n\n#####################\nWith J as goal\n##################\n")
    print("Fitting J to J")
    jj = model1(df_J, df_J)
    print("--------------------------\nFitting O to J")
    oj = model1(df_O, df_J)
    print("--------------------------\nFitting B to J")
    bj = model1(df_B, df_J)

    print("\n\n#####################\nWith B as goal\n##################\n")
    print("Fitting B to B")
    bb = model1(df_B, df_B)
    print("--------------------------\nFitting J to B")
    jb = model1(df_J, df_B)
    print("--------------------------\nFitting O to B")
    ob = model1(df_O, df_B)

    plotCompare(bj, bo, df_B, df_J, df_O)
    plotCompare(oj, ob, df_O, df_J, df_B)

    plt.show()


main()