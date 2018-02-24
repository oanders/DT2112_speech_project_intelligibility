import pandas
import os

def readData():
    currentDir = os.getcwd()
    path = os.path.join(currentDir, "data-extraction/formant-values.xlsx")
    df = pandas.read_excel(path)
    return df

def main():
    df = readData()
    print(df["ID_rec"])
main()