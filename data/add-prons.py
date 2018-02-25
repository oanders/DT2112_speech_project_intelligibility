import os
from praatio import tgio

with open("rs-prondict.txt","r") as prons:
    prondict = {}
    for line in prons:
        word, trans, _ = line.strip().split("\t",2)
        prondict[word] = trans.replace('"',"'")

path = "praat/"
filenames = (item for item in os.listdir(path) if item.endswith(".TextGrid") and item.startswith("sampa"))
for filename in filenames:
    t = tgio.openTextgrid(path+filename)
    if "trans" in t.tierNameList:
        t.removeTier("trans")
    t.addTier(t.tierDict["sent - words"].new(name="trans"),1)
    for i in range(len(t.tierDict["trans"].entryList)):
        lbl = t.tierDict["trans"].entryList[i].label.strip().lower()
        if lbl in prondict.keys():
            newlbl = prondict[lbl]
        else:
            newlbl = ""
        t.tierDict["trans"].entryList[i] = tgio.Interval(t.tierDict["trans"].entryList[i][0],t.tierDict["trans"].entryList[i][1],newlbl)
    
    t.save(path+filename)