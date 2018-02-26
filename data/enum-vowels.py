import os
from praatio import tgio

vowels = ['U', 'O', '}:', 'E:', 'a', 'o:', '2', 'A:', 'y:', 'u', 'i:', 'e:', 'u:', '2:', 'E', '@', 'e', 'Y', 'I']
path = "praat/"
filenames = (item for item in os.listdir(path) if item.endswith(".TextGrid") and item.startswith("sampa"))
res = []
for filename in filenames:
    t = tgio.openTextgrid(path+filename)
    n = 0
    if "vowels" in t.tierNameList:
        t.removeTier("vowels")
    t.addTier(t.tierDict["sent - phones"].new(name="vowels"))
    for i in range(len(t.tierDict["vowels"].entryList)):
        ph = t.tierDict["vowels"].entryList[i].label.strip()
        if "*" in ph: # phonemes marked by a * are too messy and will be excluded
            ph = ph.replace("*","")
        if ph in vowels:
            lbl = str(n)
            n += 1
        else:
            lbl = ""
        t.tierDict["vowels"].entryList[i] = tgio.Interval(t.tierDict["vowels"].entryList[i][0],t.tierDict["vowels"].entryList[i][1],lbl)
    
    t.save(path+filename)
