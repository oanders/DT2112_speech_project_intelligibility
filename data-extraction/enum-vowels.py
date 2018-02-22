import os
from praatio import tgio

vowels = ['U', 'O', '}:', 'E:', 'a', 'o:', '2', 'A:', 'y:', 'u0', 'i:', 'e:', 'u:', '2:', 'ael', 'ale', 'oe', 'E', '@', 'e', 'Y']
path = "data/"
filenames = (item for item in os.listdir(path) if item.endswith(".TextGrid") and item.startswith("sampa"))
res = []
for filename in filenames:
    t = tgio.openTextgrid(path+filename)
    n = 0
    if "vowels" in t.tierDict:
        t.tierDict["vowels"].remove()
    t.addTier(t.tierDict["sent - phones"].new(name="vowels"))
    for i in range(len(t.tierDict["vowels"].entryList)):
        if t.tierDict["vowels"].entryList[i].label.strip() in vowels:
            lbl = str(n)
            n += 1
        else:
            lbl = ""
        t.tierDict["vowels"].entryList[i] = tgio.Interval(t.tierDict["vowels"].entryList[i][0],t.tierDict["vowels"].entryList[i][1],lbl)
    
    t.save(path+filename)


for i in res:
    print(i)