with open("basesdbaasp.txt", "r") as f:
    raw1 = f.read()
with open("basesunmc.txt", "r") as f:
    raw2 = f.read()

pt1 = raw1.split("\n")
pt2 = raw2.split("\n")

allentries = pt1+pt2

distinct = list(set(allentries)) #remove duplicates

f=open('antifungalAMP.txt','w')
for i in distinct:
    f.write(i + '\n')
f.close()