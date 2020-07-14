with open("unmc.json", "r") as f:
    raw=f.read()
vec=raw.split("\n\n")
bases=['placeholder']*len(vec)
n=0
for i in vec:
    k=i.split("*") #separate all values in every entry
    entry=k[3].replace(" ","") #remove whitespaces
    entry=entry.replace("\n","")   #remove newlines
    bases[n]=entry  #assign entries to list
    n+=1

standard = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

f=open('basesunmc.txt','w')
for i in bases:
    n = 0
    for p in i:
        if p in standard:
            continue
        else:
            n += 1
    if n == 0:
        f.write(i + '\n')
f.close()