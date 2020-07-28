with open("uniprotNoAFP.csv", "r") as f:
    raw = f.read()
dat = raw.split("\n")
vec=[0]*(len(dat)-1)
for i in range(len(dat)-2):
    k=dat[i+1].split(";")
    vec[i]=k[1]
vec = list(set(vec))
del vec[0]

#standard aminoacids
standard = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

f=open('basesuniprotNoAFP.txt','w')
#writes all sequences consisting only of standard aminoacids to file.
for i in vec:
    if isinstance(i,str):
        n=0
        for p in i:
            if p in standard:
                continue
            else:
                n+=1
        if n==0:
            f.write(i+'\n')
    else:
        for k in i:
            n = 0
            for p in k:
                if p in standard:
                    continue
                else:
                    n += 1
            if n == 0:
                f.write(k + '\n')
f.close()