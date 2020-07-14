with open("dbaasptable.json", "r") as f:
    raw = f.read()
vec = raw.split("</tr>")
for i in range(len(vec)-1):
    sect = vec[i].split("white-space:")
    lin = sect[2].split("\n")[0]
    reclut = lin.split(">")[1].upper()  # removes clutter and converts upper case
    reclut = reclut.split("</TD")[0]
    vec[i] = reclut.split(" ")  # splits double entries

#standard aminoacids
standard = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

f=open('basesdbaasp.txt','w')
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
