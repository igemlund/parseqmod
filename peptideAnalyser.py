


##This is the Parseq model created by LiU iGEM 2019
##Jonatan Baggman and Johan Larsson



import tensorflow as tf
import sys
from Bio import SeqIO
import numpy as np


def ParabAmpAnalysis(amp):
    allAA = 'ARNDCQEGHILKMFPSTWYV'
    amp=amp.upper()
    numberOfAA = len(amp)
    ampData = np.zeros([20, 20])
    unknown=0
    for x in range(20):     # checks the first AA and saves it in row
        if allAA[x] == amp[0]:
            dataRow = x
            break
        if x==19:
            unknown=1
    if unknown==0:
        for AACount in range(numberOfAA-1):
            secondAA = amp[AACount+1]
            column = dataRow	# takes the row of the last second AA and puts it in column
            for x in range(20):	 # Checks which is the second AA in the pair and puts its number in row
                if allAA[x] == secondAA:
                    dataRow = x
                    break
                if x==19:
                    unknown=1
            ampData[dataRow][column] += +1
    if unknown>0:
        ampData = np.zeros([20, 20])
    return(ampData,unknown)


def SeqAmpAnalysis(amp):
    allAA = 'ARNDCQEGHILKMFPSTWYV'
    maxPeptideLength = 60
    amp=amp.upper()
    numberOfAA = len(amp)
    ampData = [[0 for col in range(20)] for row in range(maxPeptideLength)]
    tooLong=0
    if numberOfAA > maxPeptideLength:
        tooLong=1
    else:
        for dataRow in range(numberOfAA):     # checks the AA and saves it in its row
            for x in range(20):
                if allAA[x] == amp[dataRow]:
                    ampData[dataRow][x] += 1
                    break
    return ampData,tooLong


def predictSequences(seqs):

    parModelName = 'PeptideParModel.model'
    seqModelName = 'PeptideSeqModel.model'

    parModel = tf.keras.models.load_model(parModelName)
    seqModel = tf.keras.models.load_model(seqModelName)

    repeats=len(seqs)

    parabData =  np.zeros([repeats,20, 20])
    seqData =  np.zeros([repeats,60, 20])
    tooLong=0
    unknown=0
    for x in range(repeats):
        parabData[x],realaa = ParabAmpAnalysis(seqs[x])
        seqData[x],long = SeqAmpAnalysis(seqs[x])
        tooLong+=long
        unknown+=realaa

    if tooLong>0:
        print("\n\n",str(tooLong), " peptides where longer than 60 aa and could not be analysed \n")
    if unknown>0:
        print("\n\n",str(unknown), " peptides contained non-typical aminoacids \n")

    parabData = tf.keras.utils.normalize(parabData, axis=1)
    seqData = tf.keras.utils.normalize(seqData, axis=1)

    parabPred=parModel.predict(parabData)
    seqPred = seqModel.predict(seqData)

    predictions = (parabPred[:,1] + seqPred[:,1])/2

    for x in range(repeats):
        dataSeqSum=sum([sum(row) for row in seqData[x]])
        dataParSum=sum([sum(row) for row in seqData[x]])
        if dataSeqSum==0:
            predictions[x]='NaN'
        if dataParSum==0:
            predictions[x]='NaN'
    return(predictions)


def main():

    args=sys.argv[1:]

    if len(args)<1:
        print("Simple mode, use a fasta file as first argument for standard mode")
        seq=[""]
        seq[0]=input("Peptide sequence:")
        if len(seq[0])<60 and len(seq[0])>1:
            pred=predictSequences(seq)
            print("The sequence is predicted to: ")
            print(pred)
        else:
            print("invalid sequence")
    elif len(args)<2 and len(args)>0:
        filename=sys.argv[1]
        record=SeqIO.parse(filename,'fasta')
        seqs=[pep.seq for pep in record]
        record=SeqIO.parse(filename,'fasta')
        ids=[pep.id for pep in record]

        preds=predictSequences(seqs)

        f=open("predicted"+filename,'w')
        count=0
        for id in ids:
            f.write(id)
            f.write("\n")
            f.write(str(preds[count]))
            f.write("\n")
            count+=1
        f.close()
    else:
        print("Too many arguments")

if __name__ == '__main__':
    main()
