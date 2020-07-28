# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 09:52:06 2020

@author: matti
"""
"""
#################################################################################
Genetic algorithm that maximizes STAAC predict function of peptide. 
Inputs:
    nGenerations: Number of generations that the algorithm should run
    mutationFrequency: Proportion of offsprings that will mutate
    nParents: Proportion of population that will reproduce
    popsize: Size of population 
    n_splitsCrossover: Currently unavailable, will allways equal 2
Note timecomplexity scales with popsize*nGenerations*nParents
As the PseAAC function is rather slow one could use only the TriSeq model by\
    changing the import predict command to "from STAAC_predict import triseq_predict as predict"
As the peptides will be generated from the aminoAcids list, one could simply remove either \
    amino acid from the list. 
    
Deptendencies: pandas, numpy, itertools, pickle, kears, sklearn, math
#################################################################################
"""
import pandas as pd
import numpy as np
import itertools
from STAAC_predict import predict

aminoAcids = ["A","R","N","C","D","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

class Peptide(object):
    #initialize the object
    def __init__(self, sequence, fitness):
        self.sequence = sequence #wether the best path at each iteration should be mutated or not
        self.fitness = fitness #wether the best and worst paths for each iteration should be shown
    
    def fitness(self):
        self.fitness = predict([self.sequence])[0][0]
    
        
 
def random_peptide():
    lenth = np.random.randint(18,31)
    peptide = ''.join(np.random.choice(aminoAcids, size=lenth, replace=True))
    return peptide
       
def mutate(peptides):
    mutations = []
    for peptide in peptides:
        seq = peptide
        indx = np.random.randint(-2, len(seq) + 3)
        while (len(seq) == 15 and indx < 0) or (len(seq) == 30 and indx >= 30):
            indx = np.random.randint(-2, len(seq) + 3)
        if indx < 0:
            seq = seq[:-1]
        elif indx == len(seq) + 2:
            seq = random_peptide()
        elif indx >= len(seq):
            seq = seq + np.random.choice(aminoAcids)
        else:
            val = np.random.choice(aminoAcids)
            while val == seq[indx]:
                val = np.random.choice(aminoAcids)
            lseq = list(seq)
            lseq[indx] = val
            seq = ''.join(lseq)
        mutations.append(seq)
    return mutations

def randomCrossover(peptide_lst, n_splits):
    peptide_output = []
    for peptide1, peptide2 in peptide_lst:
        min_length = min(len(peptide1.sequence), len(peptide2.sequence))
        split = sorted(np.random.choice(np.arange(min_length), size=2, replace=False), reverse=True)
        offspring1 = peptide1.sequence[:split[0]] + peptide2.sequence[split[0]:split[1]] + peptide1.sequence[split[1]:]
        offspring2 = peptide2.sequence[:split[0]] + peptide1.sequence[split[0]:split[1]] + peptide2.sequence[split[1]:]
        peptide_output.append(offspring1)
        peptide_output.append(offspring2)
    return peptide_output


def mating(parents, numOffsprings, n_splits):
    mates = [x for x in itertools.combinations(parents, 2)]
    numMates = len(mates)
    pairs = [mates[i % numMates] for i in range(numOffsprings // 2 +1)]
    offsprings = randomCrossover(pairs, n_splits)[:numOffsprings]
    return offsprings
    
def GA(mutationFrequency = 0.1, popsize=50, nGenerations=5, nParents=0.25, n_splitsCrossover=2):
    pop = [random_peptide() for _ in range(popsize)]
    fitness = predict(pop)[:,1]
    pop = [Peptide(pop[x], fitness[x]) for x in range(popsize)]
    parents = []
    for generation in range(nGenerations):
        print('#'*30)
        print('Generation:', generation)
        selectionOrder = sorted(pop, key=lambda x: x.fitness, reverse=True)
        parents = selectionOrder[:int(nParents*popsize)]
        print('Top 5 Parents:')
        for parent in parents[:5]:
            print(parent.sequence, 'P=', parent.fitness)
            
        offsprings = mating(parents, popsize - len(parents), n_splitsCrossover)
        mutateindxs = np.random.choice([True, False], size=len(offsprings), p=[mutationFrequency, 1-mutationFrequency])
        pop = parents
        toMutate = []
        to_fitness = []
        for i in range(len(offsprings)):
            if len(offsprings[i]) > 30:
                offsprings[i] = random_peptide()
            if mutateindxs[i]:
                toMutate.append(offsprings[i])
            else:
                to_fitness.append(offsprings[i])
        to_fitness = to_fitness + mutate(toMutate)
        fitness = predict(to_fitness)[:,1]
        pop = pop + [Peptide(to_fitness[i], fitness[i]) for i in range(len(to_fitness))]
    return pd.DataFrame([[x.sequence, x.fitness] for x in selectionOrder], columns=['Peptide', 'Fitness'])
 
  
if __name__ == '__main__':
    GA_df = GA(nGenerations=2, mutationFrequency=0.4, popsize=8, n_splitsCrossover=2)
    print(GA_df)
