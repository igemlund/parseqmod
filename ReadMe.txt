This is the Parseq model created by LiU iGEM 2019
Jonatan Baggman and Johan Larsson

Parseq is a model able to predict the antimicrobial activity of peptides.

The model consist of two neural networks, the model files, which together
creates the full parseq model.

Use the script peptideAnalyser to evaluate peptides with the model.

The script is based on python 3 and uses the libraries tensorflow,
biopython.seqIO, sys and numpy.

To use the parseq model, make sure python 3 is installed together with
tensorflow and biopython.

To run the script in simple mode, just start the script.
Simple mode allows manual input of a single peptide and evaluates the peptides

To run the script in standard mode, run the script with a file as arguments
ex ...\folder> python peptideAnalyser.py pep.fasta
Standard mode reads a fasta file and evaluates every peptide in the file. The
output is written in another file named predicted'filename'.

The parseq model output is values between 0 and 1.
predictions > 0.5 are antimicrobial and vice versa.

The model can only handle peptides with less then 60 amino acids or peptides 
containting nonstandard or ambiguous amino acids. These will be not be predicted
and nan will be returned instead.

Read more about the parseq model on the wiki
https://2019.igem.org/Team:Linkoping_Sweden/Model
