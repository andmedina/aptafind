'''
This script will extract the sequences from a DBN structured file or a FASTA structured file, 
then write results to a txt file
'''

'''
The DBN file format follows:

>sequence_name
sequence
dot-bracket notation
'''

''''
The fasta file format follows:

>sequence_name
sequence
'''

import os

inFile = input("Please enter the name of your input file: ")
outFile = os.path.splitext(inFile)[0] + "_output.txt"

with open(inFile, 'r') as f:
    file = f.readlines()

if len(file) < 3 or ">" in file[2]:  # checks if file is in fasta
    step = 2
elif ">" in file[3]:  # checks if file is in dbn
    step = 3
else:
    raise ValueError("Unrecognized file format. Please check that the file is structured in dbn or fasta.")

sequences = [sequence.strip() for sequence in file[1::step]]

with open(outFile, 'w') as f:
    for seq in sequences:
        f.write(f"{seq}\n")
print("Succes! Output file name:", outFile)