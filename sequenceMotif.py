
#1. First I want to covnert my sequences into fasta format and write that to a file
#2. Read from that file to provide input for meme
#3. Capture results using Biopython
#4. Decide if I need to use file format or if I can directly feed into a dataframe 
#5. Delete all files that I don't need

import os
from Bio.motifs.matrix import PositionWeightMatrix
from Bio import motifs
from Bio.motifs import meme
import subprocess
from typing import List, Dict, Any
import numpy as np
from Bio.motifs.matrix import PositionSpecificScoringMatrix
import pandas as pd

def get_clusters(targets : list[str], sequences : List[str]) -> Dict:
    '''Cluster sequences based on target names. Targets with similar names are grouped together if the names are distinct by whitespace.'''
    clusters = {}
    for target, sequence in zip(targets, sequences):
        target_parts = target.split()
        target_key = target_parts[0] if target_parts else target #ternary operator  
        target_sequences: list[str] = clusters.setdefault(target_key, []) # Get the list of sequences for the target key      
        target_sequences.append(sequence)   # Append the sequence to the list    
    return clusters


def write_clusters_to_files(clusters : Dict) -> str:
    '''Writes the clusters to fasta file and returns directory where files are stored'''
    directory = "meme_clusters"
    if not os.path.exists(directory):
        os.makedirs(directory)    
    for target, sequences in clusters.items():
        filename = os.path.join(directory, f"{target}.fasta")
        with open(filename, "w") as file:
            for i, sequence in enumerate(sequences, start=1):
                file.write(f">sequence{i}\n{sequence}\n")    
    return directory



def run_meme_on_files(directory : str) -> None:
    for filename in os.listdir(directory):
        if filename.endswith(".fasta"):
            file_path = os.path.join(directory, filename)
            
            # Extract the target name from the fasta file name
            target_name = os.path.splitext(filename)[0]
            
            # Create a new directory for the target name
            target_directory = os.path.join(directory, target_name)
            os.makedirs(target_directory, exist_ok=True)
            
            # Run the meme command with the -oc and -dna options
            command = f"meme {file_path} -dna -oc {target_directory} -nostatus -time 14400 -mod zoops -nmotifs 1 -objfun classic -revcomp -markov_order 0 "
            #command = f"meme {file_path} -dna -oc {target_directory} -nostatus -time 14400 -mod zoops -nmotifs 1 -minw 6 -maxw 50 -objfun classic -revcomp -markov_order 0 "
            
            os.system(command)
            
            # # Remove all files except the XML file within the target directory
            # for file in os.listdir(target_directory):
            #     if not file.endswith(".xml"):
            #         file_path = os.path.join(target_directory, file)
            #         os.remove(file_path)
    
    # # Remove all files with the .fasta extension
    # for filename in os.listdir(directory):
    #     if filename.endswith(".fasta"):
    #         file_path = os.path.join(directory, filename)
    #         os.remove(file_path)






def pssm_to_numpy(pssm_string):
    # Split the PSSM string into rows
    values = pssm_string.strip().split('\n')

    # Initialize an empty matrix
    matrix = []    

    for row in values[1:]:
        row = row.split()
        row_values = []
        for char in row:            
            if char not in ['A:', 'T:', 'C:', 'G:']:                
                if char != '-inf':                    
                    if char.isdigit():
                        row_values.append(char)
                    else:
                        row_values.append(float(char))
                else:
                    row_values.append(-2.0)    
        matrix.append(row_values)   
    return np.array(matrix)


def normalize_pssm_min_max(pssm, precision=2):
    min_value = np.min(pssm)
    max_value = np.max(pssm)
    normalized_pssm = (pssm - min_value) / (max_value - min_value)
    return np.round(normalized_pssm, precision)




def read_meme_files(directory: str) -> List[Dict]:
    results = []    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isdir(file_path):
            directory_name = os.path.basename(file_path)  # Extract the directory name
            for sub_filename in os.listdir(file_path):
                sub_file_path = os.path.join(file_path, sub_filename)
                if os.path.isfile(sub_file_path) and sub_filename.endswith(".xml"):
                    with open(sub_file_path) as f:
                        record = meme.read(f)
                        for motif in record:
                            consensus_sequence = str(motif.consensus)                         
                            data = pssm_to_numpy(str(motif.pssm))
                            norm_data = normalize_pssm_min_max(data)     
                            # Create a dictionary for the current result
                            result = {
                                'target': directory_name,
                                'consensus_sequence': consensus_sequence,
                                'pssm': norm_data
                            }
                            results.append(result)

    return results




# def getMotifs(df, motifs):
#     targets, consensus_sequences, pssms = [], [], []  
#     for values in motifs:
#         target = values['target']
#         consensus_sequence: str = values['consensus_sequence']
#         pssm: np.ndarray = values['pssm']     

#         targets.append(target)
#         consensus_sequences.append(consensus_sequence)
#         pssms.append(pssm.tolist())  

#     # Create a new dataframe with the 'targets', 'consensus_sequences', and 'pssms' lists
#     motifs_df = pd.DataFrame({'target': targets, 'consensus': consensus_sequences, 'pssm': pssms})

    
#     # Merge the motifs_df dataframe onto the aptamers dataframe using an inner merge
#     temp = pd.merge(df, motifs_df, on='target', how='inner')
#     # Update the column names if needed
#     temp.rename(columns={'consensus_sequence': 'consensus', 'pssm': 'pssm'}, inplace=True)

#     # Update the columns of the original DataFrame 'df'
#     df['consensus'] = temp['consensus']
#     df['pssm'] = temp['pssm']

#     return df



def getMotifs(df, motifs):
    # Create dictionaries to store 'consensus' and 'pssm' information for each 'target'
    consensus_dict = {}
    pssm_dict = {}

    # Extract 'consensus_sequence' and 'pssm' information from motifs and store in the dictionaries
    for motif in motifs:
        target = motif.get('target')
        if target:
            consensus_sequence = motif.get('consensus_sequence')
            pssm = motif.get('pssm')

            # Update the dictionaries only if both 'consensus_sequence' and 'pssm' are provided
            if consensus_sequence and pssm is not None:
                consensus_dict[target] = consensus_sequence
                pssm_dict[target] = pssm.tolist()

    # Update 'consensus' and 'pssm' columns in the original DataFrame 'df' based on 'target' values
    df['consensus'] = df['target'].map(consensus_dict)
    df['pssm'] = df['target'].map(pssm_dict)

    return df






if __name__ == '__main__':
    targets = ['A', 'B', 'A', 'C', 'B']
    sequences = ['ATCGGGACG', 'TTGCAGATA', 'GGCCGTAATATTGA', 'CCAGATTAGA', 'CCAGATGAGCAGTA', 'AGAVCGATACA']

    clustered_sequences = get_clusters(targets, sequences)
    #print(clustered_sequences)




