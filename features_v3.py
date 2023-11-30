



import pandas as pd
import math 
import numpy as np
import re
import json
import structureMotif
import to_fasta_file
import sequenceMotif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import targetFeature
# Set display width to None to show full content of columns
pd.set_option('display.max_colwidth', None)


#--------------------Specify input/output files----------------
dataset_filename = 'smallMolecule_aptamers_10172023.csv' #raw data 
target_features_filename = 'targets_feature_vector.csv' #unprocesses 
# dataset_filename = 'aptamers.json'

#df = pd.read_json('aptamers.json')
df = pd.read_csv(dataset_filename)
#--------------------Basic clearning/preprocessing---------------
# Filter rows with non-empty sequences and non-NaN values
df = df[(df['sequence'].str.len() != 0) & (~df['sequence'].isna())].reset_index(drop=True)

#Remove any leading or trailing whitespace
df['sequence'] = df['sequence'].str.strip()
df['target'] = df['target'].str.strip()


#--------------------------Calc gc content------------------------
def calculate_gc_content(sequence):
    '''Calculates the guanine-cytosine ratio present in the sequence'''
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    total_count = len(sequence)
    gc_content = (gc_count / total_count) * 100
    return round(gc_content, 2)

# Apply the function to the Series and create a new Series with the GC content
gc = df['sequence'].apply(calculate_gc_content)

#---------------------Encode sequences----------------------------
#One hot encode sequences 
def one_hot_encoding(df, column_name):
    '''One hot encodes the sequences, creates a new col appending _one_hot. Nothing is returned.
    Modifies dataframe in place.'''
    #Can create a new dataframe instead of doing in place modification by using new_df = df.copy() and returning new_df
    #this helps with troubleshooting but wil consume more memory 

    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    #one_hot = df[column_name].apply(lambda seq: np.array([encoding[nucleotide] for nucleotide in seq])) #2D array
    one_hot = df[column_name].apply(lambda seq: np.concatenate([encoding[nucleotide] for nucleotide in seq]))
    df[column_name + '_one_hot'] = one_hot

one_hot_encoding(df, 'sequence')
#--------------------Encode 1-mer-------------------------------
def calculate_1mer(df, column_name):
    '''Input: a dataframe will all the data, column in the dataframe that we want to encode
        Outout: No return value it will just modify the dataframe directly'''
    def one_mer_freq(sequence):
        nucleotides = ['A', 'C', 'G', 'T']
        freq_dict = {nt: 0 for nt in nucleotides}
        
        for nucleotide in sequence:
            if nucleotide in freq_dict:
                freq_dict[nucleotide] += 1
        
        sequence_length = len(sequence)
        freq_array = np.array([freq_dict[nt] / sequence_length for nt in nucleotides], dtype=np.float64)
        
        # Normalize the frequency array to have unit L2 norm
        feature_vector = freq_array / np.linalg.norm(freq_array)
        
        return feature_vector
    
    # Calculate the 1-mer feature vector for each sequence in the specified column
    df[column_name + '_1mer'] = df[column_name].apply(one_mer_freq)

calculate_1mer(df, 'sequence')

#--------------------Encode 2-mer-----------------------------------

def calculate_2mer(df, column_name):
    '''Function to calculate the 2-mer frequencies for a sequence'''

    def two_mer_freq(sequence):
        nucleotides = ['A', 'C', 'G', 'T']
        freq_dict = {nt1 + nt2: 0 for nt1 in nucleotides for nt2 in nucleotides}
        
        for i in range(len(sequence) - 1):
            kmer = sequence[i:i + 2]
            if kmer in freq_dict:
                freq_dict[kmer] += 1
        
        sequence_length = len(sequence) - 1
        freq_array = np.array([freq_dict[kmer] / sequence_length for kmer in freq_dict], dtype=np.float64)
        
        # Normalize the frequency array to have unit L2 norm
        feature_vector = freq_array / np.linalg.norm(freq_array)
        
        return feature_vector
    
    # Calculate the 2-mer feature vector for each sequence in the specified column
    df[column_name + '_2mer'] = df[column_name].apply(two_mer_freq)

calculate_2mer(df, 'sequence')

#----------------------Encode 3-mer-----------------------

def calculate_3mer(df, column_name):
    '''Calculate the 3-mer frequencies for a sequence'''
    def three_mer_freq(sequence):
        nucleotides = ['A', 'C', 'G', 'T']
        k = 3  # Length of k-mer (in this case, 3)
        freq_dict = {}
        
        # Initialize the frequency dictionary for all possible 3-mers
        for nt1 in nucleotides:
            for nt2 in nucleotides:
                for nt3 in nucleotides:
                    freq_dict[nt1 + nt2 + nt3] = 0
        
        # Count the occurrences of each 3-mer in the sequence
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if kmer in freq_dict:
                freq_dict[kmer] += 1
        
        sequence_length = len(sequence)
        freq_array = np.array([freq_dict[kmer] / sequence_length for kmer in freq_dict.keys()], dtype=np.float64)
        
        # Normalize the frequency array to have unit L2 norm
        feature_vector = freq_array / np.linalg.norm(freq_array)
        
        return feature_vector
    
    # Calculate the 3-mer feature vector for each sequence in the specified column
    df[column_name + '_3mer'] = df[column_name].apply(three_mer_freq)

calculate_3mer(df, 'sequence')

#------------------------Encode 4-mer--------------------------


# import itertools

# def calculate_4mer(df, column_name):
#     def four_mer_freq(sequence):

#         nucleotides = ['A', 'C', 'G', 'T']
#         k = 4
#         freq_dict = {kmer: 0 for kmer in itertools.product(nucleotides, repeat=k)}
        
#         if len(sequence) < k:
#             print(f"Warning: Sequence is too short for {k}-mer calculation: {sequence}")
#             return np.zeros(len(freq_dict), dtype=np.float32)
        
#        # Count the occurrences of each 3-mer in the sequence
#         for i in range(len(sequence) - k + 1):
#             kmer = sequence[i:i+k]
#             if kmer in freq_dict:
#                 freq_dict[kmer] += 1

#         sequence_length = len(sequence) - k + 1
#         freq_array = np.array([freq_dict[kmer] / sequence_length for kmer in freq_dict], dtype=np.float32)
        
#         # Normalize the frequency array to have unit L2 norm
#         feature_vector = freq_array / np.linalg.norm(freq_array)
        
#         return feature_vector
    
#     # Calculate the 4-mer feature vector for each sequence in the specified column
#     df[column_name + '_4mer'] = df[column_name].apply(four_mer_freq)


# calculate_4mer(aptamers, 'sequence')

#---------------------Encode 5-mer------------------------------

# def calculate_5mer(df, column_name):
#     def five_mer_freq(sequence):
#         nucleotides = ['A', 'C', 'G', 'T']
#         k = 5
#         freq_dict = {nt: 0 for nt in nucleotides}
        
#         for i in range(len(sequence) - k + 1):
#             kmer = sequence[i:i+k]
#             if all(nt in nucleotides for nt in kmer):
#                 freq_dict[kmer] += 1
        
#         sequence_length = len(sequence) - k + 1
#         freq_array = np.array([freq_dict[kmer] / sequence_length for kmer in freq_dict], dtype=np.float32)
        
#         # Normalize the frequency array to have unit L2 norm
#         feature_vector = freq_array / np.linalg.norm(freq_array)
        
#         return feature_vector
    
#     # Calculate the 5-mer feature vector for each sequence in the specified column
#     df[column_name + '_5mer'] = df[column_name].apply(five_mer_freq)
# calculate_5mer(aptamers, 'sequence')



#-----------------------Sequence Embeddings---------------------------

# from gensim.models import Word2Vec

# def calculate_word2vec_embeddings(df, column_name, embedding_column_name=None, vector_size=100, window=5, min_count=1, sg=1):
#     # Check if the embedding_column_name is provided, if not, use a default name
#     if embedding_column_name is None:
#         embedding_column_name = f'{column_name}_embeddings'
    
#     # Tokenize the sequences into individual characters (nucleotides)
#     sequences = df[column_name].apply(list).tolist()
    
#     # Create and train the Word2Vec model on the sequences
#     model = Word2Vec(sequences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    
#     # Calculate the embeddings for each sequence
#     embeddings = df[column_name].apply(lambda seq: [model.wv[nucleotide] for nucleotide in seq]) #I remove .tolist() from here
    
#     # Add the embeddings to the DataFrame as a new column
#     df[embedding_column_name] = embeddings
    
#     # No need to return the DataFrame as we are modifying it in-place
#     return None

# calculate_word2vec_embeddings(aptamers, 'sequence', vector_size=100, window=5, min_count=1, sg=1)

from gensim.models import Word2Vec

def calculate_word2vec_embeddings(df, column_name, embedding_column_name=None, vector_size=100, window=5, min_count=1, sg=1):
    # Check if the embedding_column_name is provided, if not, use a default name
    if embedding_column_name is None:
        embedding_column_name = f'{column_name}_embeddings'
    
    # Tokenize the sequences into individual characters (nucleotides)
    sequences = df[column_name].apply(list).tolist()
    
    # Create and train the Word2Vec model on the sequences
    model = Word2Vec(sequences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    
    # Calculate the embeddings for each sequence
    embeddings = df[column_name].apply(lambda seq: [model.wv[nucleotide] for nucleotide in seq])
    
    # Convert the embeddings list to a NumPy array
    embeddings_array = embeddings.apply(np.array).to_numpy()
    
    # Add the embeddings to the DataFrame as a new column
    df[embedding_column_name] = embeddings_array
    
    # No need to return the DataFrame as we are modifying it in-place
    return None

calculate_word2vec_embeddings(df, 'sequence', vector_size=100, window=5, min_count=1, sg=1)

#Testing needed for working with dna2vec. Its generating embeddings for kmers where I define the length of the kmer and also define the padding which is max_sequence_length
# from dna2vec.dna2vec import DNA2Vec
# import numpy as np

# def calculate_dna2vec_kmer_embeddings(df, column_name, kmer_lengths, embedding_column_prefix='', vector_size=100):
#     # Create embeddings for different k-mer lengths and store them in separate columns
#     for kmer_length in kmer_lengths:
#         embedding_column_name = f'{embedding_column_prefix}{kmer_length}_mer_embeddings'
        
#         # Pad or truncate sequences to a fixed length
#         df[f'{column_name}_{kmer_length}_mer'] = df[column_name].apply(lambda seq: seq[:max_sequence_length].ljust(max_sequence_length, 'N'))
        
#         # Tokenize the sequences into k-mers of the specified length
#         sequences = df[f'{column_name}_{kmer_length}_mer'].apply(lambda seq: [seq[i:i+kmer_length] for i in range(len(seq) - kmer_length + 1)]).tolist()
        
#         # Create and train the DNA2Vec model on the k-mers
#         dna2vec_model = DNA2Vec(vector_size=vector_size, window=5, min_count=1, negative=5, workers=4)
#         dna2vec_model.train(sequences)
        
#         # Calculate the embeddings for each k-mer
#         embeddings = df[f'{column_name}_{kmer_length}_mer'].apply(lambda seq: [dna2vec_model[kmer] for kmer in seq if kmer in dna2vec_model])
        
#         # Convert the embeddings list to a NumPy array
#         embeddings_array = embeddings.apply(np.array).to_numpy()
        
#         # Add the embeddings to the DataFrame as a new column
#         df[embedding_column_name] = embeddings_array
    
#     # No need to return the DataFrame as we are modifying it in-place
#     return None

# # Example usage with multiple k-mer lengths:
# kmer_lengths_to_generate = [2, 3, 4]
# calculate_dna2vec_kmer_embeddings(df, 'sequence', kmer_lengths_to_generate, embedding_column_prefix='kmer_', vector_size=100)


#----------------------------------------------------Secondary Structure information-----------------------------------------------------------
#Add secondary structures and mfe 

# Call the compute_mfe_structures function
results = structureMotif.compute_mfe_structures(df['sequence'])


df['structure'] = results['structure'] #one hot encode this
df['mfe'] = results['gibbs_energy'] #need to standardize this
df['matrix'] = results['matrix'] #nupack's matrix representation of the secondary structure
#df['pseudoknots'] = results['pseudoknots']  #no results found skip for now
df['stacking_energy'] = results['stacking_energy'] #Need to standardize 


#-------------------Standarsize mfe---------------------------------
scaler = StandardScaler()  # Create the StandardScaler
scaled_mfe = scaler.fit_transform(df['mfe'].values.reshape(-1, 1))  
df['mfe'] = scaled_mfe

#-------------------Standarsize stacking energy---------------------------------
scaler = StandardScaler()  # Create the StandardScaler
scaled_stacking_energy = scaler.fit_transform(df['stacking_energy'].values.reshape(-1, 1))  
df['stacking_energy'] = scaled_stacking_energy

#-------------------Reshape matrix and pad to same length---------------------------------
# Flatten all arrays in the 'matrix' column of the 'df' DataFrame
df['matrix'] = df['matrix'].apply(lambda arr: arr.flatten())

# Find the maximum length among all flattened arrays
max_length = max(len(arr) for arr in df['matrix'])

# Define a function to pad arrays to the maximum length
def pad_to_max_length(arr, max_length):
    if len(arr) < max_length:
        padding = np.zeros(max_length - len(arr))
        arr = np.concatenate((arr, padding))
    return arr

# Apply the pad_to_max_length function to each flattened array
df['matrix'] = df['matrix'].apply(lambda arr: pad_to_max_length(arr, max_length).astype(np.int64))
#-------------------structure_encoding ->one-hot dbn--------------------------
def calculate_dbn_one_hot(df, column_name, encoding_column_name=None):
    if encoding_column_name is None:
        encoding_column_name = f'{column_name}_one_hot'
    def encode_dbn(dbn_string):
        # Define the characters used in the DBN notation and their corresponding one-hot vectors
        characters = {'(': [1, 0, 0],  # Start of base pair
                      ')': [0, 1, 0],  # End of base pair
                      '.': [0, 0, 1]}  # Unpaired nucleotide

        # Encode each character in the DBN string using the one-hot vectors
        encoded_dbn = np.array([characters[char] for char in dbn_string])
        return encoded_dbn
    df[encoding_column_name] = df[column_name].apply(encode_dbn)

calculate_dbn_one_hot(df, 'structure')




#--------------------pair vs unpaired bases------------------

# def calculate_dbn_features(df, column_name):
#     '''Returns the number of unpaired, and paired bases: [unpaired, paired]'''
#     def count_unpaired_paired(sequence):
#         unpaired_count = sequence.count('.')
#         paired_count = sequence.count('(')
#         return [unpaired_count, paired_count]

#     # Calculate the number of unpaired and paired nucleotides for each sequence in the specified column
#     df['dbn_counts'] = df[column_name].apply(count_unpaired_paired).tolist()

#     return df
# calculate_dbn_features(df, 'structure')



#--------------------Extract secondary substructure counts------------------

# def extract_substructures(sequence):
#     hairpin_loops = []
#     internal_loops = []
#     bulges = []

#     # Track the start and end positions of the substructure
#     start_pos = None

#     for i, char in enumerate(sequence):
#         if char == "(":
#             # Found an opening bracket, start or extend the substructure
#             if start_pos is None:
#                 start_pos = i
#         elif char == ")":
#             # Found a closing bracket, check if we have a substructure
#             if start_pos is not None:
#                 end_pos = i + 1
#                 loop_sequence = sequence[start_pos:end_pos]
#                 if loop_sequence.count(".") >= 2:  # Check for at least two dots on one side of the bracket
#                     if "(" in loop_sequence and ")" in loop_sequence:
#                         internal_loops.append(loop_sequence)
#                     else:
#                         bulges.append(loop_sequence)
#                 else:
#                     hairpin_loops.append(loop_sequence)
            
#             # Reset the start position
#             start_pos = None

#     return hairpin_loops, internal_loops, bulges

# df['hairpin_loops'], df['internal_loops'], df['bulges'] = zip(*df['structure'].apply(extract_substructures))

# print(df['internal_loops'])



#------------------------------------Sequence Motif features derived from MEME----------------------------------------
# #Create fasta file with sequences
# sequence_file = 'sequences.txt
# fasta_file = 'sequences.fasta'
# to_fasta_file.write_sequences_to_file(SEQUENCE.tolist(), sequence_file)
# to_fasta_file.convert_file(sequence_file, fasta_file, TARGET.tolist() )


# clusters = sequenceMotif.get_clusters(TARGET.tolist(), SEQUENCE.tolist()) #Generate clusters based on target
# directory = sequenceMotif.write_clusters_to_files(clusters) #Generate fasta files for each cluster


# sequenceMotif.run_meme_on_files(directory) #Execute meme-suite on each fasta file

# motifs = sequenceMotif.read_meme_files(directory)


# aptamers = sequenceMotif.getMotifs(aptamers, motifs)

# print(aptamers['consensus'])
# #----------One hot encode the consensus sequences---------------------
# one_hot_encoding(aptamers, 'consensus')


#--------------------------------------------------------------Type Encoding----------------------------------------

#------------------Target Name Label encoding------------------------------



# def label_encode_target_name(df, column_name, new_column_suffix='_encoded'):
#     label_encoder = LabelEncoder()
#     encoded_values = label_encoder.fit_transform(df[column_name])
#     new_column_name = column_name + new_column_suffix
#     df[new_column_name] = encoded_values
#     df.drop(columns=[column_name], inplace=True)  # Drop the original column

# label_encode_target_name(df, 'target')



#-------------------Target Type One Hot Encoding-----------------------
def one_hot_encode_target_type(df: pd.DataFrame):
    unique_target_types = df['type'].unique()
    # Create a vocabulary to map each unique target type to a binary NumPy array
    vocabulary = {}
    for target_type in unique_target_types:
        # Create a binary NumPy array instead of a list
        binary_array = np.array([int(target_type == category) for category in unique_target_types])
        vocabulary[target_type] = binary_array

    # Map the 'type' column using the vocabulary to create binary NumPy arrays
    df['target_type_encoded'] = df['type'].map(vocabulary)

    # Convert the 'target_type_encoded' column to a numpy array
    target_type_array = np.array(df['target_type_encoded'].tolist())

    return target_type_array


one_hot_encode_target_type(df)

#----------------------Pad structure arrays to same length----------------------------
df['structure_one_hot'] = df['structure_one_hot'].apply(lambda arr: arr.flatten())
max_length = max(len(arr) for arr in df['structure_one_hot'])
df['structure_one_hot'] = df['structure_one_hot'].apply(lambda arr: pad_to_max_length(arr, max_length).astype(np.int64))


#-------------------------Process sequence based features---------------------------------------
#Drop some cols that we don't need anymore

cols_to_drop = ['type', 'target', 'sequence', 'cid', 'cas', 'reference', 'length', 'structure']

df.drop(columns=cols_to_drop, axis=1, inplace=True)



# Index(['kd', 'sequence_one_hot', 'sequence_1mer', 'sequence_2mer',
#        'sequence_3mer', 'sequence_embeddings', 'mfe', 'matrix',
#        'stacking_energy', 'structure_one_hot', 'target_type_encoded'],
#       dtype='object')

#----------------------standardize Kd----------------------------
scaler = StandardScaler()  # Create the StandardScaler
scaled_kd = scaler.fit_transform(df['kd'].values.reshape(-1, 1))  # Reshape and standardize the 'kd' column
df['kd'] = scaled_kd  # Replace the original 'kd' column with the standardized values

values = df['kd'].values


#----------------------Pad Kd arrays to same length----------------------------
max_length = max(len(arr) for arr in df['sequence_one_hot'])
df['sequence_one_hot'] = df['sequence_one_hot'].apply(lambda arr: pad_to_max_length(arr, max_length).astype(int))
np.set_printoptions(threshold=np.inf)

#----------------------Sequence Embeddings----------------------------------------

df['sequence_embeddings'] = df['sequence_embeddings'].apply(lambda arr: arr.flatten())
max_length = max(len(arr) for arr in df['sequence_embeddings'])
df['sequence_embeddings'] = df['sequence_embeddings'].apply(lambda arr: pad_to_max_length(arr, max_length))

#print(df.columns)
# ['kd', 'sequence_one_hot', 'sequence_1mer', 'sequence_2mer',
#        'sequence_3mer', 'sequence_embeddings', 'mfe', 'matrix',
#        'stacking_energy', 'structure_one_hot', 'target_type_encoded']

#---------------Concatenate All sequence based features---------------------------

sequences = np.vstack(df['sequence_one_hot'].values)
kd = df['kd'].values.reshape(-1, 1)
target_type = np.vstack(df['target_type_encoded'].values)
aptamer_structures = np.hstack([np.vstack(df['structure_one_hot'].values), np.vstack(df['matrix'].values)])
kmers = np.hstack([np.vstack(df['sequence_1mer'].values),np.vstack(df['sequence_2mer'].values), np.vstack(df['sequence_3mer'].values)])
sequence_embedding = np.vstack(df['sequence_embeddings'].values)
binding_energies = np.hstack([df['mfe'].values.reshape(-1, 1), df['stacking_energy'].values.reshape(-1, 1)])
#-------------------------------------------------------------------Target Features-----------------------------------

#target_features_df = targetFeature.create_target_features() #saves as df
target_features_df = pd.read_csv('targets_feature_vector.csv')

#Columns to drop: Exact Mass, smiles, Monoisotopic Mass, Complexity
columns_to_drop = ['Exact Mass', 'Smiles', 'Monoisotopic Mass', 'Complexity']

# Use the drop method to remove the specified columns
target_features_df = target_features_df.drop(columns=columns_to_drop)


#check for Nan values in the dataset
#nan_values = target_features_df.isna().sum()

print()
# Replace nan values with mean of col
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
target_features_df['xLogP3-AA'] = imputer.fit_transform(target_features_df[['xLogP3-AA']])

# Create a single StandardScaler object
scaler = StandardScaler()

# List of columns to standardize
columns_to_standardize = [
    'Mol', 'xLogP3-AA', 'Hydrogen Bond Donor Count',
    'Hydrogen Bond Acceptor Count', 'Rotatable Bond Count',
    'Topological Polar Surface Area', 'Heavy Atom Count',
    'Formal Count', 'Defined Atom Stereocenter Count',
    'Undefined Atom Stereocenter Count', 'Defined Bond Stereocenter Count',
    'Undefined Bond Stereocenter Count', 'Covalently-Bonded Unit Count'
]

# Standardize all selected columns using the same scaler
target_features_df[columns_to_standardize] = scaler.fit_transform(target_features_df[columns_to_standardize])


# # Create a single MinMaxScaler object
# scaler = MinMaxScaler()

# # List of columns to min-max scale
# columns_to_minmax_scale = [
#     'Mol', 'xLogP3-AA', 'Hydrogen Bond Donor Count',
#     'Hydrogen Bond Acceptor Count', 'Rotatable Bond Count',
#     'Topological Polar Surface Area', 'Heavy Atom Count',
#     'Formal Count', 'Defined Atom Stereocenter Count',
#     'Undefined Atom Stereocenter Count', 'Defined Bond Stereocenter Count',
#     'Undefined Bond Stereocenter Count', 'Covalently-Bonded Unit Count'
# ]

# # Min-max scale all selected columns using the same scaler
# target_features_df[columns_to_minmax_scale] = scaler.fit_transform(target_features_df[columns_to_minmax_scale])


# ['Mol', 'Finger Print', 'morgan fingerprint', 'xLogP3-AA',
#        'Hydrogen Bond Donor Count', 'Hydrogen Bond Acceptor Count',
#        'Rotatable Bond Count', 'Topological Polar Surface Area',
#        'Heavy Atom Count', 'Formal Count', 'Isotope Atom Count',
#        'Defined Atom Stereocenter Count', 'Undefined Atom Stereocenter Count',
#        'Defined Bond Stereocenter Count', 'Undefined Bond Stereocenter Count',
#        'Covalently-Bonded Unit Count']
       
       


def hex_to_binary(hex_string, desired_length):
    '''Converts hexadecimal to binary with padding'''
    binary_string = bin(int(hex_string, 16))[2:]
    return binary_string.zfill(desired_length)

max_length = max(len(bin(int(x, 16))[2:]) for x in target_features_df['Finger Print']) # Determine the maximum length of the binary strings
binary_fingerprints = [hex_to_binary(hex_string, max_length) for hex_string in target_features_df['Finger Print']] # Convert and pad each hexadecimal string to binary
fingerprint = np.array([[int(bit) for bit in fingerprint_string] for fingerprint_string in binary_fingerprints]) # Convert the list of binary strings to a uniform NumPy array of integers

def binary_str_to_array(binary_str):
    '''Convert a binary string to a binary array'''
    return np.array([int(bit) for bit in binary_str])

morgan_fingerprint = np.array([binary_str_to_array(fingerprint_str) for fingerprint_str in target_features_df['morgan fingerprint']])


mol_weight = target_features_df['Mol'].to_numpy().reshape(-1, 1)
logp = target_features_df['xLogP3-AA'].to_numpy().reshape(-1, 1)
bond_donor_count = target_features_df['Hydrogen Bond Donor Count'].to_numpy().reshape(-1, 1)
bond_acceptor_count = target_features_df['Hydrogen Bond Acceptor Count'].to_numpy().reshape(-1, 1)
rotatable_bond_count = target_features_df['Rotatable Bond Count'].to_numpy().reshape(-1, 1)
polar_surface_area = target_features_df['Topological Polar Surface Area'].to_numpy().reshape(-1, 1)
formal_count = target_features_df['Formal Count'].to_numpy().reshape(-1, 1)
isotope_atom_count = target_features_df['Isotope Atom Count'].to_numpy().reshape(-1, 1)
defined_atom_stereocenter_count = target_features_df['Defined Atom Stereocenter Count'].to_numpy().reshape(-1, 1)
undefined_atom_stereocenter_count = target_features_df['Undefined Atom Stereocenter Count'].to_numpy().reshape(-1, 1)
defined_bond_stereocenter_count = target_features_df['Defined Bond Stereocenter Count'].to_numpy().reshape(-1, 1)
undefined_bond_stereocenter_count = target_features_df['Undefined Bond Stereocenter Count'].to_numpy().reshape(-1, 1)
covalently_bonded_unit_count = target_features_df['Covalently-Bonded Unit Count'].to_numpy().reshape(-1, 1)



#---------------Concatenate All Target based features---------------------------


target_col_names_to_check = ['Mol', 'Finger Print', 'morgan fingerprint', 'xLogP3-AA',
                             'Hydrogen Bond Donor Count', 'Hydrogen Bond Acceptor Count',
                             'Rotatable Bond Count', 'Topological Polar Surface Area',
                             'Heavy Atom Count', 'Formal Count', 'Isotope Atom Count',
                             'Defined Atom Stereocenter Count', 'Undefined Atom Stereocenter Count',
                             'Defined Bond Stereocenter Count', 'Undefined Bond Stereocenter Count',
                             'Covalently-Bonded Unit Count']


#One hot encoded features: fingerprint, morgan_fingerprint
fingerprints = np.hstack([fingerprint, morgan_fingerprint])
target_molecule_properties = np.hstack([mol_weight, logp, bond_donor_count, bond_acceptor_count, rotatable_bond_count, 
                                      polar_surface_area, formal_count, isotope_atom_count, defined_atom_stereocenter_count, undefined_atom_stereocenter_count, 
                                      defined_bond_stereocenter_count, undefined_bond_stereocenter_count, covalently_bonded_unit_count])

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

pca = PCA().fit(target_molecule_properties)
explained_variance = pca.explained_variance_ratio_ # Explained variance for each component
cumulative_variance = explained_variance.cumsum() # Cumulative explained variance
components_to_keep = np.where(cumulative_variance > 0.95)[0][0] + 1 # Find the number of components that explain at least 95% of the variance
pca = PCA(n_components=components_to_keep) 
pca.fit(target_molecule_properties)
molecule_properties_reduced_data = pca.transform(target_molecule_properties) 
reconstructed_data = pca.inverse_transform(molecule_properties_reduced_data) 
mse = mean_squared_error(target_molecule_properties, reconstructed_data) 
print(f"Reconstruction Error (MSE) for target molecule properties: {mse}")

# Plot the explained variance and cumulative explained variance
# plt.figure(figsize=(10, 6))
# plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Explained Variance', color='blue')
# plt.step(range(1, len(explained_variance) + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance', color='green')
# plt.axvline(components_to_keep, color='red', linestyle='--', label=f'95% explained variance at {components_to_keep} components')
# plt.xlabel('Number of Components')
# plt.ylabel('Explained Variance')
# plt.title('Explained Variance vs. Number of Components')
# plt.legend()
# plt.grid()
# plt.show()

pca = PCA().fit(sequence_embedding)
explained_variance = pca.explained_variance_ratio_ # Explained variance for each component
cumulative_variance = explained_variance.cumsum() # Cumulative explained variance
components_to_keep = np.where(cumulative_variance > 0.95)[0][0] + 1 # Find the number of components that explain at least 95% of the variance
pca = PCA(n_components=components_to_keep) 
pca.fit(sequence_embedding)
sequence_embedding_reduced_data = pca.transform(sequence_embedding) 
reconstructed_data = pca.inverse_transform(sequence_embedding_reduced_data) 
mse = mean_squared_error(sequence_embedding, reconstructed_data) 
print(f"Reconstruction Error (MSE) for sequence embedding: {mse}")

#Plot the explained variance and cumulative explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Explained Variance', color='blue')
plt.step(range(1, len(explained_variance) + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance', color='green')
plt.axvline(components_to_keep, color='red', linestyle='--', label=f'95% explained variance at {components_to_keep} components')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.legend()
plt.grid()
plt.show()

pca = PCA().fit(kmers)
explained_variance = pca.explained_variance_ratio_ # Explained variance for each component
cumulative_variance = explained_variance.cumsum() # Cumulative explained variance
components_to_keep = np.where(cumulative_variance > 0.95)[0][0] + 1 # Find the number of components that explain at least 95% of the variance

pca = PCA(n_components=components_to_keep) 
pca.fit(kmers)
kmers_reduced_data = pca.transform(kmers) 
reconstructed_data = pca.inverse_transform(kmers_reduced_data) 
mse = mean_squared_error(kmers, reconstructed_data) 
print(f"Reconstruction Error (MSE) for kmers: {mse}")


from sklearn.decomposition import SparsePCA
components = sequences.shape[0]//6
sparse_pca = SparsePCA(n_components=components, random_state=42)
sequences_reduced_data = sparse_pca.fit_transform(sequences)
reconstructed_data = sparse_pca.inverse_transform(sequences_reduced_data) 
mse = mean_squared_error(sequences, reconstructed_data) 
print(f"Reconstruction Error (MSE) for sequences: {mse}")


components = target_type.shape[0]//6
sparse_pca = SparsePCA(n_components=components, random_state=42)
target_type_reduced_data = sparse_pca.fit_transform(target_type)
reconstructed_data = sparse_pca.inverse_transform(target_type_reduced_data) 
mse = mean_squared_error(target_type, reconstructed_data) 
print(f"Reconstruction Error (MSE) for target type: {mse}")

components = aptamer_structures.shape[0]//6
sparse_pca = SparsePCA(n_components=components, random_state=42)
aptamer_structures_reduced_data = sparse_pca.fit_transform(aptamer_structures)
reconstructed_data = sparse_pca.inverse_transform(aptamer_structures_reduced_data) 
mse = mean_squared_error(aptamer_structures, reconstructed_data) 
print(f"Reconstruction Error (MSE) for aptamer structures: {mse}")

components = fingerprints.shape[0]//6
sparse_pca = SparsePCA(n_components=components, random_state=42)
fingerprints_reduced_data = sparse_pca.fit_transform(fingerprints)
reconstructed_data = sparse_pca.inverse_transform(aptamer_structures_reduced_data) 
mse = mean_squared_error(fingerprints, reconstructed_data) 
print(f"Reconstruction Error (MSE) for fingerprints: {mse}")


#sequences, kd, target_type, aptamer_structures, kmers, sequence_embedding, binding_energies, fingerprints, target_molecule_properties
print(f"Sequences: {sequences_reduced_data.shape}")
print(f"kd: {kd.shape}")
print(f"target type: {target_type_reduced_data.shape}")
print(f"structures: {aptamer_structures_reduced_data.shape}")
print(f"kmers: {kmers_reduced_data.shape}")
print(f"sequence embedding: {sequence_embedding_reduced_data.shape}")
print(f"binding energy: {binding_energies.shape}")
print(f" fingerprint: {fingerprints_reduced_data.shape}") #This fingerpint was obtained from pubchem
print(f"molecule properties: {molecule_properties_reduced_data.shape}")
print(f"total dim: {sequences_reduced_data.shape[1] + kd.shape[1] + target_type_reduced_data.shape[1] + aptamer_structures_reduced_data.shape[1] + kmers_reduced_data.shape[1] + sequence_embedding_reduced_data.shape[1] + binding_energies.shape[1] + fingerprints_reduced_data.shape[1] + molecule_properties_reduced_data.shape[1]}")

