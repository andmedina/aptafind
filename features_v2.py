



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
# Set display width to None to show full content of columns
pd.set_option('display.max_colwidth', None)



#df = pd.read_json('aptamers.json')
df = pd.read_csv('small_molecule_081423.csv')

#Print the columns
#print(df.columns.values)

# Filter rows with non-empty sequences and non-NaN values
df = df[(df['sequence'].str.len() != 0) & (~df['sequence'].isna())].reset_index(drop=True)

#Remove any leading or trailing whitespace
df['sequence'] = df['sequence'].str.strip()
df['target'] = df['target'].str.strip()

#New Columns for new dataframe
#NAME = df['name']
TYPE = df['Type']
TARGET = df['target']
SEQUENCE = df['sequence']
LENGTH = df['length']



def extract_numeric_value(s):
    '''This will extract only the numerical values from any string.'''
    match = re.search(r'\d+[\d\.]*(?:\s*(?:/|\(|\)|·)\s*\d+[\d\.]*)*', str(s))
    if match:
        return match.group()
    else:
        return np.nan

#Get Kd
numeric_values = df['kd'].apply(extract_numeric_value).astype(float)
KD = pd.Series(numeric_values)

def calculate_gc_content(sequence):
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    total_count = len(sequence)
    gc_content = (gc_count / total_count) * 100
    return round(gc_content, 2)


# Apply the function to the Series and create a new Series with the GC content
GC = SEQUENCE.apply(calculate_gc_content)

aptamers = pd.DataFrame({'type': TYPE, 'target': TARGET, 'kd': KD, 'gcContent': GC, 'sequence': SEQUENCE})




#---------------------Encode sequences----------------------------
#One hot encode sequences 
def one_hot_encoding(df, column_name):
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    #one_hot = df[column_name].apply(lambda seq: np.array([encoding[nucleotide] for nucleotide in seq])) #2D array
    one_hot = df[column_name].apply(lambda seq: np.concatenate([encoding[nucleotide] for nucleotide in seq])) #1D arrays for each sequence looks like it's working better for model
    df[column_name + '_one_hot'] = one_hot

one_hot_encoding(aptamers, 'sequence')

#print(aptamers.iloc[:2]['sequence'])

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
        freq_array = np.array([freq_dict[nt] / sequence_length for nt in nucleotides], dtype=np.float32)
        
        # Normalize the frequency array to have unit L2 norm
        feature_vector = freq_array / np.linalg.norm(freq_array)
        
        return feature_vector
    
    # Calculate the 1-mer feature vector for each sequence in the specified column
    df[column_name + '_1mer'] = df[column_name].apply(one_mer_freq)

calculate_1mer(aptamers, 'sequence')

#--------------------Encode 2-mer-----------------------------------

def calculate_2mer(df, column_name):
    # Function to calculate the 2-mer frequencies for a sequence
    def two_mer_freq(sequence):
        nucleotides = ['A', 'C', 'G', 'T']
        freq_dict = {nt1 + nt2: 0 for nt1 in nucleotides for nt2 in nucleotides}
        
        for i in range(len(sequence) - 1):
            kmer = sequence[i:i + 2]
            if kmer in freq_dict:
                freq_dict[kmer] += 1
        
        sequence_length = len(sequence) - 1
        freq_array = np.array([freq_dict[kmer] / sequence_length for kmer in freq_dict], dtype=np.float32)
        
        # Normalize the frequency array to have unit L2 norm
        feature_vector = freq_array / np.linalg.norm(freq_array)
        
        return feature_vector
    
    # Calculate the 2-mer feature vector for each sequence in the specified column
    df[column_name + '_2mer'] = df[column_name].apply(two_mer_freq)

calculate_2mer(aptamers, 'sequence')

#----------------------Encode 3-mer-----------------------

def calculate_3mer(df, column_name):
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
        freq_array = np.array([freq_dict[kmer] / sequence_length for kmer in freq_dict.keys()], dtype=np.float32)
        
        # Normalize the frequency array to have unit L2 norm
        feature_vector = freq_array / np.linalg.norm(freq_array)
        
        return feature_vector
    
    # Calculate the 3-mer feature vector for each sequence in the specified column
    df[column_name + '_3mer'] = df[column_name].apply(three_mer_freq)

calculate_3mer(aptamers, 'sequence')

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

calculate_word2vec_embeddings(aptamers, 'sequence', vector_size=100, window=5, min_count=1, sg=1)




#print(aptamers['sequence_embeddings'].head(1))

#----------------------------------------------------Secondary Structure information-----------------------------------------------------------
#Add secondary structures and mfe 

# Call the compute_mfe_structures function
results = structureMotif.compute_mfe_structures(aptamers['sequence'])

# Assign the MFE structures and energies to the 'aptamers' DataFrame
aptamers['structure'] = results['structure']
aptamers['mfe'] = results['gibbs_energy']
aptamers['matrix'] = results['matrix']
aptamers['pseudoknots'] = results['pseudoknots']
aptamers['stacking_energy'] = results['stacking_energy']

# print(aptamers.columns)
# print(aptamers['matrix'].head(2))
# print(aptamers['mfe'].head(2))

#-------------------DBN_encoding--------------------------

def calculate_dbn_one_hot(df, column_name, encoding_column_name=None):
    if encoding_column_name is None:
        encoding_column_name = f'{column_name}_one_hot'
    def encode_dbn(dbn_string):
        # Define the characters used in the DBN notation and their corresponding one-hot vectors
        characters = {'(': [1, 0, 0],  # Start of base pair
                      ')': [0, 1, 0],  # End of base pair
                      '.': [0, 0, 1]}  # Unpaired nucleotide

        # Encode each character in the DBN string using the one-hot vectors
        encoded_dbn = [characters[char] for char in dbn_string]

        return encoded_dbn

    df[encoding_column_name] = df[column_name].apply(encode_dbn)
# Calculate the one-hot encoding of DBN for each row in the 'aptamers' DataFrame
calculate_dbn_one_hot(aptamers, 'structure')
# print(aptamers['structure'].head(2))
# print(aptamers['structure_one_hot'].head(2))

#--------------------pair vs unpaired bases------------------

def calculate_dbn_features(df, column_name):
    '''Returns the number of unpaired, and paired bases: [unpaired, paired]'''
    def count_unpaired_paired(sequence):
        unpaired_count = sequence.count('.')
        paired_count = sequence.count('(')
        return [unpaired_count, paired_count]

    # Calculate the number of unpaired and paired nucleotides for each sequence in the specified column
    df['dbn_counts'] = df[column_name].apply(count_unpaired_paired).tolist()

    return df
calculate_dbn_features(aptamers, 'structure')
# print(aptamers['dbn_counts'].head(1))


#--------------------secondary substructure counts------------------


def extract_substructures(sequence):
    hairpin_loops = []
    internal_loops = []
    bulges = []

    # Track the start and end positions of the substructure
    start_pos = None

    for i, char in enumerate(sequence):
        if char == "(":
            # Found an opening bracket, start or extend the substructure
            if start_pos is None:
                start_pos = i
        elif char == ")":
            # Found a closing bracket, check if we have a substructure
            if start_pos is not None:
                end_pos = i + 1
                loop_sequence = sequence[start_pos:end_pos]
                if loop_sequence.count(".") >= 2:  # Check for at least two dots on one side of the bracket
                    if "(" in loop_sequence and ")" in loop_sequence:
                        internal_loops.append(loop_sequence)
                    else:
                        bulges.append(loop_sequence)
                else:
                    hairpin_loops.append(loop_sequence)
            
            # Reset the start position
            start_pos = None

    return hairpin_loops, internal_loops, bulges



aptamers['hairpin_loops'], aptamers['internal_loops'], aptamers['bulges'] = zip(*aptamers['structure'].apply(extract_substructures))




#------------------------------------Sequence Motif features----------------------------------------------------
# #Create fasta file with sequences
# sequence_file = 'sequences.txt'
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


#-----------------------------------------------General Features: Target and Type Encoding----------------------------------------

#------------------Target Name Label encoding------------------------------



def label_encode_target_name(df, column_name, new_column_suffix='_encoded'):
    label_encoder = LabelEncoder()
    encoded_values = label_encoder.fit_transform(df[column_name])
    new_column_name = column_name + new_column_suffix
    df[new_column_name] = encoded_values
    df.drop(columns=[column_name], inplace=True)  # Drop the original column

label_encode_target_name(aptamers, 'target')

#print(aptamers['target_encoded'])

#-------------------Target Type One Hot Encoding-----------------------

# def one_hot_encode_target_type(df, column_name):
#     one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name)
#     df = pd.concat([df, one_hot_encoded], axis=1)
#     df.drop(columns=[column_name], inplace=True)  # Drop the original column
#     return df  # Return the modified DataFrame

# #This method will use a copy of the supplied df so i must reassign my original df 
# aptamers = one_hot_encode_target_type(aptamers, 'type')


def one_hot_encode_target_type(df: pd.DataFrame):    
    unique_target_types = df['type'].unique()
    # Create a vocabulary to map each unique target type to a binary array
    vocabulary = {}
    for target_type in unique_target_types:
        binary_array = [int(target_type == category) for category in unique_target_types]
        vocabulary[target_type] = binary_array
    # Map the 'target_type' column using the vocabulary to create binary arrays
    df['target_type_encoded'] = df['type'].map(vocabulary)
    # Convert the 'target_type_encoded' column to a numpy array
    target_type_array = df['target_type_encoded'].tolist()    
    return target_type_array


target_categories = aptamers['type'].value_counts()
target_shape = aptamers['type'].shape
df_encoded = one_hot_encode_target_type(aptamers)


# #----------------------Min-Max Scale: Kd----------------------------

# Create the MinMaxScaler
scaler = MinMaxScaler()

# Reshape the 'kd' column for scaling (necessary because it's a 1D array)
scaled_kd = scaler.fit_transform(aptamers['kd'].values.reshape(-1, 1))

# Replace the original 'kd' column with the scaled values
aptamers['kd'] = scaled_kd


# #--------------------Drop Uneeed Columns------------------------
# aptamers.drop('sequence', axis=1, inplace=True)
# #aptamers.drop('consensus', axis=1, inplace=True)
# #print(print(aptamers.dtypes))
# # print(aptamers.columns)
# # print(aptamers['sequence_one_hot'])


#-----------------Further Preprocessing to flatten arrays------------

sequences = aptamers['sequence_one_hot'].tolist()



def convert_to_numpy_array(value):
    return np.array(value)
target_type = aptamers['target_type_encoded'].apply(convert_to_numpy_array)

#print(target_category = aptamers['target_type_encoded'][0].shape)
# Find the maximum length of sequences and Pad all sequences to have the same length (assuming 0 as the padding value)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = keras.utils.pad_sequences(sequences, maxlen=max_sequence_length, padding='post', value=0)

# aptamers.to_csv('aptamers_encoded.csv', index=False)

#Flatten sequence embedding feature
# sequence_embeddings = aptamers['sequence_embeddings'].apply(lambda arr: arr.flatten())


#padded_sequence_embeddings = keras.utils.pad_sequences(sequence_embeddings, maxlen=max_sequence_length, padding='post', value=0)

#----------------------------------------------------------------------Vae.py--------------------------------------------------------------------------------------

import tensorflow as tf
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Read the data
# aptamers = pd.read_csv('aptamers_encoded.csv')
#input_dim = 392
latent_dim = 64  # Dimensionality of the latent space


#Features shape 
sequence_dim = padded_sequences[0].shape
sequence_1mer_dim = aptamers['sequence_1mer'][0].shape 
sequence_2mer_dim = aptamers['sequence_2mer'][0].shape
sequence_3mer_dim = aptamers['sequence_3mer'][0].shape
#sequence_embed_dim = sequence_embeddings[0].shape
target_type_dim = target_type[0].shape
kd_dim = 1

#concat all of the feature dimensions
input_dim = int(np.sum([sequence_dim[0], sequence_1mer_dim[0], sequence_2mer_dim[0], sequence_3mer_dim[0], target_type_dim[0], kd_dim]))

# Encoder
encoder_inputs = tf.keras.Input(shape=(input_dim,))


# Apply separate encoding layers for each feature
x_encoded = layers.Dense(64, activation="relu")(encoder_inputs)
x_encoded = layers.Dense(32, activation="relu")(x_encoded)

# Define the mean and variance layers
z_mean = layers.Dense(latent_dim, name="z_mean")(x_encoded)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x_encoded)

# Define the sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Create the encoder model
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Define the decoder network
decoder = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(latent_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')  
])

# Define the input layer
vae_inputs = tf.keras.Input(shape=(input_dim,))

# Get the outputs from the encoder network
z_mean, z_log_var, z = encoder(vae_inputs)

# Get the reconstructed outputs from the decoder network
decoder_outputs = decoder(z)

# Create the cVAE model
cvae = tf.keras.models.Model(vae_inputs, decoder_outputs)

# Compile the cVAE model
cvae.compile(optimizer='adam', loss='mse')

# Print the model summary
#cvae.summary()

# ---------------------Preprocess--------------------------



sequence_1mer_numpy = aptamers['sequence_1mer'].values
sequence_2mer_numpy = aptamers['sequence_2mer'].values
sequence_3mer_numpy = aptamers['sequence_3mer'].values
# sequence_embeddings_numpy = sequence_embeddings.values
target_type_numpy = target_type.values
kd_numpy = aptamers['kd'].values





sequence_1mer_numpy_reshaped = np.array([item.flatten() for item in sequence_1mer_numpy])
sequence_2mer_numpy_reshaped = np.array([item.flatten() for item in sequence_2mer_numpy])
sequence_3mer_numpy_reshaped = np.array([item.flatten() for item in sequence_3mer_numpy])
# sequence_embeddings_reshaped = np.array([item.flatten() for item in sequence_embeddings_numpy])
target_type_numpy_reshaped = np.array([item.flatten() for item in target_type_numpy])
kd_numpy_reshaped = np.array([item.flatten() for item in kd_numpy])

#Print the shpapes 

# print(padded_sequences.shape)
# print(sequence_1mer_numpy_reshaped.shape)
# print(sequence_2mer_numpy_reshaped.shape)
# print(sequence_3mer_numpy_reshaped.shape)
# print(target_type_numpy_reshaped.shape)
# print(kd_numpy_reshaped.shape)


# #Combine all my features into a single array 
#combined_data = np.concatenate([padded_sequences, sequence_1mer_numpy_reshaped, sequence_2mer_numpy_reshaped, sequence_3mer_numpy_reshaped, target_type_numpy_reshaped], axis=1)

#Testing combined_data
combined_data_1 = np.concatenate([padded_sequences, sequence_1mer_numpy_reshaped, sequence_2mer_numpy_reshaped, sequence_3mer_numpy_reshaped, target_type_numpy_reshaped, kd_numpy_reshaped], axis=1)


#Split the aptamers array into training and validation sets
train_data, val_data = train_test_split(combined_data_1, test_size=0.1, random_state=42)



# Define the number of epochs and batch size for training
epochs = 100
batch_size = 32

# Train the cVAE model
history = cvae.fit(train_data, 
                   train_data, 
                   batch_size=batch_size, epochs=epochs, 
                   validation_data=(val_data, val_data))

# Save the loss values for plotting
loss = history.history['loss']
val_loss = history.history['val_loss']


#-------------Print reconstruction Loss--------------------------------
import matplotlib.pyplot as plt

# Assuming you already have the 'history' variable containing loss values
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# # Plot the training and validation loss
# plt.plot(epochs, loss, 'b', label='Training Loss')
# plt.plot(epochs, val_loss, 'r', label='Validation Loss')
# plt.title('Reconstruction Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# # #----------------------------------------------------------------------

# Generate random samples from the latent space
num_samples = 10  # Specify the number of sequences to generate

#------------------Generate sequences for small molecules as a whole------------
latent_samples = np.random.normal(size=(num_samples, latent_dim))


# Use the encoder model to get the generated sequences
generated_sequences = decoder.predict(latent_samples)

# #------------------Generate sequences for steroid targets--------------------


#Choose the encoded representation for the 'Steroid' target type

# encoded_steroid = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# # Reshape encoded_steroid to have shape (1, )
# encoded_steroid = np.array(encoded_steroid).reshape(1, -1)


# # Compare each row of the target_type_encoded_series with encoded_steroid
# target_type_mask = (aptamers['target_type_encoded'] == encoded_steroid).all(axis=1)

# # Filter data using the mask
# filtered_data = combined_data[target_type_mask]

# # Get latent representations for the filtered data
# latent_representations = encoder.predict(filtered_data)[2]

# # Generate sequences based on latent representations
# generated_sequences = decoder.predict(latent_representations)








# Print the generated sequences
# print(generated_sequences)

# print(f"This is the generated sequence: {generated_sequences}")


# #--------------------Sample From latent space and decode samples-----------------------------------

# Define the reverse mapping from one-hot encoding to nucleotides
reverse_encoding = {tuple([1, 0, 0, 0]): 'A', tuple([0, 1, 0, 0]): 'C', tuple([0, 0, 1, 0]): 'G', tuple([0, 0, 0, 1]): 'T'}

# Assume the generated_sequences is the variable holding the generated sequences
# Threshold the values (using 0.5 as an example threshold)
thresholded_sequences = (generated_sequences > 0.5).astype(int)

# Convert the binary arrays to ssDNA sequences using the reverse mapping
decoded_sequences = []
for seq in thresholded_sequences:
    decoded_seq = ''.join([reverse_encoding[tuple(nucleotide)] for nucleotide in np.array_split(seq, len(seq) // 4) if tuple(nucleotide.tolist()) in reverse_encoding])
    decoded_sequences.append(decoded_seq)

#Print the decoded ssDNA sequences
print("\nGenerated Sequences: \n")
for seq in decoded_sequences:
    print(seq)

# #-------------------------Select ssDNA sequences for Testosertone-------------------------------

# # input_molecule = 'TAGGGAAGAGAAGGACATATGATTCCTGTCGAATTCAAATCGAACTAGCCTCATCTCAGCTCGTTGACTAGTACATGACCACTTGA'
# # encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
# # one_hot = np.array([encoding[nucleotide] for nucleotide in input_molecule])

# # print(one_hot.shape)
# # # Pad the one-hot encoded sequence to have the same length (assuming 0 as the padding value)
# # one_hot_padded = np.pad(one_hot, (0, input_dim - len(one_hot)), mode='constant', constant_values=0)


# # # Sample from the latent space to obtain a new representation (z_new)
# # z_new = np.random.normal(size=z_mean.shape)

# # # Decode the sampled representation (z_new) back to the ssDNA sequence
# # decoded_seq = decoder.predict(z_new)

# # # Convert the decoded sequence back to a readable ssDNA sequence
# # decoded_ssDNA = ''.join([reverse_encoding[tuple(nucleotide)] for nucleotide in np.array_split(decoded_seq[0], len(decoded_seq[0]) // 4)])

# # print("Generated ssDNA sequence:", decoded_ssDNA)




# # print("\nFeatures that will be used: \n")
# # print(aptamers.columns)


# Compute the reconstruction loss for the validation data
reconstruction_loss = cvae.evaluate(val_data, val_data)

# Print the average reconstruction loss
print("Average Reconstruction Loss on Validation Data:", reconstruction_loss)


print("Target categories in dataset:", target_categories)
print("Dataset shape: ", target_shape)
print(f"Features used: 1. sequence, 2. 1mer, 3. 2mer, 4. 3mer, 5. target_type, 6. kd")















