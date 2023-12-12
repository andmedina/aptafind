import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np
from keras.losses import mean_squared_error
from keras import backend as K

# Load the data
data = np.load('features.npz')

# Concatenating different features
ssdna_features = np.hstack([data['sequences'], data['kd'], data['target_type'], data['structures'], data['kmers'], data['sequence_embedding'], data['binding_energy']])
molecule_features = np.hstack([data['fingerprint'], data['molecule_properties']])

print(f" sequences shape {data['sequences'].shape} ")

# Shuffle the data
ssdna_indices = np.arange(len(ssdna_features))
np.random.shuffle(ssdna_indices)
ssdna_features = ssdna_features[ssdna_indices]
molecule_features = molecule_features[ssdna_indices]

# Split the data into training, testing, and validation
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

total_samples = len(ssdna_features)
train_samples = int(train_ratio * total_samples)
test_samples = int(test_ratio * total_samples)
val_samples = total_samples - train_samples - test_samples

ssdna_train = ssdna_features[:train_samples]
ssdna_test = ssdna_features[train_samples:train_samples + test_samples]
ssdna_val = ssdna_features[train_samples + test_samples:]

molecule_train = molecule_features[:train_samples]
molecule_test = molecule_features[train_samples:train_samples + test_samples]
molecule_val = molecule_features[train_samples + test_samples:]

# Sampling function for the VAE
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Custom VAE loss function
def vae_loss(x, x_decoded_mean, z_mean, z_log_var, original_dim):
    reconstruction_loss = mean_squared_error(x, x_decoded_mean) * original_dim
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)

# cVAE Encoder
def build_cvae_encoder(ssdna_input_shape, molecule_input_shape, latent_dim):
    ssdna_inputs = keras.Input(shape=ssdna_input_shape, name='ssdna_input')
    molecule_inputs = keras.Input(shape=molecule_input_shape, name='molecule_input')

    ssdna_x = layers.Dense(128, activation='relu')(ssdna_inputs)
    ssdna_x = layers.Dense(64, activation='relu')(ssdna_x)

    molecule_x = layers.Dense(128, activation='relu')(molecule_inputs)
    molecule_x = layers.Dense(64, activation='relu')(molecule_x)

    x = layers.concatenate([ssdna_x, molecule_x])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    return models.Model([ssdna_inputs, molecule_inputs], [z_mean, z_log_var], name='cvae_encoder')

# cVAE Decoder
def build_cvae_decoder(latent_dim, output_shape, molecule_input_shape):
    latent_inputs = keras.Input(shape=(latent_dim,), name='latent_input')
    molecule_inputs = keras.Input(shape=molecule_input_shape, name='molecule_input')

    x = layers.concatenate([latent_inputs, molecule_inputs])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)

    outputs = layers.Dense(output_shape, activation='sigmoid', name='output_layer')(x)
    return models.Model([latent_inputs, molecule_inputs], outputs, name='cvae_decoder')

# cVAE Model Class
class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.original_dim = ssdna_features.shape[1]

    def call(self, inputs, training=False):
        ssdna_data, molecule_data = inputs
        z_mean, z_log_var = self.encoder([ssdna_data, molecule_data])
        z = sampling([z_mean, z_log_var])
        reconstructed = self.decoder([z, molecule_data])
        # Only use the loss function during training
        if training:
            loss = vae_loss(ssdna_data, reconstructed, z_mean, z_log_var, self.original_dim)
            self.add_loss(loss)
        return reconstructed

# Model parameters
ssdna_input_shape = ssdna_features.shape[1:]
molecule_input_shape = molecule_features.shape[1:]
latent_dim = 16
output_shape = ssdna_features.shape[1]

# Build and compile the cVAE model
encoder = build_cvae_encoder(ssdna_input_shape, molecule_input_shape, latent_dim)
decoder = build_cvae_decoder(latent_dim, output_shape, molecule_input_shape)
cvae = CVAE(encoder, decoder)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
cvae.compile(optimizer=optimizer)

# Training
epochs = 50
batch_size = 6

history = cvae.fit(
    [ssdna_train, molecule_train],
    ssdna_train,  # Target data for reconstruction loss
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([ssdna_val, molecule_val], ssdna_val),
)




#----------Sampling
num_samples = 1
latent_samples = np.random.randn(num_samples, latent_dim)
molecule_input = molecule_features[28]
molecule_input = molecule_input.reshape(1, -1)
generated_samples = decoder.predict([latent_samples, molecule_input])


#print(generated_samples.shape)
#generated_samples = generated_samples.flatten()

#print(generated_samples.shape)
# Display the model summary
# cvae.summary()


#Decode Output

# Extract the shapes of the original components
sequences_shape = data['sequences'].shape
kd_shape = data['kd'].shape
target_type_shape = data['target_type'].shape
structures_shape = data['structures'].shape
kmers_shape = data['kmers'].shape
sequence_embedding_shape = data['sequence_embedding'].shape
binding_energy_shape = data['binding_energy'].shape

# Split the generated_samples back into its components using the extracted shapes
decoded_sequences = generated_samples[:, :sequences_shape[1]]
decoded_kd = generated_samples[:, sequences_shape[1]:sequences_shape[1] + kd_shape[1]]
decoded_target_type = generated_samples[:, sequences_shape[1] + kd_shape[1]:sequences_shape[1] + kd_shape[1] + target_type_shape[1]]
decoded_structures = generated_samples[:, sequences_shape[1] + kd_shape[1] + target_type_shape[1]:sequences_shape[1] + kd_shape[1] + target_type_shape[1] + structures_shape[1]]
decoded_kmers = generated_samples[:, sequences_shape[1] + kd_shape[1] + target_type_shape[1] + structures_shape[1]:sequences_shape[1] + kd_shape[1] + target_type_shape[1] + structures_shape[1] + kmers_shape[1]]
decoded_sequence_embedding = generated_samples[:, sequences_shape[1] + kd_shape[1] + target_type_shape[1] + structures_shape[1] + kmers_shape[1]:sequences_shape[1] + kd_shape[1] + target_type_shape[1] + structures_shape[1] + kmers_shape[1] + sequence_embedding_shape[1]]
decoded_binding_energy = generated_samples[:, sequences_shape[1] + kd_shape[1] + target_type_shape[1] + structures_shape[1] + kmers_shape[1] + sequence_embedding_shape[1]:]



import joblib
model_info_filename = "pca_model_info.pkl"
loaded_model_info = joblib.load(model_info_filename)

# Extract the PCA model and original shape from the loaded dictionary
sparse_pca = loaded_model_info['pca_model']
original_shape = loaded_model_info['original_shape']

# Use the PCA model to perform inverse transformation
original_data = sparse_pca.inverse_transform(decoded_sequences)
original_data = original_data.flatten()

def calculate_mean_threshold(binary_values):
    mean = sum(binary_values) / len(binary_values)
    return mean

mean_threshold = calculate_mean_threshold(original_data)
print("Mean Threshold:", mean_threshold)

def threshold_to_binary(values, threshold):
    binary_sequence = [1 if value >= threshold else 0 for value in values]
    return binary_sequence

threshold = .3
binary_result = threshold_to_binary(original_data, threshold)

print(binary_result)

def binary_to_nucleotides(binary_values, encoding):
    nucleotides = []
    binary_group = []

    for binary_value in binary_values:
        binary_group.append(binary_value)
        if len(binary_group) == 4:
            for nucleotide, encoding_value in encoding.items():
                if binary_group == encoding_value:
                    nucleotides.append(nucleotide)
                    break
            binary_group = []

    return "".join(nucleotides)




# Given encoding scheme
encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}

nucleotide_result = binary_to_nucleotides(binary_result, encoding)
print(nucleotide_result)
