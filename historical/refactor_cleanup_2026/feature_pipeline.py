from __future__ import annotations



import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA, SparsePCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from . import structureMotif


pd.set_option("display.max_colwidth", None)


DATASET_FILENAME = "smallMolecule_aptamers_10172023.csv"
TARGET_FEATURES_FILENAME = "targets_feature_vector.csv"
FEATURES_OUTPUT_FILENAME = "features.npz"
PCA_MODEL_INFO_FILENAME = "pca_model_info.pkl"


def load_aptamer_dataset(dataset_filename: str) -> pd.DataFrame:
    """Load the raw aptamer dataset."""
    return pd.read_csv(dataset_filename)


def load_target_features(target_features_filename: str) -> pd.DataFrame:
    """Load the target feature dataset."""
    return pd.read_csv(target_features_filename)


def clean_aptamer_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw aptamer dataframe."""
    dataframe = df.copy()
    dataframe = dataframe[
        (dataframe["sequence"].str.len() != 0) & (~dataframe["sequence"].isna())
    ].reset_index(drop=True)

    dataframe["sequence"] = dataframe["sequence"].str.strip()
    dataframe["target"] = dataframe["target"].str.strip()
    return dataframe


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content percentage."""
    gc_count = sequence.upper().count("G") + sequence.upper().count("C")
    total_count = len(sequence)
    gc_content = (gc_count / total_count) * 100
    return round(gc_content, 2)


def add_gc_content(df: pd.DataFrame) -> pd.DataFrame:
    """Add GC content as a feature."""
    dataframe = df.copy()
    dataframe["gc_content"] = dataframe["sequence"].apply(calculate_gc_content)
    return dataframe


def one_hot_encoding(df: pd.DataFrame, column_name: str) -> None:
    """One-hot encode nucleotide sequences in place."""
    encoding = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }

    one_hot = df[column_name].apply(
        lambda seq: np.concatenate([encoding[nucleotide] for nucleotide in seq])
    )
    df[f"{column_name}_one_hot"] = one_hot


def calculate_1mer(df: pd.DataFrame, column_name: str) -> None:
    """Calculate normalized 1-mer frequencies in place."""

    def one_mer_freq(sequence: str) -> np.ndarray:
        nucleotides = ["A", "C", "G", "T"]
        freq_dict = {nt: 0 for nt in nucleotides}

        for nucleotide in sequence:
            if nucleotide in freq_dict:
                freq_dict[nucleotide] += 1

        sequence_length = len(sequence)
        freq_array = np.array(
            [freq_dict[nt] / sequence_length for nt in nucleotides],
            dtype=np.float64,
        )
        return freq_array / np.linalg.norm(freq_array)

    df[f"{column_name}_1mer"] = df[column_name].apply(one_mer_freq)


def calculate_2mer(df: pd.DataFrame, column_name: str) -> None:
    """Calculate normalized 2-mer frequencies in place."""

    def two_mer_freq(sequence: str) -> np.ndarray:
        nucleotides = ["A", "C", "G", "T"]
        freq_dict = {nt1 + nt2: 0 for nt1 in nucleotides for nt2 in nucleotides}

        for i in range(len(sequence) - 1):
            kmer = sequence[i : i + 2]
            if kmer in freq_dict:
                freq_dict[kmer] += 1

        sequence_length = len(sequence) - 1
        freq_array = np.array(
            [freq_dict[kmer] / sequence_length for kmer in freq_dict],
            dtype=np.float64,
        )
        return freq_array / np.linalg.norm(freq_array)

    df[f"{column_name}_2mer"] = df[column_name].apply(two_mer_freq)


def calculate_3mer(df: pd.DataFrame, column_name: str) -> None:
    """Calculate normalized 3-mer frequencies in place."""

    def three_mer_freq(sequence: str) -> np.ndarray:
        nucleotides = ["A", "C", "G", "T"]
        freq_dict = {
            nt1 + nt2 + nt3: 0
            for nt1 in nucleotides
            for nt2 in nucleotides
            for nt3 in nucleotides
        }

        for i in range(len(sequence) - 2):
            kmer = sequence[i : i + 3]
            if kmer in freq_dict:
                freq_dict[kmer] += 1

        sequence_length = len(sequence)
        freq_array = np.array(
            [freq_dict[kmer] / sequence_length for kmer in freq_dict],
            dtype=np.float64,
        )
        return freq_array / np.linalg.norm(freq_array)

    df[f"{column_name}_3mer"] = df[column_name].apply(three_mer_freq)


def calculate_word2vec_embeddings(
    df: pd.DataFrame,
    column_name: str,
    embedding_column_name: str | None = None,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    sg: int = 1,
) -> None:
    """Train Word2Vec on sequences and store embeddings in place."""
    if embedding_column_name is None:
        embedding_column_name = f"{column_name}_embeddings"

    sequences = df[column_name].apply(list).tolist()
    model = Word2Vec(
        sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
    )

    embeddings = df[column_name].apply(
        lambda seq: [model.wv[nucleotide] for nucleotide in seq]
    )
    df[embedding_column_name] = embeddings.apply(np.array).to_numpy()


def pad_to_max_length(arr: np.ndarray, max_length: int) -> np.ndarray:
    """Pad an array to a fixed maximum length."""
    if len(arr) < max_length:
        padding = np.zeros(max_length - len(arr))
        arr = np.concatenate((arr, padding))
    return arr


def calculate_dbn_one_hot(
    df: pd.DataFrame,
    column_name: str,
    encoding_column_name: str | None = None,
) -> None:
    """One-hot encode dot-bracket secondary structure notation."""
    if encoding_column_name is None:
        encoding_column_name = f"{column_name}_one_hot"

    def encode_dbn(dbn_string: str) -> np.ndarray:
        characters = {
            "(": [1, 0, 0],
            ")": [0, 1, 0],
            ".": [0, 0, 1],
        }
        return np.array([characters[char] for char in dbn_string])

    df[encoding_column_name] = df[column_name].apply(encode_dbn)


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sequence-derived features."""
    dataframe = df.copy()
    one_hot_encoding(dataframe, "sequence")
    calculate_1mer(dataframe, "sequence")
    calculate_2mer(dataframe, "sequence")
    calculate_3mer(dataframe, "sequence")
    calculate_word2vec_embeddings(dataframe, "sequence", vector_size=100)
    return dataframe


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add secondary-structure-derived features."""
    dataframe = df.copy()

    results = structureMotif.compute_mfe_structures(dataframe["sequence"])
    dataframe["structure"] = results["structure"]
    dataframe["mfe"] = results["gibbs_energy"]
    dataframe["matrix"] = results["matrix"]
    dataframe["stacking_energy"] = results["stacking_energy"]

    scaler = StandardScaler()
    dataframe["mfe"] = scaler.fit_transform(dataframe["mfe"].values.reshape(-1, 1))
    dataframe["stacking_energy"] = scaler.fit_transform(
        dataframe["stacking_energy"].values.reshape(-1, 1)
    )

    dataframe["matrix"] = dataframe["matrix"].apply(lambda arr: arr.flatten())
    max_length = max(len(arr) for arr in dataframe["matrix"])
    dataframe["matrix"] = dataframe["matrix"].apply(
        lambda arr: pad_to_max_length(arr, max_length).astype(np.int64)
    )

    calculate_dbn_one_hot(dataframe, "structure")
    dataframe["structure_one_hot"] = dataframe["structure_one_hot"].apply(
        lambda arr: arr.flatten()
    )
    max_length = max(len(arr) for arr in dataframe["structure_one_hot"])
    dataframe["structure_one_hot"] = dataframe["structure_one_hot"].apply(
        lambda arr: pad_to_max_length(arr, max_length).astype(np.int64)
    )

    return dataframe


def one_hot_encode_target_type(df: pd.DataFrame) -> np.ndarray:
    """Encode target type as one-hot vectors."""
    unique_target_types = df["type"].unique()
    vocabulary = {}

    for target_type in unique_target_types:
        binary_array = np.array(
            [int(target_type == category) for category in unique_target_types]
        )
        vocabulary[target_type] = binary_array

    df["target_type_encoded"] = df["type"].map(vocabulary)
    return np.array(df["target_type_encoded"].tolist())


def build_target_features(target_features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build target fingerprint and molecular property features."""
    dataframe = target_features_df.copy()

    columns_to_drop = ["Exact Mass", "Smiles", "Monoisotopic Mass", "Complexity"]
    dataframe = dataframe.drop(columns=columns_to_drop)

    imputer = SimpleImputer(strategy="mean")
    dataframe["xLogP3-AA"] = imputer.fit_transform(dataframe[["xLogP3-AA"]])

    scaler = StandardScaler()
    columns_to_standardize = [
        "Mol",
        "xLogP3-AA",
        "Hydrogen Bond Donor Count",
        "Hydrogen Bond Acceptor Count",
        "Rotatable Bond Count",
        "Topological Polar Surface Area",
        "Heavy Atom Count",
        "Formal Count",
        "Defined Atom Stereocenter Count",
        "Undefined Atom Stereocenter Count",
        "Defined Bond Stereocenter Count",
        "Undefined Bond Stereocenter Count",
        "Covalently-Bonded Unit Count",
    ]
    dataframe[columns_to_standardize] = scaler.fit_transform(
        dataframe[columns_to_standardize]
    )

    def hex_to_binary(hex_string: str, desired_length: int) -> str:
        binary_string = bin(int(hex_string, 16))[2:]
        return binary_string.zfill(desired_length)

    max_length = max(len(bin(int(x, 16))[2:]) for x in dataframe["Finger Print"])
    binary_fingerprints = [
        hex_to_binary(hex_string, max_length) for hex_string in dataframe["Finger Print"]
    ]
    fingerprint = np.array(
        [[int(bit) for bit in fingerprint_string] for fingerprint_string in binary_fingerprints]
    )

    def binary_str_to_array(binary_str: str) -> np.ndarray:
        return np.array([int(bit) for bit in binary_str])

    morgan_fingerprint = np.array(
        [binary_str_to_array(fingerprint_str) for fingerprint_str in dataframe["morgan fingerprint"]]
    )

    fingerprints = np.hstack([fingerprint, morgan_fingerprint])

    target_molecule_properties = np.hstack(
        [
            dataframe["Mol"].to_numpy().reshape(-1, 1),
            dataframe["xLogP3-AA"].to_numpy().reshape(-1, 1),
            dataframe["Hydrogen Bond Donor Count"].to_numpy().reshape(-1, 1),
            dataframe["Hydrogen Bond Acceptor Count"].to_numpy().reshape(-1, 1),
            dataframe["Rotatable Bond Count"].to_numpy().reshape(-1, 1),
            dataframe["Topological Polar Surface Area"].to_numpy().reshape(-1, 1),
            dataframe["Formal Count"].to_numpy().reshape(-1, 1),
            dataframe["Isotope Atom Count"].to_numpy().reshape(-1, 1),
            dataframe["Defined Atom Stereocenter Count"].to_numpy().reshape(-1, 1),
            dataframe["Undefined Atom Stereocenter Count"].to_numpy().reshape(-1, 1),
            dataframe["Defined Bond Stereocenter Count"].to_numpy().reshape(-1, 1),
            dataframe["Undefined Bond Stereocenter Count"].to_numpy().reshape(-1, 1),
            dataframe["Covalently-Bonded Unit Count"].to_numpy().reshape(-1, 1),
        ]
    )

    return fingerprints, target_molecule_properties


def reduce_with_pca(data: np.ndarray, variance_threshold: float = 0.95) -> np.ndarray:
    """Reduce data with PCA while preserving target variance."""
    pca = PCA().fit(data)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    components_to_keep = np.where(cumulative_variance > variance_threshold)[0][0] + 1

    pca = PCA(n_components=components_to_keep)
    pca.fit(data)
    reduced_data = pca.transform(data)

    reconstructed_data = pca.inverse_transform(reduced_data)
    mse = mean_squared_error(data, reconstructed_data)
    print(f"PCA reconstruction error: {mse}")

    return reduced_data


def reduce_with_sparse_pca(data: np.ndarray, save_model: bool = False) -> np.ndarray:
    """Reduce data with SparsePCA."""
    components = data.shape[0] // 6
    sparse_pca = SparsePCA(n_components=components, random_state=42)
    reduced_data = sparse_pca.fit_transform(data)

    reconstructed_data = sparse_pca.inverse_transform(reduced_data)
    mse = mean_squared_error(data, reconstructed_data)
    print(f"SparsePCA reconstruction error: {mse}")

    if save_model:
        model_info = {
            "pca_model": sparse_pca,
            "original_shape": data.shape,
        }
        joblib.dump(model_info, PCA_MODEL_INFO_FILENAME)

    return reduced_data


def save_features(
    sequences_reduced_data: np.ndarray,
    kd: np.ndarray,
    target_type_reduced_data: np.ndarray,
    aptamer_structures_reduced_data: np.ndarray,
    kmers_reduced_data: np.ndarray,
    sequence_embedding_reduced_data: np.ndarray,
    binding_energies: np.ndarray,
    fingerprints_reduced_data: np.ndarray,
    molecule_properties_reduced_data: np.ndarray,
) -> None:
    """Save all engineered features to NPZ."""
    np.savez(
        FEATURES_OUTPUT_FILENAME,
        sequences=sequences_reduced_data,
        kd=kd,
        target_type=target_type_reduced_data,
        structures=aptamer_structures_reduced_data,
        kmers=kmers_reduced_data,
        sequence_embedding=sequence_embedding_reduced_data,
        binding_energy=binding_energies,
        fingerprint=fingerprints_reduced_data,
        molecule_properties=molecule_properties_reduced_data,
    )


def main() -> None:
    """Run the full feature engineering pipeline."""
    df = load_aptamer_dataset(DATASET_FILENAME)
    df = clean_aptamer_dataframe(df)
    df = add_gc_content(df)
    df = add_sequence_features(df)
    df = add_structure_features(df)

    target_type = one_hot_encode_target_type(df)

    cols_to_drop = [
        "type",
        "target",
        "sequence",
        "cid",
        "cas",
        "reference",
        "length",
        "structure",
    ]
    df.drop(columns=cols_to_drop, axis=1, inplace=True)

    scaler = StandardScaler()
    df["kd"] = scaler.fit_transform(df["kd"].values.reshape(-1, 1))

    max_length = max(len(arr) for arr in df["sequence_one_hot"])
    df["sequence_one_hot"] = df["sequence_one_hot"].apply(
        lambda arr: pad_to_max_length(arr, max_length).astype(int)
    )

    df["sequence_embeddings"] = df["sequence_embeddings"].apply(lambda arr: arr.flatten())
    max_length = max(len(arr) for arr in df["sequence_embeddings"])
    df["sequence_embeddings"] = df["sequence_embeddings"].apply(
        lambda arr: pad_to_max_length(arr, max_length)
    )

    sequences = np.vstack(df["sequence_one_hot"].values)
    kd = df["kd"].values.reshape(-1, 1)
    aptamer_structures = np.hstack(
        [np.vstack(df["structure_one_hot"].values), np.vstack(df["matrix"].values)]
    )
    kmers = np.hstack(
        [
            np.vstack(df["sequence_1mer"].values),
            np.vstack(df["sequence_2mer"].values),
            np.vstack(df["sequence_3mer"].values),
        ]
    )
    sequence_embedding = np.vstack(df["sequence_embeddings"].values)
    binding_energies = np.hstack(
        [df["mfe"].values.reshape(-1, 1), df["stacking_energy"].values.reshape(-1, 1)]
    )

    target_features_df = load_target_features(TARGET_FEATURES_FILENAME)
    fingerprints, target_molecule_properties = build_target_features(target_features_df)

    molecule_properties_reduced_data = reduce_with_pca(target_molecule_properties)
    sequence_embedding_reduced_data = reduce_with_pca(sequence_embedding)
    kmers_reduced_data = reduce_with_pca(kmers)

    sequences_reduced_data = reduce_with_sparse_pca(sequences, save_model=True)
    target_type_reduced_data = reduce_with_sparse_pca(target_type)
    aptamer_structures_reduced_data = reduce_with_sparse_pca(aptamer_structures)
    fingerprints_reduced_data = reduce_with_sparse_pca(fingerprints)

    save_features(
        sequences_reduced_data,
        kd,
        target_type_reduced_data,
        aptamer_structures_reduced_data,
        kmers_reduced_data,
        sequence_embedding_reduced_data,
        binding_energies,
        fingerprints_reduced_data,
        molecule_properties_reduced_data,
    )


if __name__ == "__main__":
    main()
