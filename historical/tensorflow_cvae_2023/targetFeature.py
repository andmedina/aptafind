
# # import pandas as pd
# # from rdkit import Chem
# # from rdkit.Chem import AllChem
# # from pubchempy import get_compounds


# # aptamers = pd.read_csv('smallMolecule_aptamers_10172023.csv')

# # print(f"Dataset shape: {aptamers.shape}")


# # target_cid = aptamers['cid']   #Get all target names


# # found_compounds = []  # Initialize an empty list to store found compounds
    
# # for id in target_cid:
# #     compounds = get_compounds(id, 'cid')  # Try to retrieve the compounds from PubChem    
# #     if compounds:
# #         for compound in compounds:
# #             mol = Chem.MolFromSmiles(compound.isomeric_smiles)  # Convert SMILES to RDKit Mol
# #             morgan = None
# #             if mol:  # Check if the Mol object is valid
# #                 morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()  # Generate fingerprint

# #             found_compounds.append({'Smiles': compound.isomeric_smiles, 
# #                                     'Mol': compound.molecular_weight, 
# #                                     'Finger Print': compound.fingerprint,
# #                                     'morgan fingerprint' : morgan, 
# #                                     'xLogP3-AA': compound.xlogp,
# #                                     'Hydrogen Bond Donor Count' : compound.h_bond_donor_count,
# #                                     'Hydrogen Bond Acceptor Count' : compound.h_bond_acceptor_count,
# #                                     'Rotatable Bond Count' : compound.rotatable_bond_count,
# #                                     'Exact Mass' : compound.exact_mass,
# #                                     'Monoisotopic Mass' : compound.monoisotopic_mass,
# #                                     'Topological Polar Surface Area' : compound.tpsa,
# #                                     'Heavy Atom Count' : compound.heavy_atom_count,
# #                                     'Formal Count' : compound.charge,
# #                                     'Complexity' : compound.complexity,
# #                                     'Isotope Atom Count' : compound.isotope_atom_count,
# #                                     'Defined Atom Stereocenter Count' : compound.defined_atom_stereo_count,
# #                                     'Undefined Atom Stereocenter Count' : compound.undefined_atom_stereo_count,
# #                                     'Defined Bond Stereocenter Count' : compound.defined_bond_stereo_count,
# #                                     'Undefined Bond Stereocenter Count' : compound.undefined_bond_stereo_count,
# #                                     'Covalently-Bonded Unit Count' : compound.covalent_unit_count})  # If the compound was found, append its information to found_compounds list
# #     else:
# #         print(f' Compound "{id}" not found in pubchem')
        
# # found_compounds_df = pd.DataFrame(found_compounds)
# # found_compounds_df.to_csv('found_compounds_feature_vector.csv', index=False)

# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from pubchempy import get_compounds

# def get_compound_features(compound):
#     mol = Chem.MolFromSmiles(compound.isomeric_smiles)
#     morgan = None
#     if mol:
#         morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()

#     return {
#         'Smiles': compound.isomeric_smiles,
#         'Mol': compound.molecular_weight,
#         'Finger Print': compound.fingerprint,
#         'morgan fingerprint': morgan,
#         'xLogP3-AA': compound.xlogp,
#         'Hydrogen Bond Donor Count': compound.h_bond_donor_count,
#         'Hydrogen Bond Acceptor Count': compound.h_bond_acceptor_count,
#         'Rotatable Bond Count': compound.rotatable_bond_count,
#         'Exact Mass': compound.exact_mass,
#         'Monoisotopic Mass': compound.monoisotopic_mass,
#         'Topological Polar Surface Area': compound.tpsa,
#         'Heavy Atom Count': compound.heavy_atom_count,
#         'Formal Count': compound.charge,
#         'Complexity': compound.complexity,
#         'Isotope Atom Count': compound.isotope_atom_count,
#         'Defined Atom Stereocenter Count': compound.defined_atom_stereo_count,
#         'Undefined Atom Stereocenter Count': compound.undefined_atom_stereo_count,
#         'Defined Bond Stereocenter Count': compound.defined_bond_stereo_count,
#         'Undefined Bond Stereocenter Count': compound.undefined_bond_stereo_count,
#         'Covalently-Bonded Unit Count': compound.covalent_unit_count
#     }

# def create_target_features():
#     try:
#         aptamers = pd.read_csv('smallMolecule_aptamers_10172023.csv')
#     except FileNotFoundError:
#         print("CSV file not found.")
#         return

#     print(f"Dataset shape: {aptamers.shape}")

#     target_cid = aptamers['cid']
#     found_compounds = []

#     for id in target_cid:
#         try:
#             compounds = get_compounds(id, 'cid')
#             if compounds:
#                 found_compounds.extend([get_compound_features(compound) for compound in compounds])
#             else:
#                 print(f'Compound "{id}" not found in PubChem.')
#         except Exception as e:
#             print(f"An error occurred while processing ID {id}: {e}")

#     found_compounds_df = pd.DataFrame(found_compounds)
#     return found_compounds_df

# if __name__ == "__main__":
#     results_df = create_target_features()
#     #result_df.to_csv('found_compounds_feature_vector.csv', index=False)

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pubchempy import get_compounds
from tqdm import tqdm

def get_compound_features(compound):
    mol = Chem.MolFromSmiles(compound.isomeric_smiles)
    morgan = None
    if mol:
        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()
    return {
        'Smiles': compound.isomeric_smiles,
        'Mol': compound.molecular_weight,
        'Finger Print': compound.fingerprint,
        'morgan fingerprint': morgan,
        'xLogP3-AA': compound.xlogp,
        'Hydrogen Bond Donor Count': compound.h_bond_donor_count,
        'Hydrogen Bond Acceptor Count': compound.h_bond_acceptor_count,
        'Rotatable Bond Count': compound.rotatable_bond_count,
        'Exact Mass': compound.exact_mass,
        'Monoisotopic Mass': compound.monoisotopic_mass,
        'Topological Polar Surface Area': compound.tpsa,
        'Heavy Atom Count': compound.heavy_atom_count,
        'Formal Count': compound.charge,
        'Complexity': compound.complexity,
        'Isotope Atom Count': compound.isotope_atom_count,
        'Defined Atom Stereocenter Count': compound.defined_atom_stereo_count,
        'Undefined Atom Stereocenter Count': compound.undefined_atom_stereo_count,
        'Defined Bond Stereocenter Count': compound.defined_bond_stereo_count,
        'Undefined Bond Stereocenter Count': compound.undefined_bond_stereo_count,
        'Covalently-Bonded Unit Count': compound.covalent_unit_count
    }

def create_target_features():
    try:
        aptamers = pd.read_csv('smallMolecule_aptamers_10172023.csv')
    except FileNotFoundError:
        print("CSV file not found.")
        return

    print(f"Dataset shape: {aptamers.shape}")

    target_cid = aptamers['cid']
    found_compounds = []

    for id in tqdm(target_cid, desc='Processing compounds'):
        try:
            compounds = get_compounds(id, 'cid')
            if compounds:
                found_compounds.extend([get_compound_features(compound) for compound in compounds])
            else:
                print(f'Compound "{id}" not found in PubChem.')
        except Exception as e:
            print(f"An error occurred while processing ID {id}: {e}")

    found_compounds_df = pd.DataFrame(found_compounds)
    return found_compounds_df

if __name__ == "__main__":
    results_df = create_target_features()
    results_df.to_csv('targets_feature_vector.csv', index=False)