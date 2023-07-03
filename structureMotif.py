
import re
import subprocess
import os
import nupack


def read_sequences(filename):
    with open(filename, 'r') as f:
        sequences = f.readlines()
    sequences = [seq.strip() for seq in sequences]
    return sequences


def compute_mfe_structures(sequences):
    #Store structure and mfe information
    struct = []
    gibbs = []
    # Define the sequence
    for sequence in sequences:
    
        # Define the strand and set the nucleic acid type (use 'dna' for DNA, 'rna' for RNA)
        strand = nupack.Strand(sequence, 'my_strand', material='DNA')

        # Create a complex with the strand
        complex = nupack.Complex([strand])

        # Set the calculation options, such as temperature and model
        options = nupack.Model(material='DNA', celsius=25)

        # Compute the MFE structure
        mfe_result = nupack.mfe(complex, model=options)

        # Print the MFE structure and its energy
        structure = str(mfe_result[0].structure)
        energy = str(mfe_result[0].energy)
        struct.append(structure)
        gibbs.append(energy)
    
    
    return struct, gibbs


def format_sequences_for_forester(sequences, struct):
    input_data = ""
    
    for i, (sequence, structure) in enumerate(zip(sequences, struct), 1):
        input_data += f">sequence{i}\n"
        input_data += f"{sequence}\n"
        input_data += f"{structure}\n"
    
    input_data += "&\n"
    forester_output_file = "forester_dataset.fasta"
    with open(forester_output_file, 'w') as f:
        f.write(input_data)

    return input_data, forester_output_file


def run_rnaforester(input_data, file):
    command = f"RNAforester -m -l -f={file}"
    
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    output, _ = process.communicate(input=input_data.encode())
    
    output = output.decode()
    
    return output


def extract_forester_results(output):
    start_marker = "*** Results ***"
    end_marker = "\n\n\n"
    start_index = output.find(start_marker)
    end_index = output.find(end_marker, start_index) + len(end_marker)
    forester_results_section = output[start_index:end_index]
    pattern = r"sequence(\d+)\s+([\w-]+)"
    matches = re.findall(pattern, forester_results_section)
    sorted_matches = sorted(matches, key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
    member_sequences = [match[1].replace('U', 'T') for match in sorted_matches]
    
    return member_sequences



def find_sequences_in_larger_strings(member_sequences, seqs, length_of_left_flank):
    '''Locate member sequences in larger sequence and return the indices'''
    found_sequences = []
    indices = []
    
    for index, (sequence, larger_string) in enumerate(zip(member_sequences, seqs)):
        pattern = sequence.replace("-", ".?")
        matches = re.finditer(pattern, larger_string)
        
        for match in matches:
            #This is using 0-based indexing 
            start_index = match.start()
            end_index = match.end() - 1
            
            #Now we need to add 1 to convert to 1-based indexing
            #name_of_sequence = f"Sequence{index + 1}"
            #found_sequences.append((name_of_sequence, sequence, start_index + length_of_left_flank, end_index + length_of_left_flank))
            indices.append((start_index + length_of_left_flank, end_index + length_of_left_flank))
            
            #print(f"Member Sequence '{sequence}' found at indices {start_index} to {end_index} in '{larger_string}'.")
    
    # if not found_sequences:
    #     print("No sequences found.")
    
    return indices


def write_sequences_to_file(filename, sequences):
    with open(filename, 'w') as f:
        for sequence in sequences:
            f.write(sequence + '\n')

def generate_Varna_object(sequences, struct, indices):
     # Combine data for generating structure images
    combined_data = []
    for sequence, structure, index in zip(sequences, struct, indices):
        item = {
            'sequence': sequence,
            'structure': structure,
            'index1': index[0],
            'index2': index[1]
        }
        combined_data.append(item)
    return combined_data
    

def generate_structure_images(var_object, output_directory, resolution):
    for i, item in enumerate(var_object):
        sequence = item['sequence']
        structure = item['structure']
        index1 = item['index1']
        index2 = item['index2']
        print(index1, index2)
        output_file = os.path.join(output_directory, f"result{i + 1}.PNG")
        highlight_region = f'"{index1}-{index2}:fill=#bcffdd"'
        
        command = f'java -cp VARNAv3-93.jar fr.orsay.lri.varna.applications.VARNAcmd -sequenceDBN "{sequence}" -structureDBN "{structure}" -o "{output_file}" -algorithm naview -resolution "{resolution}" -highlightRegion {highlight_region} -spaceBetweenBases "1.4"'
        
        process = subprocess.Popen(command, shell=True)
        process.communicate()
        
        # if process.returncode == 0:
        #     print(f"Command executed successfully for {output_file}.")
        # else:
        #     print(f"Command execution failed for {output_file}.")


def main():
    # User input for filenames and other parameters
    
    #sequences_filename = input("Please enter the filename for target sequences: ")
    
    # Read sequences from the input file
    core_sequences = read_sequences("core.txt")
    
    # Compute MFE structures
    struct, gibbs = compute_mfe_structures(core_sequences)
    
    # Format sequences for RNAforester input
    forester_input, forester_file = format_sequences_for_forester(core_sequences, struct)
    
    # Run RNAforester and capture the output
    forester_output = run_rnaforester(forester_input, forester_file)
    
    # Extract relevant sections from the RNAforester output
    memeber_sequences = extract_forester_results(forester_output)    
    
    # Extract member sequences from the forester_results_section
    indices = find_sequences_in_larger_strings(memeber_sequences, core_sequences, 18)
    
    #Create object for varna #Use full sequences now instead of just core sequences
    full_sequences = read_sequences("targetSequences.txt")
    full_struct, gibbs = compute_mfe_structures(full_sequences)

    varna_object = generate_Varna_object(full_sequences, full_struct, indices)
    
    # Generate structure images
    output_directory = "./output_structures/"  # Update with the desired output directory
    resolution = "4.0"  # Update with the desired resolution
    generate_structure_images(varna_object, output_directory, resolution)


if __name__ == '__main__':
    main()
