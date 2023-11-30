
#print("Welcome to fasta file creator. \n"
#"To read your text file correctly make sure that each line contains a unique sequence.\n"  
#"I will give you a fasta file that will be neatly organized as follow: \n\n > annotation of seq1 \n actgagaga"
#"\n > annotation of seq2 \n catgcagta ")

#print("\n*To correctly find your file please enter the correct path to your file or place this program in the same directory as your desired input file. \n\n")
#name_of_file = input()

def convert_file(input_filename, output_filename, target_names=None):
    '''Take a text file and convert it to fasta format.'''
    counter = 0
    with open(input_filename, "r") as input_file:
        with open(output_filename, "w") as output_file:
            for line in input_file:
                counter += 1
                if target_names is not None and counter <= len(target_names):
                    output_file.write(f">{target_names[counter-1]}\n{line}")
                else:
                    output_file.write(f">sequence{counter}\n{line}")



def write_sequences_to_file(sequences, output_filename):
    '''Take a list of sequences and convert to text file.'''
    with open(output_filename, "w") as output_file:
        for sequence in sequences:
            output_file.write(sequence + "\n")


if __name__ == '__main__':
    # If the script is executed directly, perform the conversion
    input_filename = "core.txt"
    output_filename = "data.fasta"
    convert_file(input_filename, output_filename)






