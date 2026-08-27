'''This program will extract the core region from the sequence when provided with the left and right flanking regions'''

'''This program will extract the core region from the sequence when provided with the left and right flanking regions'''

'''Additional Notes: This program is using 0-based indexing because it is only interested in extracting the middle region
and by using 0-based indexing, an input like 1-18 will actually count from 0 to 18 and include an additional character. 
This is a desirable feature because 1-18 means we want to start on the 19th character which counting from 0 to 18 achieves.'''
#filename = input("Please enter the filename that contains your sequences: ")
filename = "damian_sequences_full.txt"
with open(filename, 'r') as f:
    sequences = f.readlines()

# Remove leading/trailing whitespace and newline characters from sequences
sequences = [seq.strip() for seq in sequences]

#option = input("Enter '1' to provide flanking regions, or '2' to provide flanking indices: ")
option = '2'

if option == '1':
    #left_flank = input("Please enter the left flanking region: ")
    left_flank = "GGAGGCTCTCGGGACGAC"
    #right_flank = input("Please enter the right flanking region: ")
    right_flank = "GTCGTCCCGCCTTTAGGATTTACAG"
elif option == '2':
    #left_indices_range = input("Please enter the range of indices for the left flanking region (e.g., '1-18' or '1-20'): ")
    #right_indices_range = input("Please enter the range of indices for the right flanking region (e.g., '20-40' or '21-60'): ")
    left_indices_range = '1-18'
    right_indices_range = ' 49-73'

    lstart, lend = map(int, left_indices_range.split('-'))
    rstart, rend = map(int, right_indices_range.split('-'))
    print(len(sequences[0]))
    if lstart < 0 or lend >= len(sequences[0]) or rstart >= rend or lend > len(sequences[0]):
        raise ValueError("Invalid indices please check your input.")    
else:
    raise ValueError("Invalid option. Please enter '1' or '2'.")

core_sequences = []

for sequence in sequences:
    if option == '1':    

        left_start = sequence.find(left_flank)
        left_end = left_start + len(left_flank)

        right_start = sequence.find(right_flank)
        right_end = right_start + len(right_flank)

        captured_content = sequence[left_end:right_start]
        core_sequences.append(captured_content)


    elif option == '2':
        captured_content = sequence[lend:rstart - 1]
        core_sequences.append(captured_content)


print(core_sequences)
#output_filename = input("Please enter the output filename: ")
output_filename =  "damian_core_sequences.txt"

with open(output_filename, 'w') as f:
    for core_sequence in core_sequences:
        f.write(core_sequence + '\n')

print(f"Core sequences extracted and written to {output_filename}")






