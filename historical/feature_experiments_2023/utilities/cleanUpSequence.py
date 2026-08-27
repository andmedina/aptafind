'''
This script will take as input a raw sequence in this form: 

5'dCpdCpdCpdTpdApdGpdTpdTpdApdGpdCpdCpdApdTpdCpdTpdCpdCpdCp3' 

It will return a processed sequence in this form:

CCCTAGTTAGCCATCTCCC 19

Where the number is the length of the sequence.
'''

raw_sequence = input("Please enter your raw sequecnce: ")
valid_sequence = {'A', 'T', 'C', 'G'}
split_sequence = raw_sequence.split('d')
new_sequence = [char[0] for char in split_sequence if char[0] in valid_sequence]
new_sequence = ''.join(new_sequence)
print("This is the proccessed sequence:\n", new_sequence, len(new_sequence))