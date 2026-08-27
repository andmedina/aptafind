import random
'''Generate new sequence based on an existing sequence'''

sequence = 'ACTGCTGACGTACGTACGTA'


def mutate(index, sequence: str):
    '''Given the index of the sequence and the sequence, change the char found at the index into altered nucleotide '''
    sequence = list(sequence)
    library = ['A', 'T', 'C', 'G']
    
    if sequence[index] == library[index % 4]:
        sequence[index] = library[(index + 1) % 4]
        
    else:
        val = (index + 1) % 4    
        sequence[index] = library[val]

    return sequence

def insertion(index, sequence):
    '''Given the index and the sequence, insert a random nucleotide before or after the index.'''
    sequence = list(sequence)
    library = ['A', 'T', 'C', 'G']

    random_value = index % 2

    if random_value == 1:
        #Add to previous 
        sequence[index - 1] = library[(index + 1) % 4]

    else:
        #Add to next 
        val = (index + 1) % 4    
        sequence[index + 1] = library[val]
    return sequence
def deletion(index, sequence):
    '''Given a sequence and an index, delete the character found at that index'''

    sequence = list(sequence)
    
    sequence[index] = ''
    sequence = str(sequence)
    sequence = ''.join(sequence)                                 
    return sequence






print(list(sequence))

print("Mutate:", mutate(2, sequence))
print("Instertion: ", insertion(2, sequence))
print("Deletion", deletion(2, sequence))


