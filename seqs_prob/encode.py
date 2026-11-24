import numpy as np

#transfer letters to numbers
#"ACGT" = [0, 1, 2, 3]
def encode(seqs, alphabet):
    encoded_sequences = []
    for sequence in seqs:
        encoded_sequence = []
        for letter in sequence:
            encoded_sequence.append(alphabet.index(letter))
        encoded_sequences.append(encoded_sequence)
    return np.array(encoded_sequences)