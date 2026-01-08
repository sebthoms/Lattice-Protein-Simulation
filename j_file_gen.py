import numpy as np

NUM_AMINO_ACIDS = 20
FILENAME = 'j_values.tsv'

j = -4 + np.random.rand(NUM_AMINO_ACIDS,NUM_AMINO_ACIDS)*2
print(j)

file = open(FILENAME, 'w')

LAST_LINE = NUM_AMINO_ACIDS - 1

for i_1 in range(NUM_AMINO_ACIDS):

    for i_2 in range(NUM_AMINO_ACIDS):
        file.write(f'{j[i_1, i_2]}\t')
        
    if i_1 != LAST_LINE:
        file.write('\n')

file.close()