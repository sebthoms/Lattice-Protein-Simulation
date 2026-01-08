import numpy as np
import matplotlib.pyplot as plt
import tomllib

K_B = 1.0 # 1.38064852e-23

NUM_AMINO_ACIDS = 20
EMPTY_FIELD = -1

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    

class Configuration:

    def __init__(self, config_file):
        self.config_file = config_file
        
        with open(config_file, 'rb') as f:
            config = tomllib.load(f)

        # j values
        self.j_values_file = config['j_values']['j_values_file']
        self.j_values = self.load_j_values()

        # protein
        protein = config['protein']

        self.creation_criteria = protein['initialize_by']

        self.protein_sequence = None
        self.protein_length = None

        if self.creation_criteria == 'sequence_file':

            self.sequence_file = protein['sequence_file']
            self.protein_sequence = self.load_sequence('sequence_file')

            self.protein_length = len(self.protein_sequence)

        elif self.creation_criteria == 'sequence':

            self.sequence_str = protein['sequence']
            self.protein_sequence = self.load_sequence('sequence')

            self.protein_length = len(self.protein_sequence)

        elif self.creation_criteria == 'length':

            self.protein_length = protein['length']

            self.protein_sequence = np.random.randint(0, NUM_AMINO_ACIDS, self.protein_length)

        else:
            
            raise ValueError('protein.initialize_by must be "length", "sequence" or "sequence_file"')

        # grid
        self.bbuffer = config['grid']['buffer']


    def load_j_values(self):

        j_array = np.zeros((NUM_AMINO_ACIDS, NUM_AMINO_ACIDS))

        file = open(self.j_values_file)
        for r, row in enumerate(file):
            row = row.rstrip('\t').rsplit()
            for c, column in enumerate(row):
                j_array[r, c] = float(column)
        file.close()

        return j_array

    def load_sequence(self, kind): 

        aa_to_int = {
        'A': 0,  'C': 1,  'D': 2,  'E': 3,  'F': 4,
        'G': 5,  'H': 6,  'I': 7,  'K': 8,  'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }

        if kind == 'sequence_file':

            file = open(self.sequence_file)
            line = file.readline()
            sequence = np.array([aa_to_int[aa] for aa in line.strip()])
            file.close()

        elif kind == 'sequence':
            
            sequence = np.array([aa_to_int[aa] for aa in self.sequence_str])
    
        return sequence


class Protein:

    def __init__(self, config):

        self.sequence = config.protein_sequence
        self.length = config.protein_length

        self.coords = np.zeros((self.length, 2), dtype=int)
        self.coords[:] = np.array([config.bbuffer, config.bbuffer + self.length // 2])
        self.coords[:, 0] += np.arange(self.length, dtype=int)


class StandardModel:

    def __init__(self, config, temperature, protein):

        self.temperature = temperature
        self.grid_size = config.protein_length + 2 * config.bbuffer
        self.j_values = config.j_values
        self.protein = protein

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.grid.fill(EMPTY_FIELD)

        self.load_to_grid()

        self.energy = self.get_energy()
    
    def load_to_grid(self):

        self.grid.fill(EMPTY_FIELD) # to remove old chain if called later
        for i in range(self.protein.length):
            self.grid[self.protein.coords[i, 0], self.protein.coords[i, 1]] = i

    def in_bounds(self, x, y):

        return (0 <= x < self.grid_size) and (0 <= y < self.grid_size)
    
    def get_neighbours(self, x, y):

        x_0, y_0 = x, y
        neighbours = []

        for dx, dy in DIRECTIONS:

            x, y = x_0 + dx, y_0 + dy

            if self.in_bounds(x, y):
                neighbour = self.grid[x, y]
                if neighbour != EMPTY_FIELD:
                    neighbours.append(neighbour)

        return neighbours
    
    def get_neighbours_acid_nr(self, acid_nr):

        acid_coords = self.protein.coords[acid_nr]
        return self.get_neighbours(acid_coords[0], acid_coords[1])

    def get_noncovalent_neighbours_acid_nr(self, acid_nr):

        neighbours = self.get_neighbours_acid_nr(acid_nr)

        noncovalent_neighbours = []

        left_acid = acid_nr - 1
        right_acid = acid_nr + 1 
        
        for n in neighbours:
            if n not in (left_acid, right_acid):
                noncovalent_neighbours.append(n)

        return noncovalent_neighbours
    
    def get_length(self):

        x_diff = self.protein.coords[-1, 0]-self.protein.coords[0, 0]
        y_diff = self.protein.coords[-1, 1]-self.protein.coords[0, 1]

        return np.sqrt(x_diff**2 + y_diff**2)

    def get_energy(self):

        energy = 0

        for acid_nr, acid in enumerate(self.protein.sequence):
            acid_covalent_neighbours = self.get_noncovalent_neighbours_acid_nr(acid_nr)
            for i in acid_covalent_neighbours:
                energy = energy + self.j_values[acid, self.protein.sequence[i]]
        
        return energy / 2 # since every pair is counted twice

    def new_acid_pos_allowed(self, acid_nr, x_new, y_new):

        if not self.in_bounds(x_new, y_new):
            return False

        if self.grid[x_new, y_new] != EMPTY_FIELD:
            return False

        required = {
            n for n in (acid_nr - 1, acid_nr + 1)
            if 0 <= n < self.protein.length
        }

        neighbours = set(self.get_neighbours(x_new, y_new))

        return required <= neighbours

    def next_state(self):
        """
        Perform a single Metropolis Monte Carlo move.
        Returns the updated energy.
        """

        # pick random residue
        acid_nr = np.random.randint(self.protein.length)
        x_old, y_old = self.protein.coords[acid_nr]

        # propose diagonal move
        dx, dy = DIAGONAL_DIRECTIONS[np.random.randint(4)]
        x_new = x_old + dx
        y_new = y_old + dy

        # check move validity
        if not self.new_acid_pos_allowed(acid_nr, x_new, y_new):
            return self.energy

        
        E_before = self.energy

        # apply move (temporarily) # on grid and protein!
        self.grid[x_old, y_old] = EMPTY_FIELD
        self.grid[x_new, y_new] = acid_nr
        self.protein.coords[acid_nr] = [x_new, y_new]

        E_after = self.get_energy()

        delta_energy = E_after - E_before

        # always accept energy-decreasing moves
        if delta_energy <= 0:
            self.energy = E_after
            return self.energy

        # accept energy-increasing moves with Metropolis probability 
        boltzmann = np.exp(-delta_energy / (K_B * self.temperature))
        if np.random.random_sample() < boltzmann:
            self.energy = E_after
            return self.energy

        # reject move: revert # on grid and protein!
        self.grid[x_new, y_new] = EMPTY_FIELD
        self.grid[x_old, y_old] = acid_nr
        self.protein.coords[acid_nr] = [x_old, y_old]

        self.energy = E_before
        return self.energy


