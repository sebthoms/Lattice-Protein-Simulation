from backend import Configuration, Protein, StandardModel

import numpy as np
import matplotlib.pyplot as plt

def fixed_plot(config, temperature, iterations, energy_plot = True, length_plot = True):

    protein = Protein(config)
    model = StandardModel(config, temperature, protein)

    iterations = int(iterations) + 1 # so that iteration 0 becomes the starting state
    energies = np.zeros(iterations)
    lengths = np.zeros(iterations)
    iterations_indeces = np.arange(iterations)

    energies[0], lengths[0] = model.get_energy(), model.get_length()
    for i in range(1, iterations):
        energies[i] = model.next_state()
        lengths[i] = model.get_length()

    if energy_plot:
        plt.plot(iterations_indeces, energies)

        plt.grid(True)
        plt.title(f'Length = {protein.length}, iterations = {iterations-1}, temperature = {temperature}')
        plt.xlabel('Iterations')
        plt.ylabel('Energy')

        plt.show()
    if length_plot:
        plt.plot(iterations_indeces, lengths)

        plt.grid(True)
        plt.title(f'Length = {protein.length}, iterations = {iterations-1}, temperature = {temperature}')
        plt.xlabel('Iterations')
        plt.ylabel('Length')

        plt.show()


def var_plot(config, start_temp, end_temp, temp_step_nr, iterations, energy_plot = True, length_plot = True):

    protein = Protein(config)
    model = StandardModel(config, start_temp, protein)

    iterations = int(iterations) + 1 # so that iteration 0 becomes the starting state
    energies = np.zeros(iterations)
    lengths = np.zeros(iterations)
    iterations_indeces = np.arange(iterations)

    temps = np.linspace(start_temp, end_temp, temp_step_nr)
    temps_iterator = np.repeat(temps, iterations // temp_step_nr)
    temps_iterator = np.append(temps_iterator, np.full(iterations % temp_step_nr, end_temp))

    energies[0], lengths[0] = model.get_energy(), model.get_length()
    for i in range(1, iterations):
        model.temperature = temps_iterator[i]
        energies[i] = model.next_state()
        lengths[i] = model.get_length()

    if energy_plot:
        plt.plot(iterations_indeces, energies)
        plt.plot(iterations_indeces, temps_iterator)
        
        plt.grid(True)
        plt.title(f'Length = {protein.length}, iterations = {iterations-1}')
        plt.xlabel('Iterations')
        plt.ylabel('Energy / Temperature')
        plt.legend(['energy', 'temperature'])        
        
        plt.show()

    if length_plot:
        plt.plot(iterations_indeces, lengths)
        plt.plot(iterations_indeces, temps_iterator)

        plt.grid(True)
        plt.title(f'Length = {protein.length}, iterations = {iterations-1}')
        plt.xlabel('Iterations')
        plt.ylabel('Length / Temperature')
        plt.legend(['length', 'temperature'])        

        plt.show()


def var_plot_avg(config, start_temp, end_temp, temp_step_nr, iterations_per_temp, energy_plot = True, length_plot = True):

    protein = Protein(config)
    model = StandardModel(config, start_temp, protein)
    
    energies = np.zeros(temp_step_nr)
    lengths = np.zeros(temp_step_nr)

    temps = np.linspace(start_temp, end_temp, temp_step_nr)

    for i in range(temp_step_nr):

        model.temperature = temps[i]
        energies_for_this_temp = np.zeros(iterations_per_temp)
        lengths_for_this_temp = np.zeros(iterations_per_temp)

        print(f'temp step {i+1}/{temp_step_nr}')

        for j in range(iterations_per_temp):
            
            energies_for_this_temp[j] = model.next_state()
            lengths_for_this_temp[j] = model.get_length()

        energies[i] = np.average(energies_for_this_temp)
        lengths[i] = np.average(lengths_for_this_temp)

    if energy_plot:
        plt.scatter(temps, energies)

        plt.grid(True)
        plt.title(f'Length = {protein.length}, temperature steps = {temp_step_nr}, iterations per temperature step = {iterations_per_temp}')
        plt.xlabel('Iterations')
        plt.ylabel('Energy')

        plt.show()

    if length_plot:
        plt.scatter(temps, lengths)

        plt.grid(True)
        plt.title(f'Length = {protein.length}, temperature steps = {temp_step_nr}, iterations per temperature step = {iterations_per_temp}')
        plt.xlabel('Iterations')
        plt.ylabel('Length')

        plt.show()

