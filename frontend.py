from backend import Configuration, Protein, StandardModel

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

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

def fixed_animation(config, temperature, iterations, stride=1, interval=100):
    protein = Protein(config)
    model = StandardModel(config, temperature, protein)

    iterations = int(iterations)

    # Store grid snapshots
    grids = []
    grids.append(model.grid.copy())

    for i in range(1, iterations + 1):
        model.next_state()
        if i % stride == 0:
            grids.append(model.grid.copy())

    grids = np.array(grids)

    # ---- Colormap ----
    # color 0 is for empty (-1), others for amino acids
    colors = ["white"] + [plt.cm.hsv((0.85 - 0.05 )* i / protein.length) for i in range(protein.length)]
    cmap = ListedColormap(colors)

    bounds = np.arange(-1.5, protein.length + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        grids[0].T,
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    ax.set_title(
        f"Protein folding | Length={protein.length}, T={temperature}"
    )
    ax.set_xticks([])
    ax.set_yticks([])

    step_text = ax.text(
    0.02, 0.95, "",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top"
)

    def update(frame):
        im.set_data(grids[frame].T)
        step_text.set_text(f"Step {frame * stride}")
        return [im, step_text]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(grids),
        interval=interval,
        blit=True
    )

    plt.show()
    return ani
