import matplotlib.pyplot as plt
from generate_data import x_to_im
from math import sqrt, ceil

def display_network(weights, size, dim):
    s = sqrt(size)
    fig, axes = plt.subplots(int(ceil(s)), int(size/s), figsize=(12, 6), \
        subplot_kw={'xticks': [], 'yticks': []})

    for x, ax in zip(weights, axes.flat):
        ax.imshow(x_to_im(x, dim), interpolation='none', cmap='Greys')

    fig.canvas.set_window_title('Hidden units visualization')
    plt.show()
