import matplotlib.pyplot as plt
from generate_data import x_to_im

def display_network(weights, size, dim):
    fig, axes = plt.subplots(size/5+1, 5, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})

    for x, ax in zip(weights, axes.flat):
        ax.imshow(x_to_im(x, dim), interpolation='none', cmap='Greys')

    fig.canvas.set_window_title('Hidden units visualization')
    plt.show()
