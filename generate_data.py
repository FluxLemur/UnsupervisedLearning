import numpy as np
import matplotlib.pyplot as plt

dim = 8
np.random.seed(1) # use a seed when comparing different results

def rand_rect():
    (x1, y1) = np.random.random_integers(0, dim-2, 2)
    x2 = np.random.randint(x1+1, dim)
    y2 = np.random.randint(y1+1, dim)
    return (x1, y1, x2, y2)

def area(x1, y1, x2, y2):
    return (x2-x1) * (y2-y1)

def rand_rect_image(empty=True, minarea=5):
    (x1,y1,x2,y2) = rand_rect()
    while area(x1,y1,x2,y2) < minarea:
        (x1,y1,x2,y2) = rand_rect()
    image = np.zeros((dim, dim))

    if empty:
        for i in xrange(x1, x2+1):
            image[i][y1] = 1
            image[i][y2] = 1
        for j in xrange(y1, y2):
            image[x1][j] = 1
            image[x2][j] = 1
    else:
        for i in xrange(x1, x2+1):
            for j in xrange(y1, y2+1):
                image[i][j] = 1
    return image

def gen_x():
    x = rand_rect_image()
    x.resize((dim*dim,))
    return x

def gen_data(size):
    data = np.zeros((size, dim*dim))
    for i in xrange(size):
        data[i] = gen_x()
    return data

def x_to_im(x, d):
    return np.resize(x, (d, d))

def show_x(x):
    if x.shape == (dim*dim,) or x.shape == (1, dim*dim) or x.shape == (dim*dim, 1):
        show = x_to_im(x, dim)
    else:
        show = x
    plt.imshow(show, interpolation='none', cmap='Greys')
    plt.show()

def show_data(data):
    size = len(data)
    fig, axes = plt.subplots(size/5, 5, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
    for x, ax in zip(data, axes.flat):
        ax.imshow(x_to_im(x, dim), interpolation='none', cmap='Greys')
    plt.show()
