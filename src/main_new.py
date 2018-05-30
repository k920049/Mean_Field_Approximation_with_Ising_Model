import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import skimage.io

from model.IsingGridVaryingField import IsingGridVaryingField

def IsingDeNoise(noisy, q, burnin = 50000, loops = 500000):
    h = 0.5 * np.log(q / (1 - q))
    gg = IsingGridVaryingField(noisy.shape[0], noisy.shape[1], h * noisy, 2)
    gg.grid = np.array(noisy)

    # Burn-in
    for _ in range(burnin):
        gg.gibbs_move()

    # Sample
    avg = np.zeros_like(noisy).astype(np.float64)
    for _ in range(loops):
        gg.gibbs_move()
        avg += gg.grid
    return avg / loops

def make_comparison_plot(W, H, field, invtemp, samples=1000000):
    # Generate all possible combinations and compute probability.
    gg = IsingGrid(W, H, field, invtemp)
    N = gg.width * gg.height
    prob = []
    for n in range(2**N):
        gg.from_number(n)
        prob.append( gg.probability() )
    total = sum(prob)
    prob = [ (n, x/total) for n, x in enumerate(prob) ]
    prob.sort(key = lambda pair: pair[1])

    # Randomly sample a large number of states
    for _ in range(10000):
        gg.gibbs_move()

    count = [0]*(2**N)
    for _ in range(samples):
        gg.gibbs_move()
        count[ gg.to_number() ] += 1

    total = sum(count)
    count = [ x/total for x in count ]

    # Plot
    fig, axes = plt.subplots(figsize=(12,6))
    axes.set_xlim([0,2**N])
    _ = axes.plot( range(2**N), [x for _, x in prob], lw=2 )
    _ = axes.scatter( range(2**N), [count[n] for n, _ in prob] )

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def file_write(image, file_name):
    f = open(file=file_name, mode="w+")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            f.write(str(image[i, j]) + " ")
        f.write("\n")
    f.close()

def binary(image):
    mu = np.mean(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= mu:
                image[i, j] = 1
            else:
                image[i, j] = -1
    return image

def Main():
    image = skimage.io.imread("../reference/image_small.png")
    #image = skimage.io.imread("../resource/image.jpg")
    #image = rgb2gray(image)
    image = image[:, :].astype(np.int)
    image = binary(image)
    #print(image.shape)
    #file_write(image=image, file_name="../Original.data")
    #image = rgb2gray(image)
    #image = (image[:,:].astype(np.int) * 2) - 1 # Black and white so just grab one RGB channel
    fig, axes = plt.subplots(figsize=(10,6))
    axes.imshow(image, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)
    plt.show()

    q = 0.9
    noise = np.random.random(size = image.size).reshape(image.shape) > q
    noisy = np.array(image)
    noisy[noise] = -noisy[noise]
    fig, axes = plt.subplots(figsize=(10,6))
    axes.imshow(noisy, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)
    plt.show()

    avg = IsingDeNoise(noisy, 0.9)
    avg[avg >= 0] = 1
    avg[avg < 0] = -1
    avg = avg.astype(np.int)

    fig, axes = plt.subplots(ncols=2, figsize=(11,6))
    axes[0].imshow(avg, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)
    axes[1].imshow(noisy, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)
    plt.show()

if __name__ == "__main__":
    Main()
