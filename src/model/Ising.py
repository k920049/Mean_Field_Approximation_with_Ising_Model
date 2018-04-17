import tensorflow as tf
import numpy as np

from PIL import Image

methods = ['Gibbs', 'MeanfieldH']

class Ising:

    def __init__(self, image, sigma = 2, size = 28, J):

        self.X = image
        self.Y = None
        self.sigma = sigma
        self.size = size
        self.J
        self.count = 0

    def run_iteration(self):
        self.count = self.count + 1
        self.image_preprocess()
        title = "Noisy image after " + str(self.count) + " iteration"
        self.show_image(self.Y, title)

        for m in range(len(methods)):
            method = method[m]

    def gauss_log_prob(self, mu, sigma, )

    def mean_field_ising_grid(self, CPDs, image, maxiter=100, inplace_update=True, update_rate=1):
        off_state = 0
        on_state = 1
        logodds = logprobFn(CPDs[0], image) - logprobFn(CPDs[1], image)
        logodds = np.reshape(a=logodds, self.size, self.size)




    def show_image(self, image, title):
        if self.count % 32 == 0:
            image = np.reshape(a=image, newshape=(self.size, self.size, 1))
            image = Image.fromarray(image)
            image.show(title)


    def image_preprocess(self, image):
        self.Y = self.X
        mu = np.mean(a=self.Y)

        for i in range(self.size):
            for j in range(self.size):
                if self.Y[i][j] >= mu:
                    self.Y[i][j] = 1
                else:
                    self.Y[i][j] = -1

        self.Y = self.Y + np.random.randn(d0=self.size, d1=self.size)
