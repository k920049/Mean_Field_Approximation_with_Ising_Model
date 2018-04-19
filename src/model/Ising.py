import tensorflow as tf
import numpy as np

from PIL import Image

methods = ['Gibbs', 'MeanfieldH']


class Ising:

    def __init__(self, image, size=28):

        self.X = image
        self.Y = None
        self.size_X = image.shape[0]
        self.size_Y = image.shape[1]
        self.count = 0
        self.sigma = 2
        self.direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.disp_flag = False

    def run_iteration(self):

        print("Running")

        self.Y = self.image_preprocess(self.X)
        mu = self.mean_field_ising_grid(image=self.Y, J=0.2)
        title = "Full stop at iteration " + str(100)
        self.show_image(mu * 255, title, 0)

    def set_disp_flag(self, flag):
        self.disp_flag = flag

    def gauss_log_prob(self, mu, sigma, image):

        if np.isscalar(mu):
            image = np.reshape(a=image, newshape=(1, image.size))

        image = image - mu

        if not np.isscalar(num=sigma) and (sigma.size > 1):
            sigma2 = np.repeat(a=np.transpose(sigma), repeats=image.shape[0], axis=0)
            lhs = np.multiply(x1=image, x2=image)
            lhs = np.divide(x1=lhs, x2=2 * sigma2)
            lhs = -lhs
            rhs = np.log(2 * np.pi * sigma2)
            rhs = 0.5 * rhs
            tmp = lhs - rhs
            logp = np.sum(tmp, axis=1)
        else:
            # R = np.linalg.cholesky(a=sigma)
            # logp = np.divide(x1=image, x2=R)
            # logp = np.multiply(x1=logp, x2=logp)
            R = np.sqrt(sigma)
            logp = image / R
            logp = np.multiply(logp, logp)
            logp = np.sum(a=logp, axis=0)
            logp = -0.5 * logp
            logz = 0.5 * np.log(2 * np.pi)
            logz = logz + R
            logp = logp - logz
        return logp

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def within_range(self, d, ref):
        if d[0] < 0:
            return False
        elif d[0] >= ref.shape[0]:
            return False
        elif d[1] < 0:
            return False
        elif d[1] >= ref.shape[1]:
            return False
        else:
            return True

    def mean_field_ising_grid(self, image, J, maxiter=16, inplace_update=True, rate=0.1):
        logodds = self.gauss_log_prob(mu=1, sigma=self.sigma, image=image) - self.gauss_log_prob(mu=-1, sigma=self.sigma, image=image)
        logodds = np.reshape(a=logodds, newshape=(self.size_X, self.size_Y))

        p1 = self.sigmoid(logodds)
        mu = 2 * p1 - 1
        mu = np.reshape(a=mu, newshape=(self.size_X, self.size_Y))

        for iter in range(maxiter):
            title = "Iteration at " + str(iter)
            mean = np.mean(mu)
            print("Mean " + str(mean))
            self.show_image(image=mu * 255, title=title, iter=iter)
            muNew = mu
            for ix in range(mu.shape[0]):
                for iy in range(mu.shape[1]):
                    S_bar = 0
                    pos = [ix, iy]
                    k = [0, 0]
                    for d in self.direction:
                        k[0] = d[0] + pos[0]
                        k[1] = d[1] + pos[1]
                        if not self.within_range(k, mu):
                            continue
                        S_bar = S_bar + J * mu[k[0], k[1]]
                    if not inplace_update:
                        muNew[pos[0], pos[1]] = (1 - rate) * muNew[pos[0], pos[1]] + rate * np.tanh((S_bar + 0.5 * logodds[pos[0], pos[1]]))
                    else:
                        mu[pos[0], pos[1]] = (1 - rate) * mu[pos[0], pos[1]] + rate * np.tanh((S_bar + 0.5 * logodds[pos[0], pos[1]]))

            if not inplace_update:
                mu = muNew
        return mu

    def show_image(self, image, title, iter):
        if self.disp_flag and iter % 4 == 0:
            image = np.reshape(a=image, newshape=(self.size_X, self.size_Y))
            image = Image.fromarray(obj=image)
            image.show(title)

    def image_preprocess(self, image):
        mu = np.mean(a=image)
        print("mu" + str(mu))

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] >= mu:
                    image[i, j] = -1
                else:
                    image[i, j] = 1

        image = image + self.sigma * np.random.randn(image.shape[0], image.shape[1])
        return image
