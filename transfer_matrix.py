import math
import numpy as np
import sys
import copy
import sympy
import matplotlib.pyplot as plt
from dispersion import Dispersion

t = sympy.symbols("t")

class Waveguide:
    def __init__(self, curv, tors, rp, d, H, L, density=7800, E=209 * 10 ** 9, nu=0.28):
        self.curv = curv # curvature with respect to the parameter
        self.tors = tors # torsion with respect to the parameter
        self.rp = rp # |r'(t)|, where t is the parameter of the curve
        self.density = density # density
        self.E = E # Young's modulus
        self.nu = nu # poisson's ratio
        self.H = H # range of the parameter of the curve
        self.L = float(L) # the length of the curve
        self.m = density * math.pi * d ** 2 / 4  # mass per length
        timoshenko_coeff = 6 * (1 + nu) / (7 + 6 * nu) # Timoshenko coefficient
        G = E / (2 * (1 + nu))  # Shear modulus
        self.sigma_1 = timoshenko_coeff * G * math.pi * d ** 2 / 4  # shear rigidity of the wire
        self.sigma_2 = timoshenko_coeff * G * math.pi * d ** 2 / 4
        self.sigma_p = E * math.pi * d ** 2 / 4  # extensional rigidity of the wire
        self.I_1 = 1 / 64 * math.pi * d ** 4  # second moment of area
        self.I_2 = 1 / 64 * math.pi * d ** 4
        self.I_t = 1 / 32 * math.pi * d ** 4  # second moment of area(torsion)
        self.beta_1 = E * self.I_1
        self.beta_2 = E * self.I_2
        self.beta_t = G * self.I_t
        self.k_1 = self.k_2 = 1 / 2 * (d / 2)
        self.k_t = self.k_1 * math.sqrt(2)

    def c(self, v):
        tau0 = float(self.tors(v))
        kappa0 = float(self.curv(v))
        sigma_1, sigma_2, sigma_p = self.sigma_1, self.sigma_2, self.sigma_p
        beta_1, beta_2, beta_t = self.beta_1, self.beta_2, self.beta_t
        m = self.m
        k_1, k_2, k_t = self.k_1, self.k_2, self.k_t
        mat = [
            [0, tau0, -kappa0, 1, 0, 0, 1 / sigma_1, 0, 0, 0, 0, 0],  # 1
            [-tau0, 0, 0, 0, -1, 0, 0, 1 / sigma_2, 0, 0, 0, 0],  # 2
            [kappa0, 0, 0, 0, 0, 0, 0, 0, 1 / sigma_p, 0, 0, 0],  # 3
            [0, 0, 0, 0, -tau0, 0, 0, 0, 0, 1 / beta_1, 0, 0],  # 4
            [0, 0, 0, tau0, 0, -kappa0, 0, 0, 0, 0, 1 / beta_2, 0],  # 5
            [0, 0, 0, 0, kappa0, 0, 0, 0, 0, 0, 0, 1 / beta_t],  # 6
            [np.poly1d([-m, 0, 0]), 0, 0, 0, 0, 0, 0, tau0, -kappa0, 0, 0, 0],  # 7
            [0, np.poly1d([-m, 0, 0]), 0, 0, 0, 0, -tau0, 0, 0, 0, 0, 0],  # 8
            [0, 0, np.poly1d([-m, 0, 0]), 0, 0, 0, kappa0, 0, 0, 0, 0, 0],  # 9
            [0, 0, 0, np.poly1d([-m * k_1 ** 2, 0, 0]), 0, 0, -1, 0, 0, 0, -tau0, 0],  # 10
            [0, 0, 0, 0, np.poly1d([-m * k_2 ** 2, 0, 0]), 0, 0, 1, 0, tau0, 0, -kappa0],  # 11
            [0, 0, 0, 0, 0, np.poly1d([-m * k_t ** 2, 0, 0]), 0, 0, 0, 0, kappa0, 0]  # 12
        ]

        for i in range(12):
            for j in range(12):
                if type(mat[i][j]) is int:
                    mat[i][j] = float(mat[i][j])
        return np.array(mat, dtype=object)

    def get_transfer_matrix(self, iterations):
        dv = self.H / iterations

        def rk4(z):
            Nv = iterations
            dv = self.H / Nv
            rp = self.rp
            for i in range(Nv):
                v = dv * i
                k1 = np.dot(self.c(v), z) * float(rp(v))
                k2 = np.dot(self.c(v + dv / 2), z + 1 / 2 * k1 * dv) *float(rp(v+dv/2))
                k3 = np.dot(self.c(v + dv / 2), z + 1 / 2 * k2 * dv) * float(rp(v+dv/2))
                k4 = np.dot(self.c(v + dv), z + k3 * dv) * float(rp(v + dv))
                z = z + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dv
                printProgress(i, Nv, 'Runge-Kutta Iteration:', 'Complete', 1, 50)
            return z

        z = np.identity(12)
        return TransferMatrix(rk4(z), self.L)


class TransferMatrix:
    def __init__(self, mat, length=0):
        self.mat = mat
        self.L = length

    def __str__(self):
        return str(self.mat)

    def resultFunc(self, omega):
        matrix = self.mat
        n = len(matrix)
        m = len(matrix[0])
        mat = [[0] * m for i in range(n)]
        for i in range(n):
            for j in range(m):
                if type(matrix[i][j]) is np.poly1d:
                    mat[i][j] = np.polyval(matrix[i][j], omega)
                else:
                    mat[i][j] = matrix[i][j]
        return np.array(mat)

    def natFreq(self, upper, lower=0, prec=0.01):
        assert lower <= upper
        upper *= 2 * math.pi
        result = copy.deepcopy(self.mat)
        result = np.delete(result, (5, 4, 3, 2, 1, 0), axis=0)
        result = np.delete(result, (5, 4, 3, 2, 1, 0), axis=1)

        # result = np.delete(result, (11, 10, 9, 8, 7, 6), axis=1)

        result = TransferMatrix(result)
        x = np.arange(lower, upper, prec)
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = np.linalg.det(result.resultFunc(x[i]))
            printProgress(i, len(x), "Calculation:", "Complete", 3, 50)

        L = []
        for i in range(len(y) - 1):
            if y[i] * y[i + 1] < 0:
                print(x[i] / 2 / math.pi)
                L.append(x[i] / 2 / math.pi)
        return L

    def dispersion(self, limit, prec = 0.01):
        freq = np.array([])
        real = np.array([])
        imaginary = np.array([])
        limit *= 2*math.pi
        freqs_input = np.arange(0, limit, prec*2*math.pi)
        cnt = 0
        for i in freqs_input:
            l, v = np.linalg.eig(self.resultFunc(i))
            for ev in l:
                b = -math.log(abs(ev)) / self.L
                a = np.angle(ev) / self.L  # actually, we have to calculate for (angles+2npi) for all integer n
                freq = np.append(freq, i/math.pi / 2)
                real = np.append(real, a)
                imaginary = np.append(imaginary, b)
            printProgress(cnt, len(freqs_input), 'Plotting:', 'Complete', 3, 50)
            cnt += 1
        return Dispersion(freq, real, imaginary)

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()