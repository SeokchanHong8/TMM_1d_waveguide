import numpy
import sympy
import matplotlib.pyplot as plt
import math
from transfer_matrix import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad

def norm(x):
    return sympy.sqrt(x.dot(x))

def xrange(low,high, N):
    L = []
    a = low
    dv = (high-low)/N
    while a < high:
        L.append(a)
        a += dv
    return L

t = sympy.symbols("t")

#parametric equation definition
'''
#HYPERHELIX
R1 = 50 * 10 ** -3
R2 = 12 * 10 ** -3
omega1 = 105
omega2 = 442
x = sympy.cos(omega1 * t) * (R1 - R2 * sympy.cos(omega2 * t))+ R2 * sympy.sin(omega1 * t) * sympy.sin(omega2 * t) / (1 + R1 ** 2 * omega1 ** 2) ** 0.5
y = sympy.sin(omega1 * t) * (R1 - R2 * sympy.cos(omega2 * t))- R2 * sympy.cos(omega1 * t) * sympy.sin(omega2 * t) / (1 + R1 ** 2 * omega1 ** 2) ** 0.5
z = t + R1 * R2 * omega1 * sympy.sin(omega2 * t)/(1+R1 ** 2 * omega1 ** 2) ** 0.5
'''

#HELIX
R = 65 * 10 ** -3 # Radius of Helix
omega = 2 * math.pi * 6 / 0.32
d = 12 * 10 ** -3  # diameter of the beam
x = R * sympy.cos(omega * t)
y = R * sympy.sin(omega * t)
z = t

#Range of Parameter in parametric equation
#parameter ranging from vrange1 =< t <= vrange2
vrange1 = 0
vrange2 = 0.32

#plotting above defined parametric curve
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')


#If you want to increase the resolution of the figure, increase the Res
#Res: number of points where parametric equation is calculated
res = 1000
vs = xrange(vrange1,vrange2,res)
xs = []
ys = []
zs = []

for i in range(len(vs)):
    xs.append(x.subs(t, vs[i]))
    ys.append(y.subs(t, vs[i]))
    zs.append(z.subs(t, vs[i]))

ax.plot(xs,ys,zs)
plt.show()

#calculating curvature, torsion, norm(r'), and length of the parametric curve
H = sympy.Matrix([x,y,z])
H_p = H.diff(t)
H_pp = H_p.diff(t)
H_ppp = H_pp.diff(t)

H_p_H_pp_cross = H_p.cross(H_pp)
H_pp_H_ppp_cross = H_pp.cross(H_ppp)

curvature = (norm(H_p_H_pp_cross)/norm(H_p) ** 3)
torsion = (abs(H_p.dot(H_pp_H_ppp_cross))/norm(H_p_H_pp_cross) ** 2)
rp = (norm(H_p))

curvature_func = sympy.lambdify(t,curvature)
torsion_func = sympy.lambdify(t,torsion)
rp_func = sympy.lambdify(t,rp)
L = quad(rp_func,vrange1,vrange2)[0] #Length of the curve

#Creating Waveguide object
waveguide_1d = Waveguide(curvature_func, torsion_func, rp_func, d, vrange2, L)

#Calculating Transfer Matrix
#If you want to increase the division of waveguide while calculating transfer matrix, increase div
div = 1000
TM = waveguide_1d.get_transfer_matrix(div)

#Calculating natural frequncy below f_lim_nat(Hz)
f_lim_nat = 100
print(TM.natFreq(f_lim_nat))

#Calculating and plotting dispersion relation below f_lim_disp(Hz) with precision prec
f_lim_disp = 200
prec = 0.1
disp = TM.dispersion(f_lim_disp, prec)
disp.plot(xlim=f_lim_disp)

