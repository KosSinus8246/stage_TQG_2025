import numpy as np
import matplotlib as mpl
#import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 15
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

print('TQG_SOLVE_1_BIS')
print('-----------------------------------------------------')



# cf TQG notes : A.X = c.B.X
# @uthor : dimitri moreau 23/04/2025



##################################
# VARIABLES, ESPACE ...
##################################



Ny, Nk = 60, 60
Ly = np.pi
dy = Ly/Ny
y_l, k = np.linspace(0.1,Ly,Ny), np.linspace(0.1,0.1+Nk*0.1,Nk)


beta = 0 #1e-11
k, Rd = np.linspace(0.1,0.1+0.1*Nk,Nk),1 # à modifier
K2 = (k**2 + 1/(Rd**2))*dy**2
#K2 = 0
U0= 1


phi, theta, Un = np.zeros((Ny, Nk)), np.zeros((Ny, Nk)), U0*np.exp(-y_l**2)
phi_r,theta_r = phi.reshape(Nk*Ny), theta.reshape(Nk*Ny)

# X = np.array([phi_r,theta_r])

# V/G/Mn
Theta0 = 1
F1 = 0
Vn = Un * dy**2
G12 = -2*y_l*Theta0*np.exp(-y_l**2)
G11 = 2.0*Un*(1-2*y_l**2) + F1*Un+beta - G12
F11 = G11*dy**2



print('/////////////////////////////////////////////////////')
print('PARAMS : OK')



##################################
# CONSTRUCTION DE LA MATRICE B
##################################



# Diagonale principale
main_diag = -(2 + K2) * np.ones(Ny)
off_diag = np.ones(Ny-1)
# Construct tridiagonal B11 using np.diag
B11 = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
# Construct other blocks
B12 = np.zeros((Ny, Nk))
B21 = np.zeros((Ny, Nk))
B22 = np.eye(Ny, Nk)
# Combine them into full block matrix B
top = np.concatenate((B11, B12), axis=1)
bottom = np.concatenate((B21, B22), axis=1)
B = np.concatenate((top, bottom), axis=0)
print('MATRIX B : OK')



##################################
# CONSTRUCTION DE LA MATRICE A
##################################



# Block A11
main_diag_A11 = (-Un*(2 + K2) + F11)  # shape (3,)
A11 = np.zeros((Ny,Nk))
np.fill_diagonal(A11, main_diag_A11)

for i in range(Ny - 1):
    A11[i,i+1] = Un[i]
    A11[i+1,i] = Un[i]

# Block A12
A12 = np.diag(-Vn)
# Block A21
A21 = np.diag(G12)
# Block A22
A22 = np.diag(Un)

# Final block matrix A
top_A = np.concatenate((A11, A12), axis=1)
bottom_A = np.concatenate((A21, A22), axis=1)
A = np.concatenate((top_A, bottom_A), axis=0)
print('MATRIX A : OK')



##################################
# SOLUTION
##################################
# A.X = c.B.X   =>  B^(-1).A.X = c.X 



# c, X = npl.eig(npl.inv(B) @ A)
# Ou bien utiliser scipy si c'est trop lourd à inverser
c, X = spl.eig(A,B)

sigma1 = k*np.imag(c[:Ny])
sigma2 = k*np.imag(c[Ny:])



print('COMPUTATION : OK')
print('/////////////////////////////////////////////////////')



##################################
# PLOT
##################################



print('PLOT...')



plt.figure(figsize=(8, 6))
plt.scatter(k, sigma1, color='b', label=r'$\sigma_1$')
plt.scatter(k, sigma2, color='r', label=r'$\sigma_2$')
plt.axhline(0, color='k', linestyle='--', label='Real axis')
plt.axvline(0, color='k', linestyle='--', label='Imaginary axis')
plt.title('Eigenvalue and wavenumber')
plt.xlabel(r'$k$')
plt.ylabel(r'$\sigma = \mathbf{Im}\{c\}.k$')
plt.legend()





print('-----------------------------------------------------')


#####################
# stabiilité

borne = (1/4)*(Un*y_l)**2
test_crit = Un * G12

print('Stability analysis : ')
if (test_crit < borne).all() == True:
	print('Totally unstable')
else:
	print('Stable ?')



plt.figure()
plt.plot(y_l,borne,'--',label=r'Borne : $\frac{1}{4}.(\overline{U}.y)^2$')
plt.plot(y_l,test_crit,label=r'$\overline{U}.\frac{\mathrm{d}\Theta}{\mathrm{d}y}$')
plt.fill_between(y_l,borne,test_crit,alpha=0.3)

plt.xlabel(r'$y$')
plt.ylabel(r'Analysis')

plt.legend()






plt.show()

print('-----------------------------------------------------')

print('END')






