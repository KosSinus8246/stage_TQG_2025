import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt

# cf TQG notes : A.X = c.B.X
# @uthor : dimitri moreau 22/04/2025



##################################
# VARIABLES, ESPACE ...
##################################



Nx, Ny = 60, 60
N = Nx
Lx, Ly = 0, np.pi
dx, dy = Lx/Nx, Ly/Ny
x_l, y_l = np.linspace(0.1,Lx,Nx), np.linspace(0.1,Ly,Ny)


beta = 0 #1e-11
k, Rd = np.linspace(0.1,0.1+0.1*51,60),50 # à modifier
K2 = (k**2 + 1/(Rd**2))*dy**2
#K2 = 0
U0= 1


phi, theta, Un = np.zeros((Ny, Nx)), np.zeros((Ny, Nx)), U0*np.exp(-y_l**2)
phi_r,theta_r = phi.reshape(Nx*Ny), theta.reshape(Nx*Ny)

# X = np.array([phi_r,theta_r])

# V/G/Mn
F1 = 0
Vn, Mn = Un * dy**2, np.ones_like(Un)*2
G11 = 2.0*Un*(1-2*y_l**2) + F1*Un+beta
F11 = G11*dy**2
Theta0 = 1
G12 = -2*y_l*Theta0*np.exp(-y_l**2)

print('/////////////////////////////////////////////////////')
print('PARAMS : OK')



##################################
# CONSTRUCTION DE LA MATRICE B
##################################



# Diagonale principale
main_diag = -(2 + K2) * np.ones(N)
off_diag = np.ones(N-1)
# Construct tridiagonal B11 using np.diag
B11 = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
# Construct other blocks
B12 = np.zeros((N, N))
B21 = np.zeros((N, N))
B22 = np.eye(N)
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
A11 = np.zeros((N, N))
np.fill_diagonal(A11, main_diag_A11)

for i in range(N - 1):
    A11[i, i + 1] = Un[i]
    A11[i + 1, i] = Un[i]

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
#extract = c * X ou X
extract = c*X



phi = X[:N,:N]
phi2 = X[:N,N:]

theta = X[N:,:N]
theta2 = X[N:,N:]

PHI = np.real(phi + phi2)
THETA = np.real(theta + theta2)
print('COMPUTATION : OK')
print('/////////////////////////////////////////////////////')



##################################
# PLOT
##################################


fig, (ax) = plt.subplots(1,2,figsize=(15,6))

# vmin , vmax pour phi
if np.abs(np.min(PHI)) > np.abs(np.max(PHI)):
	vmin, vmax = np.min(PHI), -np.min(PHI)
else:
	vmin, vmax = -np.max(PHI), np.min(PHI)


im = ax[0].contourf(x_l,y_l,PHI,cmap='RdBu_r',vmin=vmin,vmax=vmax)
fig.colorbar(im,label=r'$\phi$')


# vmin , vmax pour theta
if np.abs(np.min(THETA)) > np.abs(np.max(THETA)):
	vmin, vmax = np.min(THETA), -np.min(THETA)
else:
	vmin, vmax = -np.max(THETA), np.min(THETA)


im = ax[1].contourf(x_l,y_l,THETA,cmap='RdBu_r',vmin=vmin,vmax=vmax)
fig.colorbar(im,label=r'$\Theta$')




phi = extract[:N,:N]
theta = extract[N:,:N]


phi2 = extract[:N,N:]
theta2 = extract[N:,N:]
fig, (ax) = plt.subplots(2,2,figsize=(15,10))

ax[0,0].set_title(r'$\phi$')
ax[0,0].set_ylabel('1')
im = ax[0,0].contourf(x_l,y_l,phi,cmap='RdBu_r')
fig.colorbar(im)
ax[1,0].set_ylabel('2')
im = ax[1,0].contourf(x_l,y_l,phi2,cmap='RdBu_r')
fig.colorbar(im)


im = ax[0,1].contourf(x_l,y_l,theta,cmap='RdBu_r')
fig.colorbar(im)
im = ax[1,1].contourf(x_l,y_l,theta2,cmap='RdBu_r')
fig.colorbar(im)
ax[0,1].set_title(r'$\Theta$')


'''
# test plot

idx = np.argsort(np.imag(c))         # sort eigenvalues by growth rate
dominant = idx[-1]                   # index of the most unstable mode

phi_mode = np.real(X[:N, dominant])
theta_mode = np.real(X[N:, dominant])

# Reconstruct full 2D fields
PHI_2D = np.real(np.outer(phi_mode, np.exp(1j * k * x_l)))
THETA_2D = np.real(np.outer(theta_mode, np.exp(1j * k * x_l)))



fig, (ax) = plt.subplots(1,2,figsize=(15,6))

im = ax[0].contourf(x_l,y_l,PHI_2D,cmap='RdBu_r')
fig.colorbar(im)

im = ax[1].contourf(x_l,y_l,THETA_2D,cmap='RdBu_r')
fig.colorbar(im)
'''




plt.show()


