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



print('COMPUTATION : OK')
print('/////////////////////////////////////////////////////')

'''#extract = c * X ou X
extract = c*X



phi = X[:N,:N]
phi2 = X[:N,N:]

theta = X[N:,:N]
theta2 = X[N:,N:]

PHI = np.real(phi + phi2)
THETA = np.real(theta + theta2)'''




##################################
# PLOT
##################################



print('PLOT...')



'''
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
ax[0,1].set_title(r'$\Theta$')'''




####################
# test plot



# Extract dominant eigenmode (largest eigenvalue)
idx = np.argmax(np.real(c))  # Index of the most dominant eigenvalue
phi_mode = np.real(X[:Ny, idx])
theta_mode = np.real(X[Ny:, idx])
# Reconstruct 2D fields (assume you're in 2D space)
PHI_2D = np.outer(phi_mode, np.exp(1j * k * y_l))  # Example, modify as needed
THETA_2D = np.outer(theta_mode, np.exp(1j * k * y_l))  # Example, modify as needed
# Plot reconstructed 2D fields
fig, (ax) = plt.subplots(1, 2, figsize=(15, 6))
im_phi = ax[0].pcolormesh(k, y_l, np.real(PHI_2D), cmap='RdBu_r')
ax[0].set_xlabel(r'$k$')
ax[0].set_ylabel(r'$y$')
fig.colorbar(im_phi, ax=ax[0])
ax[0].set_title(r'Reconstructed $\phi$')

im_theta = ax[1].pcolormesh(k, y_l, np.real(THETA_2D), cmap='RdBu_r')
fig.colorbar(im_theta, ax=ax[1])
ax[1].set_title(r'Reconstructed $\Theta$')
ax[1].set_xlabel(r'$k$')
ax[1].set_ylabel(r'$y$')



plt.figure(figsize=(8, 6))
plt.scatter(np.real(c), np.imag(c), color='b', label='Eigenvalues')
plt.axhline(0, color='k', linestyle='--', label='Real axis')
plt.axvline(0, color='k', linestyle='--', label='Imaginary axis')
plt.title('Eigenvalue Spectrum')
plt.xlabel('Real part of eigenvalue')
plt.ylabel('Imaginary part of eigenvalue')
plt.legend()



'''
fig, (ax) = plt.subplots(1,2,figsize=(15,6))

ax[0].plot(k,np.real(PHI_2D[0,:]))
ax[0].plot(k,np.real(PHI_2D[1,:]))
ax[0].set_xlabel(r'$k$')
ax[0].set_ylabel(r'$\phi$')
ax[0].legend()

ax[1].plot(k,np.real(THETA_2D[0,:]))
ax[1].plot(k,np.real(THETA_2D[1,:]))
ax[1].set_xlabel(r'$k$')
ax[1].set_ylabel(r'$\Theta$')
ax[1].legend()







phi_mode = X[:Ny, idx]
theta_mode = X[Ny:, idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(y_l,np.real(phi_mode), label='Re($\phi$)')
ax1.plot(y_l,np.imag(phi_mode), label='Im($\phi$)', linestyle='--')
ax1.set_ylabel(r'$\phi$')
ax1.set_xlabel(r'$y$')
ax1.legend()
ax1.set_title('structure of $\phi$')

ax2.plot(y_l, np.real(theta_mode), label='Re($\Theta$)')
ax2.plot(y_l, np.imag(theta_mode), label='Im($\Theta$)', linestyle='--')
ax2.set_ylabel(r'$\Theta$')
ax2.set_xlabel(r'$y$')
ax2.legend()
ax2.set_title('structure of $\Theta$')
plt.tight_layout()'''



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






