import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.linalg import eig



mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'Courier New'
mpl.rcParams['legend.edgecolor'] = '0'

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_3_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~JULIE~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')




##################################
# VARIABLES, SPACE ...
##################################






N, Nk = 20, 51

dk = 0.1
Lmin =0.1
L = np.pi
kmin, Lk = 0.1, 0.1+dk*Nk
#dy, dx = (Ly - ymin)/Ny, (Lx - xmin)/Nx
dh = L/N


###################
# fully option
regula = L / 6
##################



x_l, y_l, k = np.linspace(Lmin,L,N), np.linspace(Lmin,L,N), np.arange(kmin,Nk*dk,dk)
xx, yy = np.meshgrid(x_l,y_l)


beta = 0.
F1star = 0. # 1/Rd**2

U0 = 1.
Theta0_U0 = 1. # ratio
Theta0 = Theta0_U0 *U0


#Un = U0*np.exp(-yy**2)
Un = U0 * np.exp(-((yy - L/2)/regula)**2) # regula conf


Vn = Un*(dh**2)

Thetabar = Theta0 * np.tanh((yy - L/2)/regula) # regula conf
G12 = np.gradient(Thetabar, axis=0) / dh  # ∂Θ̄/∂y # regula conf

#G12 = -2*yy*Theta0*np.exp(-yy**2) # dThetabar/dy


#G11 = 2.0*Un*(1-2*yy**2) + F1star*Un + beta - G12
G11 = 2 * Un * (1 - 2 * ((yy - L/2)/regula)**2) + F1star * Un + beta - G12 # regulaconf
F11 = G11*dh**2






k0, l0 = 0.73, 0.

K2 = (k0**2+l0**2 + F1star)*dh**2


##################################
# CONSTRUCTION OF THE B MATRIX
##################################


B11 = np.zeros((N*N, N*N))

'''
for i in range(N*N):
	B11[i,i] = -(2 + K2)
	if i>0:
		B11[i,i-1] = 1.
	if i<(N*N)-1:
		B11[i,i+1] = 1.'''
		
def ij_to_index(i, j, N):
    return i * N + j

for i in range(N):
	for j in range(N):
		#idx = ij_to_index(i, j, N)
		idx = i * N + j
		B11[idx, idx] = -4 - K2
		if i > 0:
		    B11[idx, ij_to_index(i-1, j, N)] = 1
		if i < N-1:
		    B11[idx, ij_to_index(i+1, j, N)] = 1
		if j > 0:
		    B11[idx, ij_to_index(i, j-1, N)] = 1
		if j < N-1:
		    B11[idx, ij_to_index(i, j+1, N)] = 1

		
		



# Construct other blocks
B12 = np.zeros((N*N, N*N))
B21 = np.zeros((N*N, N*N))
B22 = np.eye(N*N, N*N)


B = np.block([[B11,B12],[B21,B22]])


print('MATRIX B : OK')


##################################
# CONSTRUCTION OF THE A MATRIX
##################################



A11 = np.zeros((N*N,N*N))
A11_star = np.zeros((N*N,N*N))# same B11 without the thermal
# term that is F11 for the non-TQG solving
# Block A11

'''
for i in range(N*N):

	A11[i,i] = -Un.ravel()[i] * (2 + K2) + F11.ravel()[i]
	#A11_star[i,i] = -Un[i] * (2 + K2) + F11[i] + G12[i]*dh**2
	if i>0:
		A11[i,i-1] = Un.ravel()[i]
		#A11_star[i,i-1] = Un.ravel()[i]
	if i<(N*N)-1:
		A11[i,i+1] = Un.ravel()[i]
		#A11_star[i,i+1] = Un[i]'''
		
		
for i in range(N):
	for j in range(N):
		idx = i * N + j
		A11[idx, idx] = -Un[i, j] * (4 + K2) + F11[i, j]
		A11_star[idx, idx] = -Un[i, j] * (4 + K2) + F11[i, j] + G12[i, j]

		if i > 0:
			A11[idx, ij_to_index(i - 1, j, N)] = Un[i, j]
			A11_star[idx, ij_to_index(i - 1, j, N)] = Un[i, j]
		if i < N - 1:
			A11[idx, ij_to_index(i + 1, j, N)] = Un[i, j]
			A11_star[idx, ij_to_index(i + 1, j, N)] = Un[i, j]
		if j > 0:
			A11[idx, ij_to_index(i, j - 1, N)] = Un[i, j]
			A11_star[idx, ij_to_index(i, j - 1, N)] = Un[i, j]
		if j < N - 1:
			A11[idx, ij_to_index(i, j + 1, N)] = Un[i, j]
			A11_star[idx, ij_to_index(i, j + 1, N)] = Un[i, j]




# Block A12
A12 = np.diag(-Vn.ravel())
# Block A21
A21 = np.diag(G12.ravel())
# Block A22
A22 = np.diag(Un.ravel())

# Final block matrix A

A = np.block([[A11,A12],[A21,A22]])

print('MATRIX A : OK')



# velocity odd
A[0,1] = 2.0*A[0,1]
B[0,1] = 2.0*B[0,1]


# velocity not odd
#A[0,1]=0.0
#B[0,1]=0.0

A[2*(N*N)-1,2*(N*N)-1] = 0.0
B[2*(N*N)-1,2*(N*N)-1] = 0.0



A11[0,1] = 2.0*A11[0,1]
A11_star[0,1] = 2.0*A11_star[0,1]
B11[0,1] = 2.0*B11[0,1]

A11[(N*N)-1,(N*N)-1] = 0.0
A11_star[(N*N)-1,(N*N)-1] = 0.0
B11[(N*N)-1,(N*N)-1] = 0.0






##################################
# SOLVING
##################################



c, X = eig(A,B)
c_NT, X_NT = eig(A11_star, B11)



print('COMPUTATION : OK')
##################################
# PLOT
##################################


# Parameters for plotting
timesteps = [0, 10, 20, 30]  # time points
mode_index = 11  # choose dominant or a specific mode

# TQG ##################
########################
# Extract eigenvalue and eigenvector
c_mode = c[mode_index]  # eigenvalue
X_mode = X[:, mode_index]  # eigenvector

# Extract Theta from second half of X
#Theta_flat = X_mode[N*N:]
phi_flat = X_mode[:N*N]
phi_xy = phi_flat.reshape((N, N))

# Normalize for visualization if needed
PHI_xy = np.real(phi_xy)





# QG ##################
########################
# Extract eigenvalue and eigenvector
c_NT_mode = c_NT[mode_index]  # eigenvalue
X_NT_mode = X_NT[:, mode_index]  # eigenvector

# Extract Theta from second half of X
#Theta_flat = X_mode[N*N:]
phi_flat_NT = X_NT_mode
phi_xy_NT = phi_flat_NT.reshape((N, N))

# Normalize for visualization if needed
PHI_xy_NT = np.real(phi_xy_NT)






# Time evolution
fig, axs = plt.subplots(2, len(timesteps), figsize=(16, 7))
fig.suptitle(r'Evolution of $\phi(t,x,y)$')

for i, t in enumerate(timesteps):
	PHI_t = np.real(PHI_xy * np.exp(c_mode * t))
	PHI_t_NT = np.real(PHI_xy_NT * np.exp(c_NT_mode * t))
	
	im = axs[0,i].pcolormesh(x_l,y_l,PHI_t,cmap='RdBu_r')
	axs[0,i].contour(x_l,y_l,PHI_t,colors='k')
	axs[0,i].set_title(f"t = {t}")
	axs[0,i].set_xlabel(r"$x$")
	axs[0,i].set_ylabel(r"$y$")
	fig.colorbar(im, ax=axs[0,i])
	axs[0,i].tick_params(top=True,right=True,direction='in',size=4,width=1)
	for spine in axs[0,i].spines.values():
		spine.set_linewidth(2)
	
	
	
	im = axs[1,i].pcolormesh(x_l,y_l,PHI_t_NT,cmap='coolwarm')
	axs[1,i].contour(x_l,y_l,PHI_t_NT,colors='k')
	axs[1,i].set_xlabel(r"$x$")
	axs[1,i].set_ylabel(r"$y$")
	fig.colorbar(im, ax=axs[1,i])
	axs[1,i].tick_params(top=True,right=True,direction='in',size=4,width=1)
	
	for spine in axs[1,i].spines.values():
		spine.set_linewidth(2)
	
	

plt.tight_layout()
plt.show()








