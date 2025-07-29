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

# cf TQG notes : A.X = c.B.X is the 2D system that is solved here
# and also the 2D non thermal system
# @uthor : dimitri moreau 27/06/2025


##################################
# VARIABLES, SPACE ...
##################################



colormap = 'RdBu_r'
lim_TQG, lim_QG = 1., 10.
levels = 30


N = 25

Lmin = 0.1
L = np.pi
#kmin, Lk = 0.1, 0.1+dk*Nk
#dy, dx = (Ly - ymin)/Ny, (Lx - xmin)/Nx
dh = L/N



x_l, y_l = np.linspace(Lmin,L,N), np.linspace(Lmin,L,N)
#k = np.arange(kmin,Nk*dk,dk)
xx, yy = np.meshgrid(x_l,y_l)


beta = 0.
F1star = 0. # 1/Rd**2

U0 = 1.
Theta0_U0 = 1. # ratio
Theta0 = Theta0_U0 *U0

mode_index = int(input('Which mode ? ')) # mode to observe


#Un = U0*np.exp(-yy**2)
Un = U0*np.exp(-(yy-L/2)**2)


Vn = Un*(dh**2)


#G12 = -2*yy*Theta0*np.exp(-yy**2) #-2*xx*Theta0*np.exp(-xx**2) # dThetabar/dy
G12 = -2*yy*Theta0*np.exp(-yy**2) -2*xx*Theta0*np.exp(-xx**2) # dThetabar/dy


G11 = 2.0*Un*(1-2*yy**2) + F1star*Un + beta - G12
F11 = G11*dh**2






k0, l0 = 2., 0.

K2 = (k0**2+l0**2 + F1star)*dh**2

print('/////////////////////////////////////////////////////')
print('PARAMS : OK')


##################################
# CONSTRUCTION OF THE B MATRIX
##################################


B11 = np.zeros((N*N, N*N))

		
def ij_to_index(i, j, N):
    return i * N + j

for i in tqdm(range(N)):
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

		
		
for i in tqdm(range(N)):
	for j in range(N):
		idx = i * N + j
		A11[idx, idx] = -Un[i, j] * (4 + K2) + F11[i, j]
		A11_star[idx, idx] = -Un[i, j] * (4 + K2) + F11[i, j] + G12[i,j]*dh**2

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


'''
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
B11[(N*N)-1,(N*N)-1] = 0.0'''


# Apply closed boundary conditions (ψ = 0) on all edges
for i in range(N):
	for j in range(N):
		idx = ij_to_index(i, j, N)

		if i == 0 or i == N-1 or j == 0 or j == N-1:
			# Zero out the row and set diagonal to 1 (Dirichlet ψ=0)
			A11[idx, :] = 0.0
			A11[idx, idx] = 1.0
			A11_star[idx, :] = 0.0
			A11_star[idx, idx] = 1.0
			B11[idx, :] = 0.0
			B11[idx, idx] = 1.0







##################################
# SOLVING
##################################
print('/////////////////////////////////////////////////////')
print('COMPUTATION...')


c, X = eig(A,B)
c_NT, X_NT = eig(A11_star, B11)


print('EIGENVALUES AND EIGENVECTORS : OK')


##################################
# PLOT
##################################


# Parameters for plotting
timesteps = [0.0, 0.25, 0.5, 0.75]  # time points




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


# extract theta
theta_flat = X_mode[N*N:]
theta_xy = theta_flat.reshape((N, N))
THETA_xy = np.real(theta_xy)




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









print('COMPUTATION : OK')



print('/////////////////////////////////////////////////////')
print('PLOT...')




# Time evolution
fig, axs = plt.subplots(2, len(timesteps), figsize=(16, 7))
fig.suptitle(r'Evolution of $\zeta_\mathbf{TQG}$ and $\zeta_\mathbf{QG}$ : mode n°'+str(mode_index))


# figure for theta
fig2, axs2 = plt.subplots(1, len(timesteps), figsize=(16, 4))


for i, t in enumerate(timesteps):
	PHI_t = np.real(PHI_xy * np.exp(c_mode * t))
	THETA_t = np.real(THETA_xy * np.exp(c_mode * t))
	PHI_t_NT = np.real(PHI_xy_NT * np.exp(c_NT_mode * t))
	
	PSI = np.real(PHI_t* np.exp(1j*(k0*xx+l0*yy - c_mode*t))) 
	THETA = np.real(THETA_t * np.exp(1j*(k0*xx+l0*yy - c_mode*t)))
	PSI_NT = np.real(PHI_t_NT* np.exp(1j*(k0*xx+l0*yy - c_NT_mode*t)))

	# VELOCITIES
	u_s, v_s = np.zeros_like(PSI), np.zeros_like(PSI)
	u_sNT, v_sNT = np.zeros_like(PSI), np.zeros_like(PSI)
	zeta, zeta_NT = np.zeros_like(PSI), np.zeros_like(PSI)


	for j in range(len(x_l)-1):
		for k in range(len(y_l)-1):
			u_s[j,k] = - (PSI[j+1,k] - PSI[j-1,k])/(2*dh) # -dpsi/dy
			v_s[j,k] =   (PSI[j,k+1] - PSI[j,k-1])/(2*dh) # dpsi/dx

			u_sNT[j,k] = - (PSI_NT[j+1,k] - PSI_NT[j-1,k])/(2*dh)
			v_sNT[j,k] =   (PSI_NT[j,k+1] - PSI_NT[j,k-1])/(2*dh)
			
			zeta[j,k] = (PSI[j,k+1] -2*PSI[j,k] + PSI[j,k-1])/(dh**2) +\
			 (PSI[j+1,k] -2*PSI[j,k] + PSI[j-1,k])/(dh**2) 
			zeta_NT[j,k] = (PSI_NT[j,k+1] -2*PSI_NT[j,k] + PSI_NT[j,k-1])/(dh**2) +\
			 (PSI_NT[j+1,k] -2*PSI_NT[j,k] + PSI_NT[j-1,k])/(dh**2) 



	#print('////////////////////////////////')
	#print(np.isnan(zeta).any())
	
	
	
	
	im1 = axs[0,i].contourf(x_l,y_l,zeta,levels, cmap=colormap,vmin=-lim_TQG,vmax=lim_TQG)
	#im1 = axs[0,i].pcolormesh(x_l,y_l,PSI,cmap=colormap,vmin=-lim_TQG,vmax=lim_TQG)
	
	#cs = axs[0,i].contour(x_l,y_l,PSI,levels,colors='k')
	#axs[0,i].clabel(cs)
	axs[0,i].streamplot(x_l,y_l,u_s,v_s,color='k',linewidth=0.5,arrowsize=0.75)

	
	axs[0,i].set_title(f"t = {t}")
	
	axs[0,i].set_xlim(Lmin,L)
	axs[0,i].set_ylim(Lmin,L)

	#fig.colorbar(im, ax=axs[0,i],extend='both')
	axs[0,i].tick_params(top=True,right=True,labelbottom=False,direction='in',size=4,width=1)
	
	for spine in axs[0,i].spines.values():
		spine.set_linewidth(2)
	
	
	
	im2 = axs[1,i].contourf(x_l,y_l,zeta_NT,levels, cmap=colormap,vmin=-lim_QG,vmax=lim_QG)
	#im2 = axs[1,i].pcolormesh(x_l,y_l,PSI_NT,cmap=colormap,vmin=-lim_QG,vmax=lim_QG)

	#cs = axs[1,i].contour(x_l,y_l,PSI_NT,levels,colors='k')
	#axs[1,i].clabel(cs)

	axs[1,i].streamplot(x_l,y_l,u_sNT,v_sNT,color='k',linewidth=0.5,arrowsize=0.75)


	axs[1,i].set_xlabel(r"$x$")
	axs[1,i].set_xlim(Lmin,L)
	axs[1,i].set_ylim(Lmin,L)

	#fig.colorbar(im, ax=axs[1,i],extend='both')
	axs[1,i].tick_params(top=True,right=True,direction='in',size=4,width=1)
	
	for spine in axs[1,i].spines.values():
		spine.set_linewidth(2)
		
		
		
	imtheta = axs2[i].contourf(x_l,y_l,THETA,levels, cmap=colormap,vmin=-1,vmax=1)
		
		
		
		
		
		
		
		
		
		
		
		
	
	
	
axs[0,0].set_ylabel(r'$y$')
cbar_1 = fig.colorbar(im1, ax=axs[0,-1])
cbar_1.ax.yaxis.set_ticks_position('both')             # Ticks on both sides
cbar_1.ax.yaxis.set_tick_params(labelleft=False,       # Hide left labels
                               direction='in',    # Tick style
                               length=2,width=1)            # Length of ticks for visibility

# Set the border (spine) linewidth of the colorbar
for spine in cbar_1.ax.spines.values():
	spine.set_linewidth(1.5)  # You can set this to any float value


# Set tick labels bold
for tick in cbar_1.ax.get_yticklabels():
    tick.set_fontweight('bold')

# Set spine linewidth
for spine in cbar_1.ax.spines.values():
    spine.set_linewidth(1.5)




axs[0,1].tick_params(top=True,right=True,labelbottom=False,labelleft=False,direction='in',size=4,width=1)
axs[0,2].tick_params(top=True,right=True,labelbottom=False,labelleft=False,direction='in',size=4,width=1)
axs[0,3].tick_params(top=True,right=True,labelbottom=False,labelleft=False,direction='in',size=4,width=1)

axs[1,0].set_ylabel(r'$y$')
cbar_2 = fig.colorbar(im2, ax=axs[1,-1])
cbar_2.ax.yaxis.set_ticks_position('both')             # Ticks on both sides
cbar_2.ax.yaxis.set_tick_params(labelleft=False,       # Hide left labels
                               direction='in',    # Tick style
                               length=2,width=1)            # Length of ticks for visibility

# Set tick labels bold
for tick in cbar_2.ax.get_yticklabels():
    tick.set_fontweight('bold')

# Set spine linewidth
for spine in cbar_2.ax.spines.values():
    spine.set_linewidth(1.5)



# Set the border (spine) linewidth of the colorbar
for spine in cbar_2.ax.spines.values():
	spine.set_linewidth(1.5)  # You can set this to any float value


axs[1,1].tick_params(top=True,right=True,labelbottom=True,labelleft=False,direction='in',size=4,width=1)
axs[1,2].tick_params(top=True,right=True,labelbottom=True,labelleft=False,direction='in',size=4,width=1)
axs[1,3].tick_params(top=True,right=True,labelbottom=True,labelleft=False,direction='in',size=4,width=1)






plt.tight_layout()


def get_ix(c, c_NT):
	norm_cNT = (np.real(c_NT)**2 + np.imag(c_NT)**2)**0.5
	norm_cNT__ = np.sort(norm_cNT)[::-1]
	ix_norm_cNT__ = np.argsort(norm_cNT)[::-1]
	
	norm_c = (np.real(c)**2 + np.imag(c)**2)**0.5
	norm_c__ = np.sort(norm_c)[::-1]
	ix_norm_c__ = np.argsort(norm_c)[::-1]
	
	return norm_cNT__, norm_c__
	
norm_cNT__, norm_c__ = get_ix(c,c_NT)
	
	


percent = norm_c__[2:]/np.nansum(norm_c__[2:])

cumsun_f = np.nancumsum(percent)

fig, (ax) = plt.subplots(1,2,figsize=(15,4))

ax[0].plot(percent,'k',label='norm')
ax[0].plot(np.linspace((10/100)*np.max(percent),(10/100)*np.max(percent),len(percent)),'--',label=r'$10\%$')
ax[0].plot(np.linspace((50/100)*np.max(percent),(50/100)*np.max(percent),len(percent)),'--',label=r'$50\%$')
ax[0].legend()

ax[1].plot(cumsun_f,'r',label='cumsum')
ax[1].plot(np.linspace(0.9,0.9,len(cumsun_f)),'--',label=r'$90\%$')
ax[1].legend()





#np.savetxt('zeta_'+str(mode_index)+'.txt', zeta, fmt='%.2f')  # '%d' is for integers




plt.show()



print('END')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')




