import os
import numpy as np
import seaborn as sns                                        
import matplotlib as mpl
import matplotlib.pyplot as plt

import imageio.v2 as imageio
from PIL import Image
from io import BytesIO
import os

from tqdm import tqdm
from scipy.linalg import eig


from scipy.optimize import curve_fit


mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'Courier New'
mpl.rcParams['legend.edgecolor'] = '0'

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_2_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~TARANIS~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')



# cf TQG notes : A.X = c.B.X is the system that is solved here
# and also the non thermal system
# @uthor : dimitri moreau 13/06/2025


def reg(x,y):

	covxy = np.cov(x,y)
	a = np.cov(x, y)[0, 1]/np.var(x) 
	b = np.mean(y) - a*np.mean(x)

	yhat = a*x+b
	err = y - yhat
	
	print(a)
	print(b)

	return yhat, err



#save_png = False  # create a folder im_para and a folder per
		 # experience with the "name_exp" variable
		 # save also the used parameters and Real
		 # and Imaginary sigma
font_size = 17
#choice_plot_name = 'max_sigma_im'

#name_exp = input('Name of the experience ?')

print('-----------------------------------------------------')

##################################
# VARIABLES, SPACE ...
##################################

Ny, Nk = 60, 51
Nk = 10

dk = 0.1
ymin, kmin, Ly, Lk = 0.1, 0.1, np.pi, 0.1+dk*Nk
dy = (Ly - ymin)/Ny

y_l, k = np.linspace(ymin,Ly,Ny), np.arange(kmin,Nk*dk,dk)


U0 = 1
Theta0_U0 = 1 # ratio
Theta0 = Theta0_U0 *U0


beta = 0
F1star = (2*Theta0 -4*U0*Theta0/np.pi)/2 # 1/Rd**2



Un = U0*np.exp(-y_l**2)
#Un = 1/(1+np.exp(-y_l)) # sigmoide

Vn = Un*(dy**2)
G12 = -2*y_l*Theta0*np.exp(-y_l**2) # dThetabar/dy


G11 = 2.0*Un*(1-2*y_l**2) + F1star*Un + beta - G12
F11 = G11*dy**2

print('/////////////////////////////////////////////////////')
print('PARAMS : OK')



print('/////////////////////////////////////////////////////')
print('COMPUTATION...')

sigma_matrix = np.zeros((len(k),2*Ny))
sigmaNT_matrix = np.zeros((len(k),Ny))

sigma_matrix_ree = np.zeros((len(k),2*Ny))
sigmaNT_matrix_ree = np.zeros((len(k),Ny))

sigma_tot = np.zeros(Ny*Ny) # for the eigenfrequencies later

# loop for each case of k
for ik in tqdm(range(len(k))):


	K2 = (k[ik]**2 + F1star)*dy**2



	##################################
	# CONSTRUCTION OF THE B MATRIX
	##################################

	
	B11 = np.zeros((Ny, Ny))
	
	
	for i in range(Ny):
		B11[i,i] = -(2 + K2)
		if i>0:
			B11[i,i-1] = 1.
		if i<Ny-1:
			B11[i,i+1] = 1.
	
	
	# Construct other blocks
	B12 = np.zeros((Ny, Ny))
	B21 = np.zeros((Ny, Ny))
	B22 = np.eye(Ny, Ny)


	B = np.block([[B11,B12],[B21,B22]])


	##################################
	# CONSTRUCTION OF THE A MATRIX
	##################################



	A11 = np.zeros((Ny,Ny))
	A11_star = np.zeros((Ny,Ny)) # same B11 without the thermal
	# term that is F11 for the non-TQG solving

	# Block A11
	
	for i in range(Ny):
	
		A11[i,i] = -Un[i] * (2 + K2) + F11[i]
		A11_star[i,i] = -Un[i] * (2 + K2) + F11[i] + G12[i]*dy**2
		if i>0:
			A11[i,i-1] = Un[i]
			A11_star[i,i-1] = Un[i]
		if i<Ny-1:
			A11[i,i+1] = Un[i]
			A11_star[i,i+1] = Un[i]
    

	
	# Block A12
	A12 = np.diag(-Vn)
	# Block A21
	A21 = np.diag(G12)
	# Block A22
	A22 = np.diag(Un)

	# Final block matrix A

	A = np.block([[A11,A12],[A21,A22]])



	
	# velocity odd
	A[0,1] = 2.0*A[0,1]
	B[0,1] = 2.0*B[0,1]
	
	
	# velocity not odd
	#A[0,1]=0.0
	#B[0,1]=0.0

	A[2*Ny-1,2*Ny-1] = 0.0
	B[2*Ny-1,2*Ny-1] = 0.0
	
	
	
	A11[0,1] = 2.0*A11[0,1]
	A11_star[0,1] = 2.0*A11_star[0,1]
	B11[0,1] = 2.0*B11[0,1]
	
	A11[Ny-1,Ny-1] = 0.0
	A11_star[Ny-1,Ny-1] = 0.0
	B11[Ny-1,Ny-1] = 0.0
	
	





	##################################
	# SOLUTION
	##################################
	# A.X = c.B.X 


	###### THERMAL SOLVING (TQG)

	c, X = eig(A,B)



	sigma = np.imag(c) * k[ik]
	sigma_matrix[ik,:] = sigma
	
	sigma_ree = np.real(c) * k[ik]
	sigma_matrix_ree[ik,:] = sigma_ree
	
	sigma_tot = c * k[ik]



	###### NON THERMAL SOLVING (QG)

	c_NT, X_NT = eig(A11_star,B11)

	sigma_NT = np.imag(c_NT) * k[ik]
	sigmaNT_matrix[ik,:] = sigma_NT
	
	sigma_NT_ree = np.real(c_NT) * k[ik]
	sigmaNT_matrix_ree[ik,:] = sigma_NT_ree

	




val_c = np.max(sigma_matrix, axis=1)       
val_cNT = np.max(sigmaNT_matrix, axis=1)

val_c_ree = np.max(sigma_matrix_ree, axis=1)       
val_cNT_ree = np.max(sigmaNT_matrix_ree, axis=1)


print('COMPUTATION : OK')


######################
# computes phi(t,y) and so ..

omega_matrice = np.zeros((len(c),len(k)))

phiy = X[0:Ny,0:Ny]
thetay = X[0:Ny,Ny:2*Ny]




for i in range(len(k)):
	omega_matrice[:,i] = c*k[i]









##################################
# PLOT
##################################


print('/////////////////////////////////////////////////////')



print('PLOT...')












fig, (ax) = plt.subplots(1,1)

ax.plot(k, val_c, 'k--', label='TQG')
ax.plot(k, val_cNT, 'k-', label='QG')
ax.set_xlabel(r'k', fontweight="bold")
#ax.set_ylabel(r'$\sigma_i = \mathbf{Im}\{c\}.k ~\geq~ 0$', fontweight="bold")
ax.set_ylabel(r'σ$_i$ ≥ Im{c}.k',fontweight='bold')
ax.tick_params(top=True,right=True,direction='in',size=4,width=1)
ax.legend(fancybox=False, prop={'weight': 'bold'})
ax.axhline(0, color='gray', linestyle=':')
ax.axvline(0, color='gray', linestyle=':')
for spine in ax.spines.values():
    spine.set_linewidth(2)
    
for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')

for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')
    
plt.tight_layout()



fig, (ax) = plt.subplots(1,2,figsize=(13,6))

ax[0].set_title('$y_\mathbf{Linear}=a.x+b$ and $y_\mathbf{Non-Linear}=a.\sqrt{x}+b.x+c$ fit')

ax[0].plot(k, val_c, 'k--', label='$\sigma_\mathbf{Im}$ TQG')
ax[0].plot(k, val_cNT, 'k-', label='$\sigma_\mathbf{Im}$ QG')
ax[0].set_xlabel(r'$k$')
ax[0].set_ylabel(r'$\sigma_\mathbf{Im} = \mathbf{Im}\{c\}.k ~\geq~ 0$')
ax[0].tick_params(top=True,right=True,direction='in',size=4,width=1)

ax[0].axhline(0, color='gray', linestyle=':')
ax[0].axvline(0, color='gray', linestyle=':')
for spine in ax[0].spines.values():
    spine.set_linewidth(2)








# FIT LINEAR
yhat, err = reg(k, val_c)
ax[0].plot(k,yhat,'r--',label='Linear fit : TQG')

yhatNT, errNT = reg(k, val_cNT)
ax[0].plot(k,yhatNT,'r-',label='Linear fit : QG')

# FIT NON-LINEAR
# Your custom model function (e.g., a quadratic)
def custom_model(x,a, b, c):
    return a*x**(0.5)+b*x+c

# Curve fitting
params, covariance = curve_fit(custom_model, k, val_c)
paramsNT, covarianceNT = curve_fit(custom_model, k, val_cNT)


ax[0].plot(k,custom_model(k, *params),'g--',label='Non-Linear fit : TQG')
ax[0].plot(k,custom_model(k, *paramsNT),'g-',label='Non-Linear fit : QG')
ax[0].legend(fancybox=False)











ax[1].set_title('Residual data')

ax[1].plot(k,err,'r--',label='TQG')
ax[1].plot(k,errNT,'r-',label='QG')

ax[1].plot(k,val_c - custom_model(k, *params),'g--',label='TQG')
ax[1].plot(k,val_cNT - custom_model(k, *paramsNT),'g-',label='QG')



ax[1].legend(fancybox=False)
ax[1].set_ylabel(r'$\sigma_i - \widehat{\sigma}_i$')
ax[1].tick_params(top=True,right=True,direction='in',size=4,width=1)
ax[1].axhline(0, color='gray', linestyle=':')
ax[1].axvline(0, color='gray', linestyle=':')
ax[1].set_xlabel(r'$k$')
for spine in ax[1].spines.values():
    spine.set_linewidth(2)

plt.tight_layout()







# Snapshots of Theta(y, t) at t = 0, 10, 20 for the most unstable mode
times = [0, 10, 20]
k_index = np.argmax(val_c)  # index of the most unstable k
omega = omega_matrice[:, k_index]
theta_eigvecs = thetay[:, k_index]

fig, ax = plt.subplots(1, len(times), figsize=(15, 5))

for i, t in enumerate(times):
	theta_t = np.real(theta_eigvecs * np.exp(1j * omega[k_index] * t))
	ax[i].plot(y_l, theta_t, 'b-')

	ax[i].set_title(r'Time : '+str(t))

	ax[i].tick_params(top=True,right=True,direction='in',size=4,width=1)

	ax[i].set_xlabel(r'$y$')
	ax[i].set_ylim(-5e-3,5e-3)
	
	for spine in ax[i].spines.values():
		spine.set_linewidth(2)


ax[0].set_ylabel(r'$\Theta(t,y)$')
ax[1].tick_params(labelleft=False)
ax[2].tick_params(labelleft=False)


plt.tight_layout()



from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as mcolors


fig, (ax) = plt.subplots(1,1)

t = np.linspace(0,33,500)



cmap = colormaps.get_cmap('plasma').resampled(len(val_c)-1)



# Plot each k mode with a color from the colormap
for i in range(0, len(val_c)):
	color = cmap(i-1)
	energy = np.exp(2*val_c[i]*t)
	ax.plot(t,energy, color=color)



for spine in ax.spines.values():
	spine.set_linewidth(2)
	
for tick in ax.get_xticklabels():
    	tick.set_fontweight('bold')

for tick in ax.get_yticklabels():
    	tick.set_fontweight('bold')
ax.tick_params(top=True,right=True,direction='in',size=4,width=1)
ax.set_xlabel('Time',fontweight='bold')
#ax.set_ylabel(r'Energy : $\mathbf{exp}(2.\sigma_i.t)$',fontweight='bold')
ax.set_ylabel(r'Energy : exp(2.σ$_i$.t)',fontweight='bold')

ax.axhline(0, color='gray', linestyle=':')
ax.axvline(0, color='gray', linestyle=':')



sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=len(val_c)-1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r'Mode index', fontweight='bold')



cbar.ax.yaxis.set_ticks_position('both')             # Ticks on both sides
cbar.ax.yaxis.set_tick_params(labelleft=False,       # Hide left labels
                               direction='in',    # Tick style
                               length=2,width=1)            # Length of ticks for visibilit


# Set the border (spine) linewidth of the colorbar
for spine in cbar.ax.spines.values():
	spine.set_linewidth(1.5)  # You can set this to any float value

# Set tick labels bold
for tick in cbar.ax.get_yticklabels():
	tick.set_fontweight('bold')

# Set spine linewidth
for spine in cbar.ax.spines.values():
	spine.set_linewidth(1.5)

ax.set_xlim(None, t[-1])
ax.set_yscale('log')
from matplotlib.ticker import LogLocator

ax.tick_params(axis='y', which='both', direction='in', length=4, width=1, color='k', labelcolor='k')
ax.tick_params(top=True, right=True, direction='in', length=4, width=1)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
ax.yaxis.set_ticks_position('both')  # Ticks on both left and right

plt.show()

print('END')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')




