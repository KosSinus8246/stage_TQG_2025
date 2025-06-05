import os
import numpy as np                                        
import matplotlib as mpl
import matplotlib.pyplot as plt

import imageio.v2 as imageio
from PIL import Image
from io import BytesIO
import os

from tqdm import tqdm
from scipy.linalg import eig

mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'Courier New'
mpl.rcParams['legend.edgecolor'] = '0'

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_2_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')



# cf TQG notes : A.X = c.B.X is the system that is solved here
# and also the non thermal system
# @uthor : dimitri moreau 20/05/2025


save_png = False  # create a folder im_para and a folder per
		 # experience with the "name_exp" variable
		 # save also the used parameters and Real
		 # and Imaginary sigma
font_size = 17
choice_plot_name = 'max_sigma_im'

name_exp = input('Name of the experience ?')

print('-----------------------------------------------------')

##################################
# VARIABLES, SPACE ...
##################################

Ny, Nk = 60, 51
dk = 0.1
ymin, kmin, Ly, Lk = 0.1, 0.1, np.pi, 0.1+dk*Nk
#Ly = 25.6 # article

dy = (Ly - ymin)/Ny

y_l, k = np.linspace(ymin,Ly,Ny), np.arange(kmin,5,dk)


beta = np.round(np.linspace(0, 3, 15), 3)
#beta = 0
F1star = 0 # 1/Rd**2

U0 = 1
Theta0_U0 = 1 # ratio
Theta0 = Theta0_U0 *U0



contourf_beta_k_matrix = np.zeros((len(beta),len(k)))
contourf_beta_k_matrix_NT = np.zeros((len(beta),len(k)))

times = np.linspace(0,200,100)

ix = 0



for var in beta:

	Un = U0*np.exp(-y_l**2)
	#Un = (2/np.sqrt(np.pi)) * np.exp(-y_l**2) -2/Ly #article

	Vn = Un*(dy**2)
	G12 = -2*y_l*Theta0*np.exp(-y_l**2) # dThetabar/dy
	
	#G12 = -np.exp(-y_l**2) + 2/Ly #article 



	G11 = 2.0*Un*(1-2*y_l**2) + F1star*Un + var - G12
	F11 = G11*dy**2

	print('/////////////////////////////////////////////////////')
	print('PARAMS : OK')


	# Save all parameters in a txt file


	print('/////////////////////////////////////////////////////')
	print('COMPUTATION...')

	#sigma_matrix = np.zeros([len(beta),len(k),2*Ny])
	#sigmaNT_matrix = np.zeros([len(beta),len(k),Ny])

	#sigma_matrix_ree = np.zeros((len(k),2*Ny))
	#sigmaNT_matrix_ree = np.zeros((len(k),Ny))

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
	
	
	contourf_beta_k_matrix[ix,:] = val_c
	contourf_beta_k_matrix_NT[ix,:] = val_cNT

	
	indice = np.argmax(val_c)
	E = np.exp(2*val_c[indice]*times)
	#plt.plot(times,E)
	#plt.yscale('log')
	
	
	ix = ix+1
	print(ix)



print('COMPUTATION : OK')


##################################
# PLOT
##################################


print('/////////////////////////////////////////////////////')



print('PLOT...')
#res = 5
res = np.arange(0.0,0.5,0.05)


fig, (ax) = plt.subplots(1,2,figsize=(13,6))
cs = ax[0].contour(k,beta,contourf_beta_k_matrix,res,cmap='Blues')
#ax[0].contourf(k,beta,contourf_beta_k_matrix,res,cmap='Grays')
ax[0].clabel(cs,fontsize=15,colors='k')
ax[0].set_xlabel(r'$k$')
ax[0].set_ylabel(r'$\beta$')
ax[0].set_title(r'Contours : $\sigma_\mathbf{TQG}$')
ax[0].tick_params(top=True,right=True,direction='in',size=4,width=1)

ax[0].set_ylim(np.min(beta), np.max(beta))
ax[0].set_xlim(0.1, np.max(k))

for spine in ax[0].spines.values():
    spine.set_linewidth(2)


cs = ax[1].contour(k,beta,contourf_beta_k_matrix_NT,res,cmap='Oranges')
#ax[1].contourf(k,beta,contourf_beta_k_matrix_NT,res,cmap='Grays')
ax[1].clabel(cs,fontsize=15,colors='k')
ax[1].set_xlabel(r'$k$')
ax[1].set_title(r'Contours : $\sigma_\mathbf{QG}$')
ax[1].tick_params(top=True,right=True, labelleft=False,direction='in',size=4,width=1)
for spine in ax[1].spines.values():
    spine.set_linewidth(2)
    
ax[1].set_ylim(np.min(beta), np.max(beta))
ax[1].set_xlim(0.1, np.max(k))









fig, (ax) = plt.subplots(1,1)

cs = ax.contour(k,beta,contourf_beta_k_matrix,res,colors='C0',alpha=0.6)
cs2 = ax.contour(k,beta,contourf_beta_k_matrix_NT,res,colors='C1',alpha=0.6)

ax.clabel(cs,fontsize=15)
ax.clabel(cs2,fontsize=15)
ax.set_ylim(np.min(beta), np.max(beta))
ax.set_xlim(0.1, np.max(k))
ax.set_title(r'Contours : $\sigma_\mathbf{TQG} ~~;~~ \sigma_\mathbf{QG}$')

ax.tick_params(top=True,right=True,direction='in',size=4,width=1)

for spine in ax.spines.values():
    spine.set_linewidth(2)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='C0', edgecolor='k',alpha=0.6, label='TQG'),
    Patch(facecolor='C1', edgecolor='k',alpha=0.6, label='QG')]

ax.legend(handles=legend_elements, fancybox=False)
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\beta$')









fig, (ax) = plt.subplots(1,1)

cs = ax.contour(k, beta, contourf_beta_k_matrix-contourf_beta_k_matrix_NT,10, cmap='Reds',alpha=0.65)
ax.clabel(cs,fontsize=15,colors='k')

ax.set_title(r'Contours : $\sigma_\mathbf{TQG} - \sigma_\mathbf{QG}$')
ax.set_ylim(np.min(beta), np.max(beta))
ax.set_xlim(0.1, np.max(k))

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\beta$')

ax.tick_params(top=True,right=True,direction='in',size=4,width=1)

for spine in ax.spines.values():
    spine.set_linewidth(2)


  

    
    
  

plt.show()

print('END')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')




