import os
import numpy as np                                        
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.linalg import eig

mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'
#mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.family'] = 'Courier New'
mpl.rcParams['legend.edgecolor'] = '0'




# cf TQG notes : A.X = c.B.X is the system that is solved here
# and also the non thermal system
# @uthor : dimitri moreau 20/05/2025




print('-----------------------------------------------------')

##################################
# VARIABLES, SPACE ...
##################################




def compute_sigmas(Ny, Nk, dk, ymin, kmin, Ly, Lk, Lstar, beta, F1star, U0, Theta0_U0,config):


	font_size = 17

	choice_plot_name = 'max_sigma_im'
	
	Theta0 = Theta0_U0 *U0
	dy = (Ly - ymin)/Ny
	y_l, k = np.linspace(ymin,Ly,Ny), np.arange(kmin,Nk*dk,dk)
	
	if config == 'conf_1':
		Un = U0*np.exp(-y_l**2)
		G12 = -2*y_l*Theta0*np.exp(-y_l**2) # dThetabar/dy
		Lstar = None
	elif config == 'conf_2':
		Un = U0*np.exp(-y_l**2)
		G12 = -(2/Ly**2)*y_l*Theta0*np.exp(-(y_l**2)/(Lstar**2)) # dThetabar/dy
	
	
	else:
		import sys
		sys.exit("ERROR : NO CONFIGURATION")
		
	Vn = Un*(dy**2)


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
	
	
	print(sigmaNT_matrix.shape)


	print('COMPUTATION : OK')
	
	


	##################################
	# PLOT
	##################################


	print('/////////////////////////////////////////////////////')

	
	


	print('PLOT...')

	fig, (ax) = plt.subplots(1,1)

	ax.plot(k, val_c, 'k--', label='TQG')
	ax.plot(k, val_cNT, 'k-', label='QG')
	ax.set_xlabel(r'$k$')
	ax.set_ylabel(r'$\sigma_\mathbf{Im} = \mathbf{Im}\{c\}.k ~\geq~ 0$')
	ax.tick_params(top=True,right=True,direction='in',size=4,width=1)
	ax.legend(fancybox=False)
	ax.axhline(0, color='gray', linestyle=':')
	ax.axvline(0, color='gray', linestyle=':')
	ax.set_ylim(-0.01, 0.5)
	
	
	
	for spine in ax.spines.values():
	    spine.set_linewidth(2)
	
	
	# option
	return Un, G12, fig, (ax)
	
	print('END')
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')



# EXAMPLE

'''
Ny, Nk = 60, 51
dk = 0.1
ymin, kmin, Ly, Lk = 0.1, 0.1, np.pi, 0.1+dk*Nk


beta = 0 
F1star = 0 # 1/Rd**2

U0 = 1

compute_sigmas(False, Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, 1)
compute_sigmas(False, Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, 0.1)
plt.show()'''



