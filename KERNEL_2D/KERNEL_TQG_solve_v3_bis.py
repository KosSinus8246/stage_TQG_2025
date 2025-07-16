import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.linalg import eig
import sys



'''
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_3_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~JULIE~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')'''

# cf TQG notes : A.X = c.B.X is the 2D system that is solved here
# and also the 2D non thermal system
# @uthor : dimitri moreau 16/07/2025


def get_ix(c, c_NT,nb_modes):

	'''
	Function that select the most important modes following
	the desired criteria
	'''

	norm_cNT = (np.imag(c_NT)**2)**0.5
	#norm_cNT = (np.real(c_NT)**2 + np.imag(c_NT)**2)**0.5
	norm_cNT__ = np.sort(norm_cNT)[::-1]
	ix_norm_cNT__ = np.argsort(norm_cNT)[::-1]
	
	norm_c = (np.real(c)**2 + np.imag(c)**2)**0.5
	#norm_c = (np.imag(c)**2)**0.5
	norm_c__ = np.sort(norm_c)[::-1]
	ix_norm_c__ = np.argsort(norm_c)[::-1]
	
	
	if nb_modes > len(c):
		print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		print("!  THE NUMBER OF MODES REQUESTED IS TOO IMPORTANT  !")
		print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		sys.exit()
	
	

	fig, (ax) = plt.subplots(1,1)

	# [2:] to remove the inf of begin and end because they are sorted

	ax.plot(100*norm_c__[2:]/np.nanmax(norm_c__[2:]),'k',label='TQG')
	ax.plot(100*norm_cNT__[2:]/np.nanmax(norm_cNT__[2:]),'r',label='QG')
	ax.axvline(nb_modes, 0, 100, color='C1', linestyle='--')
	
	ax.set_xlabel('Number of mode', fontweight="bold")
	ax.set_ylabel(r'Importance of the sorted mode', fontweight="bold")
	ax.legend(fancybox=False, prop={'weight': 'bold'})
	ax.tick_params(top=True,right=True,direction='in',size=4,width=1)
	
	ax.axhline(0, color='gray', linestyle=':')
	ax.axvline(0, color='gray', linestyle=':')
	
	

	
	for tick in ax.get_xticklabels():
	    tick.set_fontweight('bold')
	for tick in ax.get_yticklabels():
	    tick.set_fontweight('bold')
	for spine in ax.spines.values():
		spine.set_linewidth(2)

	
	return ix_norm_c__, ix_norm_cNT__





def compute_TQG_2D(N, Lmin, L, beta, F1star, U0, Theta0_U0, k0, l0, dh, BC):

	'''
	Function that computes eigenvalues and eigenvectors
	'''


	x_l, y_l = np.linspace(Lmin,L,N), np.linspace(Lmin,L,N)
	xx, yy = np.meshgrid(x_l,y_l)
	
	Theta0 = Theta0_U0 *U0
	
	Un = U0*np.exp(-(yy-L/2)**2)
	
	
	Vn = Un*(dh**2)
	K2 = (k0**2+l0**2 + F1star)*dh**2


	G12 = -2*yy*Theta0*np.exp(-yy**2) # dThetabar/dy
	Thetabar = Theta0* np.exp(-yy**2)


	G11 = 2.0*Un*(1-2*yy**2) + F1star*Un + beta - G12
	F11 = G11*dh**2


	print('PARAMETERS : OK')



	##################################
	# CONSTRUCTION OF THE B MATRIX
	##################################


	B11 = np.zeros((N*N, N*N))

			
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

			
			
	for i in range(N):
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


	if BC == 'activated':
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
		
	elif BC == '':
		print('NO BC\'S IMPLEMENTED')

	




	##################################
	# SOLVING
	##################################
	


	c, X = eig(A,B)
	c_NT, X_NT = eig(A11_star, B11)
	
	
	print('EIGENVALUES AND EIGENVECTORS : OK')
	
	
	#############################################"
	return x_l, y_l, xx, yy, c, c_NT, X, X_NT, Un, Thetabar
	
	
	
	
	
	
	

	
	
def compute_variables(N,ix_norm_c__, ix_norm_cNT__, c, c_NT, X, X_NT,timesteps, k0, l0, xx, yy, dh, Un, Thetabar):

	'''
	Function that computes the parameters zeta, us, vs
	'''

	
	# TQG ##################
	########################
	# Extract eigenvalue and eigenvector
	c_mode = c[ix_norm_c__]  # eigenvalue
	X_mode = X[:, ix_norm_c__]  # eigenvector

	# Extract Theta from second half of X
	phi_flat = X_mode[:N*N]
	phi_xy = phi_flat.reshape((N, N))
	# Normalize for visualization if needed
	PHI_xy = np.real(phi_xy)
	
	# Extract Theta from second half of X
	theta_flat = X_mode[N*N:]
	theta_xy = theta_flat.reshape((N, N))
	# Normalize for visualization if needed
	THETA_xy = np.real(theta_xy)






	# QG ##################
	########################
	# Extract eigenvalue and eigenvector
	c_NT_mode = c_NT[ix_norm_cNT__]  # eigenvalue
	X_NT_mode = X_NT[:, ix_norm_cNT__]  # eigenvector

	# Extract Theta from second half of X
	#Theta_flat = X_mode[N*N:]
	phi_flat_NT = X_NT_mode
	phi_xy_NT = phi_flat_NT.reshape((N, N))

	# Normalize for visualization if needed
	PHI_xy_NT = np.real(phi_xy_NT)



	zeta_list = []
	zeta_listNT = []
	theta_list = []
	

	for i, t in enumerate(timesteps):
		PHI_t = np.real(PHI_xy * np.exp(c_mode * t))
		THETA_t = np.real(THETA_xy * np.exp(c_mode * t))
		PHI_t_NT = np.real(PHI_xy_NT * np.exp(c_NT_mode * t))
		
		PSI = np.real(PHI_t* np.exp(1j*(k0*xx+l0*yy - c_mode*t)))
		THETA = np.real(THETA_t* np.exp(1j*(k0*xx+l0*yy - c_mode*t)))
		PSI_NT = np.real(PHI_t_NT* np.exp(1j*(k0*xx+l0*yy - c_NT_mode*t)))

		
		zeta, zeta_NT = np.zeros_like(PSI), np.zeros_like(PSI)

		PSI = PSI - Un*yy
		PSI_NT = PSI_NT - Un*yy

		THETA = THETA + Thetabar

		# loop to compute physical params

		for j in range(N-1):
			for k in range(N-1):
				
				
				zeta[j,k] = (PSI[j,k+1] -2*PSI[j,k] + PSI[j,k-1])/(dh**2) +\
				 (PSI[j+1,k] -2*PSI[j,k] + PSI[j-1,k])/(dh**2) 
				zeta_NT[j,k] = (PSI_NT[j,k+1] -2*PSI_NT[j,k] + PSI_NT[j,k-1])/(dh**2) +\
				 (PSI_NT[j+1,k] -2*PSI_NT[j,k] + PSI_NT[j-1,k])/(dh**2)
		
		# stack it into a list	 
				 
		zeta_list.append(zeta)
		
		#u_s_list.append(u_s)
		#v_s_list.append(v_s)
		
		zeta_listNT.append(zeta_NT)
		#u_s_listNT.append(u_sNT)
		#v_s_listNT.append(v_sNT)
		
		theta_list.append(THETA)
		



	
	# convert the final list into an array

	zeta_list = np.array(zeta_list)
	zeta_listNT = np.array(zeta_listNT)
	theta_list = np.array(theta_list)
	
	

	
	return zeta_list, zeta_listNT, theta_list 



