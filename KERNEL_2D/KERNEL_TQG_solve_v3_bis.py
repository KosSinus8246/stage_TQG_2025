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


def get_ix(c, c_NT, crit):

	'''
	Function that select the most important modes following
	the desired criteria
	'''

	if crit == 'imag':
		norm_cNT = (np.imag(c_NT)**2)**0.5
		norm_c = (np.imag(c)**2)**0.5
	
	elif crit == 'imag_real':
		norm_cNT = (np.real(c_NT)**2 + np.imag(c_NT)**2)**0.5
		norm_c = (np.real(c)**2 + np.imag(c)**2)**0.5

	elif crit == 'real':
		norm_cNT = (np.real(c_NT)**2)**0.5
		norm_c = (np.real(c)**2)**0.5


	norm_cNT__ = np.sort(norm_cNT)[::-1]
	ix_norm_cNT__ = np.argsort(norm_cNT)[::-1]

	norm_c__ = np.sort(norm_c)[::-1]
	ix_norm_c__ = np.argsort(norm_c)[::-1]

	print('-----------------------------------------------------')
	print('The plot shows you the available modes : close this')
	print('plot and choose a number of mode to sum whenever you')
	print('are ready !')
	print('')
	print('Warning : you can\'t enter a number of mode over the')
	print('smallest number of eigenvalues (i.e. the QG one)')
	print('-----------------------------------------------------')


	fig, (ax) = plt.subplots(1,1)

	# [2:] to remove the inf of begin and end because they are sorted

	max_ = (1/100)*np.max(np.sort((np.real(c)**2 + np.imag(c)**2)**0.5))
	max__ = (1/100)*np.max(np.sort((np.real(c_NT)**2 + np.imag(c_NT)**2)**0.5))

	ax.plot(np.sort((np.real(c)**2 + np.imag(c)**2)**0.5)[::-1]/max_,'k--',label='TQG')
	ax.plot(np.sort((np.real(c_NT)**2 + np.imag(c_NT)**2)**0.5)[::-1]/max__,'k-',label='QG')


	ax.set_xlabel('Number of modes summed', fontweight="bold")
	ax.set_ylabel(r'% contribution', fontweight="bold")
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

	plt.show()


	return ix_norm_c__, ix_norm_cNT__





def compute_TQG_2D(N, Lmin, L, beta, F1star, U0, Theta0_U0, k0, dh, BC, Lstar, std):

	'''
	Function that computes eigenvalues and eigenvectors
	'''


	x_l, y_l = np.linspace(Lmin,L,N), np.linspace(Lmin,L,N)
	xx, yy = np.meshgrid(x_l,y_l)

	Theta0 = Theta0_U0 *U0

	Un = U0*np.exp(-(yy-L/2)**2)
	Vn = Un*(dh**2)
	K2 = (k0**2 + F1star)*dh**2



	if Lstar == 0.:
		print('CONF : NO LSTAR')
		G12 = -2*yy*Theta0*np.exp(-yy**2) # dThetabar/dy
	else:
		print('CONF : LSTAR-EFFECT')
		G12 = (-2/Lstar**2)*yy*Theta0*np.exp(-(y_l**2)/(Lstar**2)) # dThetabar/dy

		if std == 0:
			print('NOISE : NONE')
			G12 = G12
			Un = Un
		#G12 = -(2/Ly**2)*y_l*Theta0*np.exp(-(y_l**2)/(Lstar**2)) # dThetabar/dy	
		else:
			print('NOISE : GAUSSIAN NOISE ; STD = ',std)
			noise = np.random.normal(0,std,len(G12))
			noise = np.abs(noise)/(np.max(noise))
			G12 = G12 + noise


			fig, (ax) = plt.subplots(1,1)
			#ax.hist(noise,color='skyblue',edgecolor='k',density=True)
			ax.plot(G12)
			ax.set_xlabel('noise',fontweight='bold')
			ax.set_ylabel('%',fontweight='bold')
			ax.tick_params(top=True, right=True, direction='in', length=4, width=1)
			for spine in ax.spines.values():
				spine.set_linewidth(2)
			for tick in ax.get_yticklabels() + ax.get_xticklabels():
				tick.set_fontweight('bold')





	Thetabar = Theta0 * np.exp(-yy**2)


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


	print('COMPUTING 1/2 ...')
	c, X = eig(A,B)
	print('COMPUTING 2/2 ...')
	c_NT, X_NT = eig(A11_star, B11)


	print('EIGENVALUES AND EIGENVECTORS : OK')


	#############################################"
	return x_l, y_l, xx, yy, c, c_NT, X, X_NT, Un, Thetabar










def compute_variables_prime(N,ix_norm_c__, ix_norm_cNT__, c, c_NT, X, X_NT,timesteps, k0, xx, yy, dh, Un, Thetabar,epsilon):

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
	psi_list = []
	psi_listNT = []


	for i, t in enumerate(timesteps):
		PHI_t = np.real(PHI_xy * np.exp(c_mode * t))
		THETA_t = np.real(THETA_xy * np.exp(c_mode * t))
		PHI_t_NT = np.real(PHI_xy_NT * np.exp(c_NT_mode * t))

		PSI = np.real(PHI_t * np.exp(-1j*(k0*xx - c_mode*t)))
		#PSI = PHI_t


		THETA = np.real(THETA_t* np.exp(1j*(k0*xx - c_mode*t)))
		#THETA = np.real(THETA_t)

		PSI_NT = np.real(PHI_t_NT * np.exp(-1j*(k0*xx - c_NT_mode*t)))
		#PSI_NT = PHI_t_NT

	
		#zeta, zeta_NT = np.zeros_like(PSI), np.zeros_like(PSI)

		theta_list.append(THETA)


		psi_list.append(PSI)
		psi_listNT.append(PSI_NT)


	theta_list = np.array(theta_list)

	psi_list = np.array(psi_list)*epsilon
	psi_listNT = np.array(psi_listNT)*epsilon



	return theta_list, psi_list, psi_listNT



def compute_zeta(N,Ntime,PSI, PSI_NT,dh):
	zeta = np.zeros_like(PSI)
	zeta_NT = np.zeros_like(PSI_NT)
	
	zeta_list = []
	zetaNT_list = []

	for i in range(Ntime):
		for j in range(1,N-1):
			for k in range(1,N-1):


				zeta[i,j,k] = (PSI[i,j,k+1] -2*PSI[i,j,k] + PSI[i,j,k-1])/(dh**2) +\
				(PSI[i,j+1,k] -2*PSI[i,j,k] + PSI[i,j-1,k])/(dh**2) 
				zeta_NT[i,j,k] = (PSI_NT[i,j,k+1] -2*PSI_NT[i,j,k] + PSI_NT[i,j,k-1])/(dh**2) +\
				(PSI_NT[i,j+1,k] -2*PSI_NT[i,j,k] + PSI_NT[i,j-1,k])/(dh**2)

		# stack it into a list

	zeta_list.append(zeta)
	zetaNT_list.append(zeta_NT)
	
	ZETA = np.array(zeta_list)
	ZETATN = np.array(zetaNT_list)
	
	return ZETA, ZETATN






def spatial_fourier_decomposition(psi, x, k_vals, dx):
	"""
	psi: numpy array of shape (T, X, Y)
	x: 1D numpy array of x positions (length X)
	k_vals: 1D array of wave numbers (length K)

	Returns:
	A_k: shape (K, T, Y)
	phi_k: shape (K, T, Y)
	A_bar_k: shape (K, T)
	phi_bar_k: shape (K, T)
	"""
	T, X, Y = psi.shape
	K = len(k_vals)

	# Expand x for broadcasting
	x = x.reshape(1, X, 1)  # shape (1, X, 1)
	

	# Result arrays
	c_k = np.zeros((K, T, Y))
	s_k = np.zeros((K, T, Y))

	for i, k in enumerate(k_vals):
		cos_kx = np.cos(k * x)  # shape (1, X, 1)
		sin_kx = np.sin(k * x)

		# Element-wise multiply and sum over x
		c_k[i] = np.sum(psi * cos_kx, axis=1)*dx
		s_k[i] = np.sum(psi * sin_kx, axis=1)*dx

	# Amplitude and phase
	A_k = np.sqrt(c_k**2 + s_k**2)
	#phi_k = np.arctan2(s_k, c_k)
	phi_k = np.arctan(s_k/c_k)

	# Integration over y (average for discrete grid)
	A_bar_k = np.mean(A_k, axis=2)
	phi_bar_k = np.mean(phi_k, axis=2)

	return A_k, phi_k, A_bar_k, phi_bar_k
