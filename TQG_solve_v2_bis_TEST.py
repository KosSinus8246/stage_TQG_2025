import os
#import warnings
import numpy as np                                         
import matplotlib as mpl
import scipy.linalg as spl
#from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from tqdm import tqdm

mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['legend.edgecolor'] = '0'
#warnings.filterwarnings("ignore")

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_2_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')



# cf TQG notes : A.X = c.B.X is the system that is solved here
# and also the non thermal system !!
# @uthor : dimitri moreau 15/05/2025


save_png = False
debug_mode = False
font_size = 17

name_exp = input('Name of the experience ?')

print('-----------------------------------------------------')

##################################
# VARIABLES, SPACE ...
##################################

Ny, Nk = 100, 100	



Ly, Lk = np.pi, 6
dy = Ly/Ny
y_l, k = np.linspace(0,Ly,Ny), np.linspace(0,Lk,Nk)
dk = Lk/Nk


beta = 0 #1e-11
#F1star = 0 #1/Rd**2
F1star = 0

U0 = 1
Theta0_U0 = 1 # ratio
Theta0 = Theta0_U0 *U0


Un = U0*np.exp(-y_l**2)
Vn = Un*(dy**2)
G12 = -2*y_l*Theta0*np.exp(-y_l**2) # dThetabar/dy





#Un = 1/(1+np.exp(-y_l)) # sigmoide
G11 = 2.0*Un*(1-2*y_l**2) + F1star*Un + beta - G12
F11 = G11*dy**2


if save_png == True:

	print('/////////////////////////////////////////////////////')


	# Create the full directory path
	folder_path = os.path.join("im_para", name_exp)
	os.makedirs(folder_path, exist_ok=True)  # Create directories if they don't exist

	# Create full file path
	file_path = os.path.join(folder_path, 'variables_used_' + name_exp + '.txt')

	# Open a file in write mode
	with open('im_para/'+name_exp+'/variables_used_'+name_exp+'.txt', 'w') as file:
	    file.write('Used variables for : '+name_exp+'\n')
	    file.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
	    file.write(f"Ny = {Ny}\n")
	    file.write(f"Nk = {Nk}\n")
	    file.write(f"Ly = {Ly}\n")
	    file.write(f"Lk = {Lk}\n")
	    file.write(f"F1star = {F1star}\n")
	    file.write(f"beta = {beta}\n")
	    #file.write(f"Rd = {Rd}\n")
	    file.write(f"U0 = {U0}\n")
	    file.write(f"Theta0 = {Theta0}\n")
	    file.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

	print('Variables stored into : variables_used_'+name_exp+'.txt')



print('/////////////////////////////////////////////////////')
print('PARAMS : OK')


val_c = []
val_cNT = []
c_matrix = np.zeros((len(k),2*Ny))
cNT_matrix = np.zeros((len(k),2*Ny))

for ik in tqdm(range(len(k))):


	K2 = (k[ik]**2 + F1star)*dy**2


	##################################
	# CONSTRUCTION OF THE B MATRIX
	##################################



	# Main diagonal
	main_diag = -(2 + K2) * np.ones(Ny)
	off_diag = np.ones(Ny-1)
	# Construct tridiagonal B11 using np.diag
	B11 = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
	# Construct other blocks
	B12 = np.zeros((Ny, Ny))
	B21 = np.zeros((Ny, Ny))
	B22 = np.eye(Ny, Ny)


	B = np.block([[B11,B12],[B21,B22]])

	'''
	# Combine them into full block matrix B
	top = np.concatenate((B11, B12), axis=1)
	bottom = np.concatenate((B21, B22), axis=1)
	B = np.concatenate((top, bottom), axis=0)

	top = np.concatenate((B11, B12), axis=0)
	bottom = np.concatenate((B21, B22), axis=0)
	B = np.concatenate((top, bottom), axis=1)'''


	##################################
	# CONSTRUCTION OF THE A MATRIX
	##################################



	# Block A11
	main_diag_A11 = -Un*(2 + K2) + F11  # shape (3,)
	A11 = np.zeros((Ny,Ny))
	np.fill_diagonal(A11, main_diag_A11)

	for i in range(Ny-1):
	    A11[i,i+1] = Un[i] # i initialement
	    A11[i+1,i] = Un[i]
	    

	# Block A12
	A12 = np.diag(-Vn)
	# Block A21
	A21 = np.diag(G12)
	# Block A22
	A22 = np.diag(Un)

	# Final block matrix A

	A = np.block([[A11,A12],[A21,A22]])

	'''
	top_A = np.concatenate((A11, A12), axis=1)
	bottom_A = np.concatenate((A21, A22), axis=1)
	A = np.concatenate((top_A, bottom_A), axis=0)


	top_A = np.concatenate((A11, A12), axis=0)
	bottom_A = np.concatenate((A21, A22), axis=0)
	A = np.concatenate((top_A, bottom_A), axis=1)'''


	# BC's
	
	# velocity odd
	A[0,1]=2.0*A[0,1]
	B[0,1]=2.0*B[0,1]

	
	

	A[Ny,Ny-1]=0.0
	B[Ny,Ny-1]=0.0
	'''
	
	
	# velocity not odd
	A[0,1]=0.0
	B[0,1]=0.0'''


	##################################
	# SOLUTION
	##################################
	# A.X = c.B.X 


	###### THERMAL SOLVING (TQG)

	c, _ = spl.eig(A,B)
	c_matrix[ik,:] = np.imag(c)
	
	#val_c.append(np.max(np.imag(c)))


	###### NON THERMAL SOLVING (QG)

	c_NT, _ = spl.eig(A11,B11)
	cNT_matrix[ik,:] = np.imag(c)
	
	#val_cNT.append(np.max(np.imag(c)))







val_c = np.array(val_c)
val_cNT = np.array(val_cNT)

'''
if np.max(np.imag(c_matrix[0,:Ny])) > np.max(np.imag(c_matrix[0,:Ny])):
	big_img_part_c = c_matrix[0,:Ny]
	print('111')
else:
	big_img_part_c = c_matrix[0,Ny:]
	print('222')'''



for j in range(len(k)):

	if np.max(np.imag(c_matrix[j,:Ny])) > np.max(np.imag(c_matrix[j,:Ny])):
		big_img_part_c = c_matrix[j,:Ny]
		print('111')
	else:
		big_img_part_c = c_matrix[j,Ny:]
		print('222')
		
	plt.plot(k[j],np.max(big_img_part_c[big_img_part_c>0]),'+')


##################################
# PLOT
##################################


print('/////////////////////////////////////////////////////')




plt.show()

print('END')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')




