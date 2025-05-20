import os
import numpy as np                                        
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.linalg import eig

mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['legend.edgecolor'] = '0'

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_2_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')



# cf TQG notes : A.X = c.B.X is the system that is solved here
# and also the non thermal system
# @uthor : dimitri moreau 20/05/2025


save_png = True  # create a folder im_para and a folder per
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
dy = (Ly - ymin)/Ny

y_l, k = np.linspace(ymin,Ly,Ny), np.arange(kmin,Nk*dk,dk)


beta = 0 
F1star = 0 # 1/Rd**2

U0 = 1
Theta0_U0 = 0.1 # ratio
Theta0 = Theta0_U0 *U0


Un = U0*np.exp(-y_l**2)
#Un = 1/(1+np.exp(-y_l)) # sigmoide

Vn = Un*(dy**2)
G12 = -2*y_l*Theta0*np.exp(-y_l**2) # dThetabar/dy


G11 = 2.0*Un*(1-2*y_l**2) + F1star*Un + beta - G12
F11 = G11*dy**2

print('/////////////////////////////////////////////////////')
print('PARAMS : OK')


# Save all parameters in a txt file

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
	    file.write(f"ymin = {ymin}\n")
	    file.write(f"Ly = {Ly}\n")
	    file.write(f"kmin = {kmin}\n")
	    file.write(f"Lk = {Lk}\n")
	    file.write(f"dy = {dy}\n")
	    file.write(f"dk = {dk}\n")
	    file.write(f"F1star = {F1star}\n")
	    file.write(f"beta = {beta}\n")
	    #file.write(f"Rd = {Rd}\n")
	    file.write(f"U0 = {U0}\n")
	    file.write(f"Theta0 = {Theta0}\n")
	    file.write(f"ratio_Theta0_U0 = {Theta0_U0}\n")
	    file.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

	print('Variables stored into : variables_used_'+name_exp+'.txt')




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
for spine in ax.spines.values():
    spine.set_linewidth(2)


##################################
# Plot 1

'''

fig, (ax) = plt.subplots(2,2,figsize=(15,10))


fig.suptitle('Experience : '+name_exp)


ax[0,0].axhline(0, color='gray', linestyle=':')
ax[0,0].axvline(0, color='gray', linestyle=':')
ax[0,0].tick_params(top=True,right=True,direction='in',size=4,width=1)

ax[0,0].plot(k, val_c, 'k--', label='TQG')
ax[0,0].plot(k, val_cNT, 'k-', label='QG')


ax[0,0].legend(fancybox=False,loc='upper left')




for spine in ax[0,0].spines.values():
    spine.set_linewidth(2)
    
ax[0,0].set_xlabel(r'$k$',size=font_size)
ax[0,0].set_ylabel(r'$\sigma_\mathbf{Im} = \mathbf{Im}\{c\}.k ~\geq~ 0$',size=font_size)
ax[0,0].set_title(r'Growth rates',size=font_size)








ax[1,0].plot(Un,y_l,'b')
ax[1,0].tick_params(right=True, top=True,size=4,width=1,direction='in')
ax[1,0].spines[['top','bottom','right','left']].set_linewidth(2)
ax[1,0].set_ylabel(r'$y$',size=font_size)
ax[1,0].set_xlabel(r'$\overline{U}$',size=font_size)
ax[1,0].set_title('Velocity profile',size=font_size)
ax[1,0].axhline(0, color='gray', linestyle=':')
ax[1,0].axvline(0, color='gray', linestyle=':')
ax[1,0].set_ylim(np.min(y_l), np.max(y_l))



ax[0,1].scatter(val_c_ree,val_c,color='b',marker='*',s=50,alpha=0.6,edgecolor='k',label='TQG')
ax[0,1].scatter(val_cNT_ree,val_cNT,color='b',marker='+',s=50,alpha=0.6,label='QG')
ax[0,1].legend()
ax[0,1].set_xlabel(r'$\mathbf{Re}\{\omega\}$',size=font_size)
ax[0,1].set_ylabel(r'$\mathbf{Im}\{\omega\}$',size=font_size)
ax[0,1].tick_params(right=True,top=True,direction='in',size=4,width=1)
ax[0,1].axhline(0, color='gray', linestyle=':')
ax[0,1].axvline(0, color='gray', linestyle=':')
ax[0,1].set_title(r'Eigenfrequencies $\omega = c.k$')
# Make the axes (spines) bold
for spine in ax[0,1].spines.values():
    spine.set_linewidth(2)
    
    
    
print('-----------------------------------------------------')


borne = (1/4)*(Un*y_l)**2
test_crit = Un * G12


ax[1,1].axhline(0, color='gray', linestyle=':')
ax[1,1].axvline(0, color='gray', linestyle=':')
ax[1,1].plot(borne,y_l,'k--',label=r'Bound : $\frac{1}{4}.(\overline{U}.y)^2$')
ax[1,1].plot(test_crit,y_l,'orange',label=r'$\overline{U}.\frac{\mathrm{d}\Theta}{\mathrm{d}y}$')
ax[1,1].fill_betweenx(y_l,borne,test_crit,color='orange',alpha=0.3)
ax[1,1].tick_params(left=True,right=True,top=True,bottom=True,labelleft=False,direction='in',size=4,width=1)
ax[1,1].set_xlabel(r'Value proportional to $\overline{U}.y^2$',size=font_size)
ax[1,1].legend(loc='best',fancybox=False)
ax[1,1].set_title('Stability',size=font_size)
ax[1,1].set_ylim(np.min(y_l), np.max(y_l))


# Make the axes (spines) bold
for spine in ax[1,1].spines.values():
    spine.set_linewidth(2)

'''

plt.tight_layout()


if save_png == True:
	plt.savefig('im_para/'+name_exp+'/fig1_'+name_exp+choice_plot_name+'.png',dpi=300)




# save outputs
np.savetxt('im_para/'+name_exp+'/sigma_TQG.txt', 
           np.column_stack([k, val_c, val_c_ree]),
           fmt='%.18e', 
           header='k	sigmaIm		sigmaRe')
           
np.savetxt('im_para/'+name_exp+'/sigma_QG.txt', 
           np.column_stack([k, val_cNT, val_cNT_ree]),
           fmt='%.18e', 
           header='k	sigmaIm		sigmaRe')





plt.show()

print('END')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')




