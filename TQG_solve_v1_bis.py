import numpy as np
import seaborn as sns
import matplotlib as mpl
#import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['legend.edgecolor'] = '0'

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_1_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')



# cf TQG notes : A.X = c.B.X
# @uthor : dimitri moreau 29/04/2025



# if True : it will plot the 2 varibale phi and theta
# if False : it will only plot the part where Im(c) is important
choice_plot = False
nb_bins = 50
figsize_tuple = (15,6.5)
font_size = 17

name_exp = input('Name of the experience ?')


print('-----------------------------------------------------')

print('Preparing your experience : '+name_exp)

print('-----------------------------------------------------')
##################################
# VARIABLES, SPACE ...
##################################

Ny, Nk = 200, 200
Ly, Lk = np.pi, 40
dy = Ly/Ny
y_l, k = np.linspace(0.1,Ly,Ny), np.linspace(0.1,Lk,Nk)
dk = Lk/Nk


beta, Rd = 5, 1 #1e-11
F1star = 0 #1/Rd**2
K2 = (k**2 + F1star)*dy**2
#K2 = 0
U0= 1


phi, theta = np.zeros((Ny, Nk)), np.zeros((Ny, Nk))

Un = U0*np.exp(-y_l**2)
#Un = 1/(1+np.exp(-y_l)) # sigmoide


# V/G/Mn
Theta0 = 1
Vn = Un * dy**2
G12 = -2*y_l*Theta0*np.exp(-y_l**2) # dThetabar/dy

#G12 = np.zeros_like(y_l) # ondes de rossby (stable)

G11 = 2.0*Un*(1-2*y_l**2) + F1star*Un+beta - G12
F11 = G11*dy**2



print('/////////////////////////////////////////////////////')
print('PARAMS : OK')



##################################
# CONSTRUCTION OF THE B MATRIX
##################################



# Diagonale principale
main_diag = -(2 + K2) * np.ones(Ny)
off_diag = np.ones(Ny-1)
# Construct tridiagonal B11 using np.diag
B11 = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
# Construct other blocks
B12 = np.zeros((Ny, Ny))
B21 = np.zeros((Ny, Ny))
B22 = np.eye(Ny, Ny)
# Combine them into full block matrix B
top = np.concatenate((B11, B12), axis=1)
bottom = np.concatenate((B21, B22), axis=1)
B = np.concatenate((top, bottom), axis=0)
print('MATRIX B : OK')



##################################
# CONSTRUCTION OF THE A MATRIX
##################################



# Block A11
main_diag_A11 = (-Un*(2 + K2) + F11)  # shape (3,)
A11 = np.zeros((Ny,Ny))
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



##################################
# PLOT
##################################


print('COMPUTATION : OK')
print('/////////////////////////////////////////////////////')
print('PLOT...')




#################### for only the 2 values phi and theta

if choice_plot == True:

	choice_plot_name = '_phi_n_theta'
	print('-----------------------------------------------------')
	print('You have chosen the **'+choice_plot_name+'** plot')

	sigma1 = k*np.imag(c[:Ny])
	sigma2 = k*np.imag(c[Ny:])
	
	
	
	##################################
	# Plot 1



	fig, ax = plt.subplots(1, 2, figsize=figsize_tuple)
	
	fig.suptitle('Experience : '+name_exp)


	ax[0].plot(Un,y_l,'b')
	ax[0].tick_params(right=True, top=True,size=4,width=1,direction='in')
	ax[0].spines[['top','bottom','right','left']].set_linewidth(2)
	ax[0].set_ylabel(r'$y$',size=font_size)
	ax[0].set_xlabel(r'$\overline{U}$',size=font_size)
	ax[0].set_title('Velocity profile',size=font_size)
	ax[0].axhline(0, color='gray', linestyle=':')
	ax[0].axvline(0, color='gray', linestyle=':')



	print('-----------------------------------------------------')


	borne = (1/4)*(Un*y_l)**2
	test_crit = Un * G12

	print('Stability analysis : ')
	if (test_crit < borne).all() == True:
		print('Totally unstable')
	else:
		print('Stable ?')
	print('-----------------------------------------------------')


	ax[1].axhline(0, color='gray', linestyle=':')
	ax[1].axvline(0, color='gray', linestyle=':')
	ax[1].plot(y_l,borne,'k--',label=r'Bound : $\frac{1}{4}.(\overline{U}.y)^2$')
	ax[1].plot(y_l,test_crit,'orange',label=r'$\overline{U}.\frac{\mathrm{d}\Theta}{\mathrm{d}y}$')
	ax[1].fill_between(y_l,borne,test_crit,color='orange',alpha=0.3)
	ax[1].tick_params(left=True,right=True,top=True,bottom=True,direction='in',size=4,width=1)
	ax[1].set_xlabel(r'$y$',size=font_size)
	ax[1].set_ylabel(r'Value proportional to $\overline{U}.y^2$',size=font_size)
	ax[1].legend(loc='best',fancybox=False)
	ax[1].set_title('Stability',size=font_size)

	# Make the axes (spines) bold
	for spine in ax[1].spines.values():
	    spine.set_linewidth(2)


	plt.tight_layout()

	plt.savefig('img/fig1_'+name_exp+choice_plot_name+'.png',dpi=300)

	##################################
	# Plot 2



	fig, (ax) = plt.subplots(2,2,figsize=(15,10))
	
	fig.suptitle('Experience : '+name_exp)

	omega_phi = c[:Ny]*k
	omega_theta = c[Ny:]*k


	ax[0,0].axhline(0, color='gray', linestyle=':')
	ax[0,0].axvline(0, color='gray', linestyle=':')
	ax[0,0].plot(k,sigma1,'b:',alpha=0.15)
	ax[0,0].scatter(k, sigma1, marker='o', color='b', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	ax[0,0].axhline(0, color='gray', linestyle=':')
	ax[0,0].axvline(0, color='gray', linestyle=':')

	axbis = ax[0,0].twinx()
	axbis.plot(k,sigma2,'r:',alpha=0.1)
	axbis.scatter(k, sigma2, marker='^', color='r', edgecolor='k', alpha=0.6,s=50, label=r'$\Theta$')

	ax[0,0].set_xscale('log')

	# Axis colors
	ax[0,0].set_ylabel(r'$\sigma_\phi$', color='blue',size=font_size)
	ax[0,0].tick_params(axis='y', colors='blue',direction='in',size=4,width=1)
	ax[0,0].spines['left'].set_color('blue')
	ax[0,0].spines['left'].set_linewidth(2)

	axbis.set_ylabel(r'$\sigma_\Theta$', color='red',size=font_size)
	axbis.tick_params(axis='y', colors='red',direction='in',size=4,width=1)
	axbis.spines['right'].set_color('red')
	axbis.spines['right'].set_linewidth(2)

	ax[0,0].spines['bottom'].set_linewidth(2)
	ax[0,0].spines['top'].set_linewidth(2)

	# Common elements
	ax[0,0].set_xlabel(r'$k$',size=font_size)
	ax[0,0].set_title(r'$\sigma = \mathbf{Im}\{c\}.k$',size=font_size)
	ax[0,0].tick_params(top=True,direction='in', size=4, width=1)

	# Combine legends
	handles1, labels1 = ax[0,0].get_legend_handles_labels()
	handles2, labels2 = axbis.get_legend_handles_labels()
	ax[0,0].legend(handles1 + handles2, labels1 + labels2, loc='best',fancybox=False)

	if np.abs(np.max(sigma1)) - np.abs(np.min(sigma1)) < 0:
		ax[0,0].set_ylim(-np.abs(np.min(sigma1)), np.abs(np.min(sigma1)))
	else:
		ax[0,0].set_ylim(-np.abs(np.max(sigma1)), np.abs(np.max(sigma1)))
		
		
	if np.abs(np.max(sigma2)) - np.abs(np.min(sigma2)) < 0:
		axbis.set_ylim(-np.abs(np.min(sigma2)), np.abs(np.min(sigma2)))
	else:
		axbis.set_ylim(-np.abs(np.max(sigma2)), np.abs(np.max(sigma2)))


	ax[1,0].plot(Un,y_l,'b')
	ax[1,0].tick_params(right=True, top=True,size=4,width=1,direction='in')
	ax[1,0].spines[['top','bottom','right','left']].set_linewidth(2)
	ax[1,0].set_ylabel(r'$y$',size=font_size)
	ax[1,0].set_xlabel(r'$\overline{U}$',size=font_size)
	ax[1,0].set_title('Velocity profile',size=font_size)
	ax[1,0].axhline(0, color='gray', linestyle=':')
	ax[1,0].axvline(0, color='gray', linestyle=':')




	ax[0,1].scatter(np.real(omega_phi),np.imag(omega_phi),color='b',marker='*',s=50,alpha=0.6,edgecolor='k')
	ax[0,1].set_xlabel(r'$\mathbf{Re}\{\omega_\phi\}$',size=font_size)
	ax[0,1].set_ylabel(r'$\mathbf{Im}\{\omega_\phi\}$',size=font_size)
	ax[0,1].tick_params(right=True,top=True,direction='in',size=4,width=1)
	ax[0,1].axhline(0, color='gray', linestyle=':')
	ax[0,1].axvline(0, color='gray', linestyle=':')
	ax[0,1].set_title(r'Eigenfrequencies $\omega_\phi = c_\phi.k$')
	# Make the axes (spines) bold
	for spine in ax[0,1].spines.values():
	    spine.set_linewidth(2)


	ax[1,1].scatter(np.real(omega_theta),np.imag(omega_theta),color='r',marker='s',s=50,alpha=0.6,edgecolor='k')
	ax[1,1].set_ylabel(r'$\mathbf{Im}\{\omega_\Theta\}$',size=font_size)
	ax[1,1].set_xlabel(r'$\mathbf{Re}\{\omega_\Theta\}$',size=font_size)
	ax[1,1].tick_params(right=True,top=True,direction='in',size=4,width=1)
	ax[1,1].axhline(0, color='gray', linestyle=':')
	ax[1,1].axvline(0, color='gray', linestyle=':')
	ax[1,1].set_title(r'Eigenfrequencies $\omega_\Theta = c_\Theta.k$')
	# Make the axes (spines) bold
	for spine in ax[1,1].spines.values():
	    spine.set_linewidth(2)



	plt.tight_layout()


	plt.savefig('img/fig2_'+name_exp+choice_plot_name+'.png',dpi=300)






	##################################
	# Plot 3


	# derivative of sigma

	dsigma1, dsigma2 = np.zeros_like(sigma1), np.zeros_like(sigma2)

	for i in range(len(k)-1):
		dsigma1[i] = (sigma1[i+1] - sigma1[i-1])/(2*dk)
		dsigma2[i] = (sigma2[i+1] - sigma2[i-1])/(2*dk)




	fig, ax = plt.subplots(1, 2, figsize=figsize_tuple)
	
	fig.suptitle('Experience : '+name_exp)

	sns.histplot(sigma1,bins=nb_bins,ax=ax[0],kde=True,stat='percent',color='b',label=r'$\phi$')
	#ax[0].set_ylim(0,40)
	ax1 = ax[0].twiny()
	sns.histplot(sigma2,bins=nb_bins,ax=ax1,kde=True,stat='percent',color='r',label=r'$\Theta$')
	#ax[0].set_ylim(0,40)
	ax[0].set_xlabel(r'$\sigma_\phi$',color='blue',size=font_size)
	ax1.set_xlabel(r'$\sigma_\Theta$',color='red',size=font_size)

	ax[0].set_ylim(0,100)
	ax1.set_ylim(0,100)


	ax[0].set_ylabel('Percent %')

	# Axis colors
	ax[0].tick_params(axis='x', colors='blue',direction='in',size=4,width=1)
	ax[0].tick_params(right=True,direction='in',size=4,width=1)
	ax[0].spines['bottom'].set_color('blue')
	ax[0].spines['bottom'].set_linewidth(2)
	ax[0].axhline(0, color='gray', linestyle=':')
	ax[0].axvline(0, color='gray', linestyle=':')

	ax1.tick_params(axis='x', colors='red',direction='in',size=4,width=1)
	ax1.spines['top'].set_color('red')
	ax1.spines['top'].set_linewidth(2)

	ax[0].spines[['left','right']].set_linewidth(2)

	# Combine legends
	handles1, labels1 = ax[0].get_legend_handles_labels()
	handles2, labels2 = ax1.get_legend_handles_labels()
	ax[0].legend(handles1 + handles2, labels1 + labels2, loc='best',fancybox=False,title='Values for')




	if np.abs(np.max(sigma1)) - np.abs(np.min(sigma1)) < 0:
		ax[0].set_xlim(-np.abs(np.min(sigma1)), np.abs(np.min(sigma1)))
	else:
		ax[0].set_xlim(-np.abs(np.max(sigma1)), np.abs(np.max(sigma1)))
		
		
	if np.abs(np.max(sigma2)) - np.abs(np.min(sigma2)) < 0:
		ax1.set_xlim(-np.abs(np.min(sigma2)), np.abs(np.min(sigma2)))
	else:
		ax1.set_xlim(-np.abs(np.max(sigma2)), np.abs(np.max(sigma2)))









	sns.histplot(dsigma1,bins=nb_bins,ax=ax[1],kde=True,stat='percent',color='b',label=r'$\phi$')
	ax[1].set_xlabel(r'$\partial \sigma_\phi/\partial k$',color='blue',size=font_size)
	ax2 = ax[1].twiny()
	sns.histplot(dsigma2,bins=nb_bins,ax=ax2,kde=True,stat='percent',color='r',label=r'$\Theta$')
	ax[1].set_ylabel('')
	ax2.set_xlabel(r'$\partial \sigma_\Theta/\partial k$',color='red',size=font_size)



	# Axis colors
	ax[1].tick_params(axis='x', colors='blue',direction='in',size=4,width=1)
	ax[1].tick_params(labelleft=False,right=True,direction='in',size=4,width=1)
	ax[1].spines['bottom'].set_color('blue')
	ax[1].spines['bottom'].set_linewidth(2)

	ax2.tick_params(axis='x', colors='red',direction='in',size=4,width=1)
	ax2.spines['top'].set_color('red')
	ax2.spines['top'].set_linewidth(2)

	ax[1].set_ylim(0,100)
	ax2.set_ylim(0,100)


	ax[1].spines[['left','right']].set_linewidth(2)
	ax[1].axhline(0, color='gray', linestyle=':')
	ax[1].axvline(0, color='gray', linestyle=':')


	# Combine legends
	handles1, labels1 = ax[1].get_legend_handles_labels()
	handles2, labels2 = ax2.get_legend_handles_labels()
	ax[1].legend(handles1 + handles2, labels1 + labels2, loc='best',fancybox=False,title='Values for')



	if np.abs(np.max(dsigma1)) - np.abs(np.min(dsigma1)) < 0:
		ax[1].set_xlim(-np.abs(np.min(dsigma1)), np.abs(np.min(dsigma1)))
	else:
		ax[1].set_xlim(-np.abs(np.max(dsigma1)), np.abs(np.max(dsigma1)))
		
		
	if np.abs(np.max(dsigma2)) - np.abs(np.min(dsigma2)) < 0:
		ax2.set_xlim(-np.abs(np.min(dsigma2)), np.abs(np.min(dsigma2)))
	else:
		ax2.set_xlim(-np.abs(np.max(dsigma2)), np.abs(np.max(dsigma2)))



	plt.tight_layout()
	plt.savefig('img/fig3_'+name_exp+choice_plot_name+'.png',dpi=300)
	





#################### for only the 1 imaginary part
else:

	choice_plot_name = '_max_imag'
	print('-----------------------------------------------------')
	print('You have chosen the **'+choice_plot_name+'** plot')
	###############################
	# prendre la partie im de c la plus importante

	if np.max(np.imag(c[:Ny])) > np.max(np.imag(c[Ny:])):
		big_img_part_c = c[:Ny]
	else:
		big_img_part_c = c[Ny:]
		
	
	
	sigma_big_img = k*np.imag(big_img_part_c)
	omega_big_c = big_img_part_c*k

	##################################
	# Plot 1



	fig, (ax) = plt.subplots(2,2,figsize=(15,10))
	
	
	fig.suptitle('Experience : '+name_exp)


	ax[0,0].axhline(0, color='gray', linestyle=':')
	ax[0,0].axvline(0, color='gray', linestyle=':')
	ax[0,0].plot(k,sigma_big_img,'b:',alpha=0.15)
	ax[0,0].scatter(k, sigma_big_img, marker='o', color='b', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	ax[0,0].axhline(0, color='gray', linestyle=':')
	ax[0,0].axvline(0, color='gray', linestyle=':')
	
	for spine in ax[0,0].spines.values():
	    spine.set_linewidth(2)


	ax[0,0].set_xscale('log')

	# Common elements
	ax[0,0].set_xlabel(r'$k$',size=font_size)
	ax[0,0].set_ylabel(r'$\sigma$',size=font_size)
	ax[0,0].set_title(r'$\sigma = \mathbf{Im}\{c\}.k$',size=font_size)
	ax[0,0].tick_params(top=True,right=True,direction='in', size=4, width=1)

	if np.abs(np.max(sigma_big_img)) - np.abs(np.min(sigma_big_img)) < 0:
		ax[0,0].set_ylim(-np.abs(np.min(sigma_big_img)), np.abs(np.min(sigma_big_img)))
	else:
		ax[0,0].set_ylim(-np.abs(np.max(sigma_big_img)), np.abs(np.max(sigma_big_img)))
	

	
	ax[1,0].plot(Un,y_l,'b')
	ax[1,0].tick_params(right=True, top=True,size=4,width=1,direction='in')
	ax[1,0].spines[['top','bottom','right','left']].set_linewidth(2)
	ax[1,0].set_ylabel(r'$y$',size=font_size)
	ax[1,0].set_xlabel(r'$\overline{U}$',size=font_size)
	ax[1,0].set_title('Velocity profile',size=font_size)
	ax[1,0].axhline(0, color='gray', linestyle=':')
	ax[1,0].axvline(0, color='gray', linestyle=':')




	ax[0,1].scatter(np.real(omega_big_c),np.imag(omega_big_c),color='b',marker='*',s=50,alpha=0.6,edgecolor='k')
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

	print('Stability analysis : ')
	if (test_crit < borne).all() == True:
		print('Totally unstable')
	else:
		print('Stable ?')
	print('-----------------------------------------------------')


	ax[1,1].axhline(0, color='gray', linestyle=':')
	ax[1,1].axvline(0, color='gray', linestyle=':')
	ax[1,1].plot(y_l,borne,'k--',label=r'Bound : $\frac{1}{4}.(\overline{U}.y)^2$')
	ax[1,1].plot(y_l,test_crit,'orange',label=r'$\overline{U}.\frac{\mathrm{d}\Theta}{\mathrm{d}y}$')
	ax[1,1].fill_between(y_l,borne,test_crit,color='orange',alpha=0.3)
	ax[1,1].tick_params(left=True,right=True,top=True,bottom=True,direction='in',size=4,width=1)
	ax[1,1].set_xlabel(r'$y$',size=font_size)
	ax[1,1].set_ylabel(r'Value proportional to $\overline{U}.y^2$',size=font_size)
	ax[1,1].legend(loc='best',fancybox=False)
	ax[1,1].set_title('Stability',size=font_size)

	# Make the axes (spines) bold
	for spine in ax[1,1].spines.values():
	    spine.set_linewidth(2)



	plt.tight_layout()


	plt.savefig('img/fig1_'+name_exp+choice_plot_name+'.png',dpi=300)

	
	##################################
	# Plot 2
	
	
	
	# derivative of sigma

	dsigma = np.zeros_like(sigma_big_img)

	for i in range(len(k)-1):
		dsigma[i] = (sigma_big_img[i+1] - sigma_big_img[i-1])/(2*dk)




	fig, ax = plt.subplots(1, 2, figsize=figsize_tuple)
	
	
	fig.suptitle('Experience : '+name_exp)
	

	sns.histplot(sigma_big_img,bins=nb_bins,ax=ax[0],kde=True,stat='percent',color='b')
	ax[0].set_xlabel(r'$\sigma$',size=font_size)
	ax[0].set_ylim(0,100)
	ax[0].set_ylabel('Percent %')
	ax[0].spines[['left','right','top','bottom']].set_linewidth(2)
	ax[0].tick_params(right=True,top=True,direction='in',size=4,width=1)
	ax[0].axhline(0, color='gray', linestyle=':')
	ax[0].axvline(0, color='gray', linestyle=':')




	if np.abs(np.max(sigma_big_img)) - np.abs(np.min(sigma_big_img)) < 0:
		ax[0].set_xlim(-np.abs(np.min(sigma_big_img)), np.abs(np.min(sigma_big_img)))
	else:
		ax[0].set_xlim(-np.abs(np.max(sigma_big_img)), np.abs(np.max(sigma_big_img)))



	sns.histplot(dsigma,bins=nb_bins,ax=ax[1],kde=True,stat='percent',color='b')
	ax[1].set_xlabel(r'$\partial \sigma/\partial k$',size=font_size)


	ax[1].set_ylim(0,100)

	ax[1].spines[['left','right','top','bottom']].set_linewidth(2)
	ax[1].axhline(0, color='gray', linestyle=':')
	ax[1].axvline(0, color='gray', linestyle=':')
	ax[1].tick_params(labelleft=False,right=True,top=True,direction='in',size=4,width=1)
	ax[1].set_ylabel('')



	if np.abs(np.max(dsigma)) - np.abs(np.min(dsigma)) < 0:
		ax[1].set_xlim(-np.abs(np.min(dsigma)), np.abs(np.min(dsigma)))
	else:
		ax[1].set_xlim(-np.abs(np.max(dsigma)), np.abs(np.max(dsigma)))



	plt.tight_layout()
	plt.savefig('img/fig2_'+name_exp+choice_plot_name+'.png',dpi=300)
	
	
	
	
	# temporaire

	fig, (ax) = plt.subplots(2,2,figsize=(15,10))



	ax[0,0].scatter(k, sigma_big_img, marker='o', color='b', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	ax[0,0].set_xscale('log')

	ax[0,1].scatter(np.real(omega_big_c),np.imag(omega_big_c),color='b',marker='*',s=50,alpha=0.6,edgecolor='k')






	# pour k,sigma_big_img
	Nfourier = len(sigma_big_img)
	kmax = k[-1] - k[0]
	Fs = Nfourier/kmax
	fourier11 = np.fft.fft(sigma_big_img)/(Nfourier/2)
	freq11 = np.fft.fftfreq(Nfourier,1/Fs)


	ax[1,0].plot(freq11[0:Nfourier//2],np.abs(fourier11[0:Nfourier//2]))

	# pour  np.real(omega_big_c),np.imag(omega_big_c)
	Nfourier = len(omega_big_c)
	x_max = np.real(omega_big_c)[-1] - np.real(omega_big_c)[0]
	Fs = Nfourier/x_max
	fourier12 = np.fft.fft(np.imag(omega_big_c))/(Nfourier/2)
	freq12 = np.fft.fftfreq(Nfourier,1/Fs)


	ax[1,1].plot(freq12[0:Nfourier//2],np.abs(fourier12[0:Nfourier//2]))







plt.show()


print('END')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')






