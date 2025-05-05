import os
import warnings
import numpy as np
import seaborn as sns                                         
import matplotlib as mpl
import scipy.linalg as spl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['legend.edgecolor'] = '0'
warnings.filterwarnings("ignore")

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_2_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')



# cf TQG notes : A.X = c.B.X is the system that is solved here
# and also the non thermal system !!
# @uthor : dimitri moreau 05/05/2025


save_png = False
partie_pos = True # pour n'affichier que la partie positive des k.c_i
nb_bins = 50
figsize_tuple = (15,6.5)
font_size = 17

name_exp = input('Name of the experience ?')

print('-----------------------------------------------------')

##################################
# VARIABLES, SPACE ...
##################################

Ny, Nk = 300, 300
Ly, Lk = np.pi, 100
dy = Ly/Ny
y_l, k = np.linspace(0.1,Ly,Ny), np.linspace(0.1,Lk,Nk)
dk = Lk/Nk


beta = 0 #1e-11
#F1star = 0 #1/Rd**2
F1star = 4
K2 = (k**2 + F1star)*dy**2
U0= 1


Un = U0*np.exp(-y_l**2)
#Un = 1/(1+np.exp(-y_l)) # sigmoide


# V/G/Mn
Theta0 = 1
Vn = Un * dy**2
G12 = -2*y_l*Theta0*np.exp(-y_l**2) # dThetabar/dy

#G12 = np.zeros_like(y_l) # ondes de rossby (stable)

G11 = 2.0*Un*(1-2*y_l**2) + F1star*Un+beta - G12
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

c, X = spl.eig(A,B)

print('THERMAL COMPUTATION : OK')

###### NON THERMAL SOLVING

c_NT, X_NT = spl.eig(A11,B11)

print('NON-THERMAL COMPUTATION : OK')



# THERMAL
#################### for only the 1 imaginary part 


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
sigma_big_ree = k*np.real(big_img_part_c) 
omega_big_c = big_img_part_c*k

# NON-THERMAL

sigma_img_NT = k*np.imag(c_NT)
sigma_ree_NT = k*np.real(c_NT) 
omega_NT = c_NT*k



##################################
# PLOT
##################################



print('/////////////////////////////////////////////////////')

'''
print('PLOT...')





##################################
# Plot 1



fig, (ax) = plt.subplots(2,2,figsize=(15,10))


fig.suptitle('Experience : '+name_exp)


ax[0,0].axhline(0, color='gray', linestyle=':')
ax[0,0].axvline(0, color='gray', linestyle=':')


if partie_pos==False:
	ax[0,0].plot(k,sigma_big_img,'b:',alpha=0.15)
	ax[0,0].scatter(k, sigma_big_img, marker='o', color='b', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	
	ax[0,0].plot(k,sigma_big_ree,'r:',alpha=0.15)
	ax[0,0].scatter(k, sigma_big_ree, marker='o', color='r', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	
	if np.abs(np.max(sigma_big_img)) - np.abs(np.min(sigma_big_img)) < 0:
		ax[0,0].set_ylim(-np.abs(np.min(sigma_big_img)), np.abs(np.min(sigma_big_img)))
	else:
		ax[0,0].set_ylim(-np.abs(np.max(sigma_big_img)), np.abs(np.max(sigma_big_img)))
	
		
else:
	ax[0,0].plot(k[sigma_big_img>=0],sigma_big_img[sigma_big_img>=0],'b:',alpha=0.15)
	ax[0,0].scatter(k[sigma_big_img>=0], sigma_big_img[sigma_big_img>=0], marker='o', color='b', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	
	ax1 = ax[0,0].twinx()
	
	ax1.plot(k[sigma_big_img>=0],sigma_big_ree[sigma_big_img>=0],'r:',alpha=0.15)
	ax1.scatter(k[sigma_big_img>=0], sigma_big_ree[sigma_big_img>=0], marker='o', color='r', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	ax1.tick_params(right=True,direction='in', size=4, width=1,color='red',labelcolor='red')
	ax1.set_ylabel(r'$\sigma_\mathbf{Re} = \mathbf{Re}\{c\}.k ~\geq~ 0$',size=font_size,color='red')
	
	ax1.spines['right'].set_color('red')                         # spine
	ax1.yaxis.label.set_color('red')
	ax1.spines['right'].set_linewidth(2.25)  # Adjust thickness here   
	
	ax1.set_ylim(0,np.max(sigma_big_ree[sigma_big_img>=0]))
	
	
	if np.abs(np.max(sigma_big_img)) - np.abs(np.min(sigma_big_img)) < 0:
		ax[0,0].set_ylim(0, np.abs(np.min(sigma_big_img)))
	else:
		ax[0,0].set_ylim(0, np.abs(np.max(sigma_big_img)))
	
ax[0,0].axhline(0, color='gray', linestyle=':')
ax[0,0].axvline(0, color='gray', linestyle=':')

for spine in ax[0,0].spines.values():
    spine.set_linewidth(2)


#ax[0,0].set_xscale('log')

# Common elements
ax[0,0].set_xlabel(r'$k$',size=font_size)
ax[0,0].set_ylabel(r'$\sigma_\mathbf{Im} = \mathbf{Im}\{c\}.k ~\geq~ 0$',size=font_size,color='blue')
ax[0,0].set_title(r'$\sigma_\mathbf{Im}$ and $\sigma_\mathbf{Re}$',size=font_size)
ax[0,0].tick_params(top=True,right=False,direction='in', size=4, width=1,color='blue',labelcolor='blue')
ax[0,0].spines['left'].set_color('blue')                         # spine
ax[0,0].yaxis.label.set_color('blue')
ax[0,0].spines['left'].set_linewidth(2.25)  # Adjust thickness here      
ax[0,0].tick_params(axis='x', colors='black', direction='in', size=4, width=1)





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


if save_png == True:
	plt.savefig('im_para/'+name_exp+'/fig1_'+name_exp+choice_plot_name+'.png',dpi=300)


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

if save_png == True:
	plt.savefig('im_para/'+name_exp+'/fig2_'+name_exp+choice_plot_name+'.png',dpi=300)




##################################
# Plot 3

fig, (ax) = plt.subplots(2,2,figsize=(15,10))

len_fft = 5000


fig.suptitle('Experience : '+name_exp)


ax[0,0].axhline(0, color='gray', linestyle=':')
ax[0,0].axvline(0, color='gray', linestyle=':')


if partie_pos==False:
	ax[0,0].plot(k,sigma_big_img,'b:',alpha=0.15)
	ax[0,0].scatter(k, sigma_big_img, marker='o', color='b', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	
	ax[0,0].plot(k,sigma_big_ree,'r:',alpha=0.15)
	ax[0,0].scatter(k, sigma_big_ree, marker='o', color='r', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	
	if np.abs(np.max(sigma_big_img)) - np.abs(np.min(sigma_big_img)) < 0:
		ax[0,0].set_ylim(-np.abs(np.min(sigma_big_img)), np.abs(np.min(sigma_big_img)))
	else:
		ax[0,0].set_ylim(-np.abs(np.max(sigma_big_img)), np.abs(np.max(sigma_big_img)))
	
		
else:
	ax[0,0].plot(k[sigma_big_img>=0],sigma_big_img[sigma_big_img>=0],'b:',alpha=0.15)
	ax[0,0].scatter(k[sigma_big_img>=0], sigma_big_img[sigma_big_img>=0], marker='o', color='b', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	
	ax1 = ax[0,0].twinx()
	
	ax1.plot(k[sigma_big_img>=0],sigma_big_ree[sigma_big_img>=0],'r:',alpha=0.15)
	ax1.scatter(k[sigma_big_img>=0], sigma_big_ree[sigma_big_img>=0], marker='o', color='r', edgecolor='k', alpha=0.6,s=50, label=r'$\phi$')
	ax1.tick_params(right=True,direction='in', size=4, width=1,color='red',labelcolor='red')
	ax1.set_ylabel(r'$\sigma_\mathbf{Re} = \mathbf{Re}\{c\}.k ~\geq~ 0$',size=font_size,color='red')
	
	ax1.spines['right'].set_color('red')                         # spine
	ax1.yaxis.label.set_color('red')
	ax1.spines['right'].set_linewidth(2.25)  # Adjust thickness here   
	
	ax1.set_ylim(0,np.max(sigma_big_ree[sigma_big_img>=0]))
	
	
	if np.abs(np.max(sigma_big_img)) - np.abs(np.min(sigma_big_img)) < 0:
		ax[0,0].set_ylim(0, np.abs(np.min(sigma_big_img)))
	else:
		ax[0,0].set_ylim(0, np.abs(np.max(sigma_big_img)))
	
ax[0,0].axhline(0, color='gray', linestyle=':')
ax[0,0].axvline(0, color='gray', linestyle=':')

for spine in ax[0,0].spines.values():
    spine.set_linewidth(2)


#ax[0,0].set_xscale('log')

# Common elements
ax[0,0].set_xlabel(r'$k$',size=font_size)
ax[0,0].set_ylabel(r'$\sigma_\mathbf{Im} = \mathbf{Im}\{c\}.k ~\geq~ 0$',size=font_size,color='blue')
ax[0,0].set_title(r'$\sigma_\mathbf{Im}$ and $\sigma_\mathbf{Re}$',size=font_size)
ax[0,0].tick_params(top=True,right=False,direction='in', size=4, width=1,color='blue',labelcolor='blue')
ax[0,0].spines['left'].set_color('blue')                         # spine
ax[0,0].yaxis.label.set_color('blue')
ax[0,0].spines['left'].set_linewidth(2.25)  # Adjust thickness here      
ax[0,0].tick_params(axis='x', colors='black', direction='in', size=4, width=1)

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





# pour k,sigma_big_img
Nfourier = len(sigma_big_img)
kmax = k[-1] - k[0]
Fs = Nfourier/kmax
fourier11 = np.fft.fft(sigma_big_img,len_fft)/(Nfourier/2)
freq11 = np.fft.fftfreq(Nfourier,1/Fs)


ax[1,0].plot(freq11[0:Nfourier//2],np.abs(fourier11[0:Nfourier//2]),color='r',label=r'DFT of $\sigma$')
ax[1,0].tick_params(right=True,top=True,direction='in',size=4,width=1)
ax[1,0].axhline(0, color='gray', linestyle=':')
ax[1,0].axvline(0, color='gray', linestyle=':')
ax[1,0].set_xlabel('Frequency',size=font_size)
ax[1,0].set_ylabel('Amplitude',size=font_size)

ax[1,0].set_ylim(0,np.max(np.abs(fourier11[0:Nfourier//2])))
ax[1,0].legend(loc='best',fancybox=False,title='Zero-padding : '+str(len_fft-Nfourier))


for spine in ax[1,0].spines.values():
    spine.set_linewidth(2)


# pour  np.real(omega_big_c),np.imag(omega_big_c)
Nfourier = len(omega_big_c)
x_max = np.real(omega_big_c)[-1] - np.real(omega_big_c)[0]
Fs = Nfourier/x_max
fourier12 = np.fft.fft(np.imag(omega_big_c),len_fft)/(Nfourier/2)
freq12 = np.fft.fftfreq(Nfourier,1/Fs)


ax[1,1].plot(freq12[0:Nfourier//2],np.abs(fourier12[0:Nfourier//2]),color='green',label=r'DFT of $\omega$')

ax[1,1].tick_params(labelleft=False,right=True,top=True,direction='in',size=4,width=1)
ax[1,1].axhline(0, color='gray', linestyle=':')
ax[1,1].axvline(0, color='gray', linestyle=':')
ax[1,1].set_xlabel('Frequency',size=font_size)
	
ax[1,1].legend(loc='best',fancybox=False,title='Zero-padding : '+str(len_fft-Nfourier))
ax[1,1].set_ylim(0,np.max(np.abs(fourier11[0:Nfourier//2])))

for spine in ax[1,1].spines.values():
    spine.set_linewidth(2)


plt.tight_layout()

if save_png == True:
	plt.savefig('im_para/'+name_exp+'/fig3_'+name_exp+choice_plot_name+'.png',dpi=300)



'''
plt.show()

print('END')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')





