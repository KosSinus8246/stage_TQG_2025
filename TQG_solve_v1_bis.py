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

print('TQG_SOLVE_1_BIS')
print('-----------------------------------------------------')



# cf TQG notes : A.X = c.B.X
# @uthor : dimitri moreau 23/04/2025



##################################
# VARIABLES, SPACE ...
##################################



Ny, Nk = 60, 60
Ly, Lk = np.pi, 0.1+Nk*0.1
dy = Ly/Ny
y_l, k = np.linspace(0.1,Ly,Ny), np.linspace(0.1,Lk,Nk)
dk = Lk/Nk


beta, Rd = 5, 1 #1e-11
F1star = 0 #1/Rd**2
K2 = (k**2 + F1star)*dy**2
#K2 = 0
U0= 1


phi, theta, Un = np.zeros((Ny, Nk)), np.zeros((Ny, Nk)), U0*np.exp(-y_l**2)
# phi_r,theta_r = phi.reshape(Nk*Ny), theta.reshape(Nk*Ny)

# X = np.array([phi_r,theta_r])

# V/G/Mn
Theta0 = 1
Vn = Un * dy**2
G12 = -2*y_l*Theta0*np.exp(-y_l**2)
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
A11 = np.zeros((Ny,Nk))
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

sigma1 = k*np.imag(c[:Ny])
sigma2 = k*np.imag(c[Ny:])



print('COMPUTATION : OK')
print('/////////////////////////////////////////////////////')



##################################
# PLOT
##################################



print('PLOT...')


##################################
# Plot 1



fig, ax = plt.subplots(1, 2, figsize=(15, 6))


ax[0].axhline(0, color='gray', linestyle=':')
ax[0].axvline(0, color='gray', linestyle=':')
ax[0].plot(k,sigma1,'b:',alpha=0.15)
ax[0].scatter(k, sigma1, marker='o', color='b', edgecolor='k', alpha=0.6, label=r'$\sigma_\phi$')

axbis = ax[0].twinx()
axbis.plot(k,sigma2,'r:',alpha=0.1)
axbis.scatter(k, sigma2, marker='^', color='r', edgecolor='k', alpha=0.6, label=r'$\sigma_\Theta$')

# Axis colors
ax[0].set_ylabel(r'$\sigma_\phi$', color='blue')
ax[0].tick_params(axis='y', colors='blue',direction='in',size=4,width=1)
ax[0].spines['left'].set_color('blue')
ax[0].spines['left'].set_linewidth(2)

axbis.set_ylabel(r'$\sigma_\Theta$', color='red')
axbis.tick_params(axis='y', colors='red',direction='in',size=4,width=1)
axbis.spines['right'].set_color('red')
axbis.spines['right'].set_linewidth(2)

ax[0].spines['bottom'].set_linewidth(2)
ax[0].spines['top'].set_linewidth(2)

# Common elements
ax[0].set_xlabel(r'$k$')
ax[0].set_title(r'$\sigma = \mathbf{Im}\{c\}.k$')
ax[0].tick_params(top=True,direction='in', size=4, width=1)

# Combine legends
handles1, labels1 = ax[0].get_legend_handles_labels()
handles2, labels2 = axbis.get_legend_handles_labels()
ax[0].legend(handles1 + handles2, labels1 + labels2, loc='best',fancybox=False)




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
ax[1].plot(y_l,borne,'k--',label=r'Borne : $\frac{1}{4}.(\overline{U}.y)^2$')
ax[1].plot(y_l,test_crit,'orange',label=r'$\overline{U}.\frac{\mathrm{d}\Theta}{\mathrm{d}y}$')
ax[1].fill_between(y_l,borne,test_crit,color='orange',alpha=0.3)
ax[1].tick_params(left=True,right=True,top=True,bottom=True,direction='in',size=4,width=1)
ax[1].set_xlabel(r'$y$')
ax[1].set_ylabel(r'Value $\propto ~\overline{U}.y^2$')
ax[1].legend(loc='best',fancybox=False)

# Make the axes (spines) bold
for spine in ax[1].spines.values():
    spine.set_linewidth(2)


plt.tight_layout()




##################################
# Plot 2


# derivative of sigma

dsigma1, dsigma2 = np.zeros_like(sigma1), np.zeros_like(sigma2)

for i in range(len(k)-1):
	dsigma1[i] = (sigma1[i+1] - sigma1[i-1])/(2*dk)
	dsigma2[i] = (sigma2[i+1] - sigma2[i-1])/(2*dk)




nb_bins = 30
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

sns.histplot(sigma1,bins=nb_bins,ax=ax[0],kde=True,stat='percent',color='b',label=r'$\sigma_\phi$')
#ax[0].set_ylim(0,40)
ax1 = ax[0].twiny()
sns.histplot(sigma2,bins=nb_bins,ax=ax1,kde=True,stat='percent',color='r',label=r'$\sigma_\theta$')
#ax[0].set_ylim(0,40)
ax[0].set_title(r'$\sigma$')

# Axis colors
ax[0].tick_params(axis='x', colors='blue',direction='in',size=4,width=1)
ax[0].tick_params(right=True,direction='in',size=4,width=1)
ax[0].spines['bottom'].set_color('blue')
ax[0].spines['bottom'].set_linewidth(2)

ax1.tick_params(axis='x', colors='red',direction='in',size=4,width=1)
ax1.spines['top'].set_color('red')
ax1.spines['top'].set_linewidth(2)

ax[0].spines[['left','right']].set_linewidth(2)

# Combine legends
handles1, labels1 = ax[0].get_legend_handles_labels()
handles2, labels2 = ax1.get_legend_handles_labels()
ax[0].legend(handles1 + handles2, labels1 + labels2, loc='best',fancybox=False)






sns.histplot(dsigma1,bins=nb_bins,ax=ax[1],kde=True,stat='percent',color='b',label=r'$\frac{\partial \sigma_\phi}{\partial k}$')
#ax[1].set_ylim(0,40)
ax[1].set_title(r'$\frac{\partial \sigma}{\partial k}$')
ax2 = ax[1].twiny()
sns.histplot(dsigma2,bins=nb_bins,ax=ax2,kde=True,stat='percent',color='r',label=r'$\frac{\partial \sigma_\theta}{\partial k}$')
ax[1].set_ylabel('')
#ax[1].set_ylim(0,40)


# Axis colors
ax[1].tick_params(axis='x', colors='blue',direction='in',size=4,width=1)
ax[1].tick_params(right=True,direction='in',size=4,width=1)
ax[1].spines['bottom'].set_color('blue')
ax[1].spines['bottom'].set_linewidth(2)

ax2.tick_params(axis='x', colors='red',direction='in',size=4,width=1)
ax2.spines['top'].set_color('red')
ax2.spines['top'].set_linewidth(2)

ax[1].spines[['left','right']].set_linewidth(2)


# Combine legends
handles1, labels1 = ax[1].get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax[1].legend(handles1 + handles2, labels1 + labels2, loc='best',fancybox=False)





##################################
# Plot 3



fig, (ax) = plt.subplots(1,2,figsize=(15,6))

omega = np.sqrt(c)

ax[0].plot(np.real(omega[:Ny]),np.imag(omega[:Ny]),'b+')
ax[0].set_xlabel(r'$\mathbf{Re}\{\omega\}$')
ax[0].set_ylabel(r'$\mathbf{Im}\{\omega\}$ of $\phi$')
ax1=ax[0].twinx()
ax1.plot(np.real(omega[Ny:]),np.imag(omega[Ny:]),'r+')
ax1.set_ylabel(r'$\mathbf{Im}\{\omega\}$ of $\Theta$')



# Axis colors
ax[0].set_ylabel(r'$\sigma_\phi$', color='blue')
ax[0].tick_params(axis='y', colors='blue',direction='in',size=4,width=1)
ax[0].spines['left'].set_color('blue')
ax[0].spines['left'].set_linewidth(2)
ax[0].tick_params(bottom=True, top=True,size=4,width=1,direction='in')

ax1.set_ylabel(r'$\sigma_\Theta$', color='red')
ax1.tick_params(axis='y', colors='red',direction='in',size=4,width=1)
ax1.spines['right'].set_color('red')
ax1.spines['right'].set_linewidth(2)

ax[0].spines[['bottom','top']].set_linewidth(2)
#ax.axhline(0, color='gray', linestyle=':')
#ax.axvline(0, color='gray', linestyle=':')


#ax.set_ylim(1e-5,1)
#ax1.set_ylim(1e-5,1)

ax[0].set_yscale('log')
ax1.set_yscale('log')


ax[1].plot(Un,y_l,'k')
ax[1].tick_params(right=True, top=True,size=4,width=1,direction='in')
ax[1].spines[['top','bottom','right','left']].set_linewidth(2)
ax[1].set_ylabel(r'$y$')
ax[1].set_xlabel(r'$\overline{U}$')

    
plt.tight_layout()


plt.show()



print('END')






