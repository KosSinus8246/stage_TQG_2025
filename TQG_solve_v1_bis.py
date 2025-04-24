import numpy as np
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
# VARIABLES, ESPACE ...
##################################



Ny, Nk = 60, 60
Ly = np.pi
dy = Ly/Ny
y_l, k = np.linspace(0.1,Ly,Ny), np.linspace(0.1,0.1+Nk*0.1,Nk)


beta, Rd = 0, 1 #1e-11
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
# CONSTRUCTION DE LA MATRICE B
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
# CONSTRUCTION DE LA MATRICE A
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


fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.axhline(0, color='gray', linestyle=':')
ax.axvline(0, color='gray', linestyle=':')
ax.scatter(k, sigma1, marker='o', color='b', edgecolor='k', alpha=0.6, label=r'$\sigma_\phi$')

axbis = ax.twinx()
axbis.scatter(k, sigma2, marker='^', color='r', edgecolor='k', alpha=0.6, label=r'$\sigma_\Theta$')

# Axis colors
ax.set_ylabel(r'$\sigma_\phi$', color='blue')
ax.tick_params(axis='y', colors='blue',direction='in',size=4,width=1)
ax.spines['left'].set_color('blue')
ax.spines['left'].set_linewidth(2)

axbis.set_ylabel(r'$\sigma_\Theta$', color='red')
axbis.tick_params(axis='y', colors='red',direction='in',size=4,width=1)
axbis.spines['right'].set_color('red')
axbis.spines['right'].set_linewidth(2)

ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# Common elements
ax.set_xlabel(r'$k$')
ax.set_title(r'$\sigma = \mathbf{Im}\{c\}.k$')
ax.tick_params(top=True,direction='in', size=4, width=1)

# Combine legends
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = axbis.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, loc='best',fancybox=False)


# stuff to put 0 on the same line
if np.abs(sigma1.max()) - np.abs(sigma1.min()) > 0:
	vlim1phi, vlim2phi = -sigma1.max(), sigma1.max()
	print('!!')
else:
	vlim1phi, vlim2phi = sigma1.min(), -sigma1.min()
	print('**')
	
ax.set_ylim(vlim1phi, vlim2phi)

# stuff to put 0 on the same line
if np.abs(sigma2.max()) - np.abs(sigma2.min()) > 0:
	vlim1theta, vlim2theta = -sigma2.max(), sigma2.max()
	print('!!')
else:
	vlim1theta, vlim2theta = sigma2.min(), -sigma2.min()
	print('**')

axbis.set_ylim(vlim1theta, vlim2theta)




plt.tight_layout()



print('-----------------------------------------------------')


#####################
# stabiilité

borne = (1/4)*(Un*y_l)**2
test_crit = Un * G12

print('Stability analysis : ')
if (test_crit < borne).all() == True:
	print('Totally unstable')
else:
	print('Stable ?')
print('-----------------------------------------------------')


fig, (ax) = plt.subplots(1,1,figsize=(8,5))

ax.axhline(0, color='gray', linestyle=':')
ax.axvline(0, color='gray', linestyle=':')
ax.plot(y_l,borne,'r--',label=r'Borne : $\frac{1}{4}.(\overline{U}.y)^2$')
ax.plot(y_l,test_crit,'b',label=r'$\overline{U}.\frac{\mathrm{d}\Theta}{\mathrm{d}y}$')
ax.fill_between(y_l,borne,test_crit,color='orange',alpha=0.3)
ax.tick_params(left=True,right=True,top=True,bottom=True,direction='in',size=4,width=1)
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'Analysis')
ax.legend(loc='best',fancybox=False)

# Make the axes (spines) bold
for spine in ax.spines.values():
    spine.set_linewidth(2)


plt.tight_layout()


plt.show()



print('END')






