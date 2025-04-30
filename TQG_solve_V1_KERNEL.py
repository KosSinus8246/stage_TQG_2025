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



# if 1 : it will plot the 2 varibale phi and theta
# if 2 : it will only plot the part where Im(c) is important
choice_plot = 2
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
Ly, Lk = np.pi, 100 #40
dy = Ly/Ny
y_l, k = np.linspace(0.1,Ly,Ny), np.linspace(0.1,Lk,Nk)
dk = Lk/Nk


beta, Rd = 5, 1 #1e-11
F1star = 0 #1/Rd**2
K2 = (k**2 + F1star)*dy**2
#K2 = 0
U0= 1


#phi, theta = np.zeros((Ny, Nk)), np.zeros((Ny, Nk))

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



# Open a file in write mode
with open('im_para/variables_used_'+name_exp+'.txt', 'w') as file:
    file.write(f"beta = {beta}\n")
    file.write(f"Rd = {Rd}\n")
    file.write(f"U0 = {U0}\n")
    file.write(f"Theta0 = {Theta0}\n")

print('Variables stored into variables_used_'+name_exp+'.txt')



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
