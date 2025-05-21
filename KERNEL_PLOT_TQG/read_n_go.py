from KERNEL_TQG_solve_v2_bis import *


Ny, Nk = 60, 51
dk = 0.1
ymin, kmin, Ly, Lk = 0.1, 0.1, np.pi, 0.1+dk*Nk


beta = 0 
F1star = 0 # 1/Rd**2

U0 = 1

compute_sigmas(False, Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, 1)
compute_sigmas(False, Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, 0.1)
plt.show()
