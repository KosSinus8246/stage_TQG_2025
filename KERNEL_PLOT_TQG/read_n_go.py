from KERNEL_TQG_solve_v2_bis import *


Ny, Nk = 60, 51
dk = 0.1
ymin, kmin, Ly, Lk = 0.1, 0.1, np.pi, 0.1+dk*Nk


beta = 0 
F1star = 0 # 1/Rd**2

U0 = 1


# compute_sigmas(False, Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, 1)
# compute_sigmas(False, Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, 0.1)

Theta0_U0 = np.linspace(0.1, 1.0, 2)  # 5 values between 0.1 and 1.0

for theta in Theta0_U0:
    compute_sigmas(False, Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, theta)
    plt.savefig('output/fig_ratio_'+str(theta)+'.png',dpi=300)

# vider la mémoire avec show et tout refermer
plt.show(block=False)
plt.close('all')
