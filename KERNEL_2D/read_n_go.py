from KERNEL_TQG_solve_v3_bis import *
import imageio.v2 as imageio
import os
from io import BytesIO
from PIL import Image

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_2_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')



N = 20

Lmin = 0.1
L = np.pi

beta = 0.
F1star = 0. # 1/Rd**2

U0 = 1.
Theta0_U0 = 2. # ratio
#mode_index = 5 # mode to observe

k0, l0 = 2., 0.


zeta_list_2 = []

for i in tqdm(range(1,3)):

	zeta_list_2.append(compute_TQG_2D(N, Lmin, L, beta, F1star, U0, Theta0_U0, i, k0, l0))


zeta_list_2 = np.array(zeta_list_2)





fig, (ax) = plt.subplots(1,4,figsize=(16, 4))


for i in range(zeta_list_2.shape[0]):
	ax[i].contourf(zeta_final[i,:,:])
	
	
plt.show()
