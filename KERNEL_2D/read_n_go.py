from KERNEL_TQG_solve_v3_bis import *
import imageio.v2 as imageio
import os
from io import BytesIO
from PIL import Image




N = 20

Lmin = 0.1
L = np.pi
x, y = np.linspace(Lmin,L,N), np.linspace(Lmin,L,N)

beta = 0.
F1star = 0. # 1/Rd**2

U0 = 1.
Theta0_U0 = 2. # ratio
#mode_index = 5 # mode to observe

k0, l0 = 2., 0.

nb_mode = int(input('How many modes ?')) 
# Parameters for plotting
timesteps = [0., 0.25, 0.50, 0.75]  # time points






zeta_list_2 = []
u_s_list_2 = []
v_s_list_2 = []

for i in tqdm(range(nb_mode)):
	
	zeta, us, vs = compute_TQG_2D(N, Lmin, L, beta, F1star, U0, Theta0_U0, i, k0, l0,timesteps)

	zeta_list_2.append(zeta)
	u_s_list_2.append(us)
	v_s_list_2.append(vs)
	




zeta_list_2 = np.array(zeta_list_2)
zeta_final = np.nansum(zeta_list_2,axis=0)

u_s_list_2 = np.array(u_s_list_2)
u_s_final = np.nansum(u_s_list_2,axis=0)

v_s_list_2 = np.array(v_s_list_2)
v_s_final = np.nansum(v_s_list_2,axis=0)




fig, (ax) = plt.subplots(1,4,figsize=(16, 4))

vmax = np.nanmax(zeta_final)
vmin = -vmax



for i in range(zeta_final.shape[0]):
	im = ax[i].contourf(x,y,zeta_final[i,:,:],20,cmap='RdBu_r',vmin=vmin,vmax=vmax)
	ax[i].streamplot
	ax[i].set_title(str(timesteps[i]))
	ax[i].streamplot(x,y,u_s_final[i,:,:],v_s_final[i,:,:],color='k',linewidth=0.5,arrowsize=0.75)
	
	
fig.colorbar(im,ax=ax[3])
	
	
plt.show()
