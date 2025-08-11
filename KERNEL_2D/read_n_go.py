from KERNEL_TQG_solve_v3_bis import *
import imageio.v2 as imageio
import cmocean
'''
import os
from io import BytesIO
from PIL import Image'''


mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'Courier New'
mpl.rcParams['legend.edgecolor'] = '0'


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_3_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~JULIE~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')



#########################"
# 1) Computes c and X for QG and TQG with : compute_TQG_2D()
# 2) Find the most unstable modes with : get_ix()
# 3) Computes PSI, zeta, u, v with : compute_variables()




N = 25
Lmin = 0.1
L = 2*np.pi
dh = L/N
x, y = np.linspace(Lmin,L,N), np.linspace(Lmin,L,N)

beta = 0.
F1star = 0. # 1/Rd**2
U0 = 1.
Theta0_U0 = 1. # ratio
k0, l0 = 2., 0.
Lstar = 0.5


BC = ''
crit = 'imag'
timesteps = [0., 1., 3., 5., 7.]
#timesteps = [0., 2.5, 5., 7.5, 10.]

#####
# compute eigenvalues and eigenvectors
x_l, y_l, xx, yy, c, c_NT, X, X_NT, Un, Thetabar = compute_TQG_2D(N, Lmin, L, beta, F1star, U0, Theta0_U0, k0, l0, dh, BC, Lstar)



#####
# finding the modes that are important
ix_norm_c__, ix_norm_cNT__ = get_ix(c,c_NT,crit)
nb_modes = int(input('How many modes ? '))

	
	

ix_norm_c__2, ix_norm_cNT__2 = ix_norm_c__[:nb_modes], ix_norm_cNT__[:nb_modes]

# final list that stack each paramaters for each mode
zeta_list_2 = []
zeta_list_2NT = []
theta_list_2 = []


for i in tqdm(range(nb_modes)):
	
	zeta, zetaNT, theta  = compute_variables(N,ix_norm_c__2[i], ix_norm_cNT__2[i], c, c_NT, X, X_NT,timesteps, k0, l0, xx, yy, dh, Un, Thetabar)

	zeta_list_2.append(zeta)
	zeta_list_2NT.append(zetaNT)
	
	theta_list_2.append(theta)




	



# convert all in an array
zeta_list_2 = np.array(zeta_list_2)
zeta_final = np.nansum(zeta_list_2,axis=0)

zeta_list_2NT = np.array(zeta_list_2NT)
zeta_finalNT = np.nansum(zeta_list_2NT,axis=0)


theta_list_2 = np.array(theta_list_2)
theta_final = np.nansum(theta_list_2,axis=0)



fig, ax = plt.subplots(2, len(timesteps), figsize=(16, 6))
fig.suptitle(r'Evolution of ζ (top : TQG and bottom : QG) : sum of '+str(nb_modes)+' modes', fontweight='bold')

# figure for theta
fig2, ax2 = plt.subplots(1, len(timesteps), figsize=(16, 4))
fig2.suptitle(r'Evolution of Θ : sum of '+str(nb_modes)+' modes', fontweight='bold')

#vmax = np.mean(zeta_final)+200
#vmin = -vmax


 #maxNT = np.mean(zeta_finalNT)+200
#minNT = -vmaxNT

#vmax_theta = 1
#vmin_theta = -vmax_theta


for i in range(zeta_final.shape[0]):
	#im1 = ax[0,i].contourf(x,y,zeta_final[i,:,:],30,cmap='coolwarm',vmin=vmin,vmax=vmax)
	im1 = ax[0,i].contourf(x[1:-1],y[1:-1],zeta_final[i,1:-1,1:-1],30,cmap='coolwarm') 

	ax[0,i].set_title(str(timesteps[i]))

	#cs = ax[0,i].contour(x,y,zeta_final[i,:,:],15,colors='k')
	cs = ax[0,i].contour(x[1:-1],y[1:-1],zeta_final[i,1:-1,1:-1],7,colors='k')
	ax[0,i].clabel(cs,colors='k')
	
	ax[0,i].set_xlim(np.min(x[1:-1]),np.max(x[1:-1]))
	ax[0,i].set_ylim(np.min(y[1:-1]),np.max(y[1:-1]))
	
	ax[0,i].set_title('t = '+str(timesteps[i]), fontweight="bold")
	ax[0,i].tick_params(top=True,right=True,labelbottom=False,direction='in',size=4,width=1)
	
	for spine in ax[0,i].spines.values():
		spine.set_linewidth(2)
	for tick in ax[0, i].get_xticklabels():
		tick.set_fontweight('bold')

	for tick in ax[0, i].get_yticklabels():
		tick.set_fontweight('bold')
	    
	    
	im_theta = ax2[i].contourf(x[1:-1],y[1:-1],theta_final[i,1:-1,1:-1],30,cmap='coolwarm') 
	ax2[i].set_title(str(timesteps[i]))
	ax2[i].set_xlim(np.min(x[1:-1]),np.max(x[1:-1]))
	ax2[i].set_ylim(np.min(y[1:-1]),np.max(y[1:-1]))
	ax2[i].set_title('t = '+str(timesteps[i]), fontweight="bold")
	ax2[i].tick_params(top=True,right=True,direction='in',size=4,width=1)
	cs = ax2[i].contour(x[1:-1],y[1:-1],theta_final[i,1:-1,1:-1],7,colors='k')
	ax2[i].clabel(cs,colors='k')
	ax2[i].set_xlabel('x', fontweight="bold")
	ax2[i].clabel(cs)
	for spine in ax2[i].spines.values():
		spine.set_linewidth(2)
	for tick in ax2[i].get_xticklabels():
		tick.set_fontweight('bold')

	for tick in ax2[i].get_yticklabels():
		tick.set_fontweight('bold')
	
	
	
	#im2 = ax[1,i].contourf(x,y,zeta_finalNT[i,:,:],30,cmap='coolwarm',vmin=vminNT,vmax=vmaxNT)
	im2 = ax[1,i].contourf(x[1:-1],y[1:-1],zeta_finalNT[i,1:-1,1:-1],30,cmap='coolwarm')  
	#cs = ax[1,i].contour(x,y,zeta_finalNT[i,:,:],15,colors='k')
	cs = ax[1,i].contour(x[1:-1],y[1:-1],zeta_finalNT[i,1:-1,1:-1],7,colors='k')
	
	ax[1,i].clabel(cs,colors='k')
	#im2 = ax[1,i].pcolormesh(x,y,zeta_finalNT[i,:,:],cmap='RdBu_r',vmin=vminNT,vmax=vmaxNT)
	ax[1,i].set_xlim(np.min(x[1:-1]),np.max(x[1:-1]))
	ax[1,i].set_ylim(np.min(y[1:-1]),np.max(y[1:-1]))
	ax[1,i].set_xlabel(r"x", fontweight="bold")
	ax[1,i].tick_params(top=True,right=True,direction='in',size=4,width=1)
	for tick in ax[1, i].get_xticklabels():
	    tick.set_fontweight('bold')

	for tick in ax[1, i].get_yticklabels():
		tick.set_fontweight('bold')

	
	for spine in ax[1,i].spines.values():
		spine.set_linewidth(2)
	
	
ax[0,0].set_ylabel(r'y', fontweight="bold")
ax[1,0].set_ylabel(r'y', fontweight="bold")
ax2[0].set_ylabel('y', fontweight="bold")


for i in range(1,len(timesteps)):
	ax[0,i].tick_params(top=True,right=True,labelbottom=False,labelleft=False,direction='in',size=4,width=1)
	ax[1,i].tick_params(top=True,right=True,labelbottom=True,labelleft=False,direction='in',size=4,width=1)







plt.tight_layout()

	
plt.show()
