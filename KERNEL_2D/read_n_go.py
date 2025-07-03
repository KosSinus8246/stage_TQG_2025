from KERNEL_TQG_solve_v3_bis import *
import imageio.v2 as imageio

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




N = 20

Lmin = 0.1
L = 2*np.pi
dh = L/N
x, y = np.linspace(Lmin,L,N), np.linspace(Lmin,L,N)

beta = 0.
F1star = 0. # 1/Rd**2

U0 = 1.
Theta0_U0 = 1. # ratio

k0, l0 = 2., 0.

# Parameters for plotting
timesteps = [0., 0.25, 0.50, 0.75]  # time points

nb_modes = int(input('How many modes ? '))



#####
# compute eigenvalues and eigenvectors
x_l, y_l, xx, yy, c, c_NT, X, X_NT = compute_TQG_2D(N, Lmin, L, beta, F1star, U0, Theta0_U0, k0, l0, dh)



#####
# finding the modes that are important
ix_norm_c__, ix_norm_cNT__ = get_ix(c,c_NT)
ix_norm_c__2, ix_norm_cNT__2 = ix_norm_c__[:nb_modes], ix_norm_cNT__[:nb_modes]


# final list that stack each paramaters for each mode
zeta_list_2 = []
u_s_list_2 = []
v_s_list_2 = []

zeta_list_2NT = []
u_s_list_2NT = []
v_s_list_2NT = []



for i in tqdm(range(nb_modes)):
	
	zeta, u_s, v_s, zetaNT, u_sNT, v_sNT  = compute_variables(N,ix_norm_c__2[i], ix_norm_cNT__2[i], c, c_NT, X, X_NT,timesteps, k0, l0, xx, yy, dh)

	zeta_list_2.append(zeta)
	u_s_list_2.append(u_s)
	v_s_list_2.append(v_s)
	
	zeta_list_2NT.append(zetaNT)
	u_s_list_2NT.append(u_sNT)
	v_s_list_2NT.append(v_sNT)





	



# convert all in an array
zeta_list_2 = np.array(zeta_list_2)
zeta_final = np.nansum(zeta_list_2,axis=0)

u_s_list_2 = np.array(u_s_list_2)
u_s_final = np.nansum(u_s_list_2,axis=0)

v_s_list_2 = np.array(v_s_list_2)
v_s_final = np.nansum(v_s_list_2,axis=0)



zeta_list_2NT = np.array(zeta_list_2NT)
zeta_finalNT = np.nansum(zeta_list_2NT,axis=0)

u_s_list_2NT = np.array(u_s_list_2NT)
u_s_finalNT = np.nansum(u_s_list_2NT,axis=0)

v_s_list_2NT = np.array(v_s_list_2NT)
v_s_finalNT = np.nansum(v_s_list_2NT,axis=0)





fig, ax = plt.subplots(2, len(timesteps), figsize=(16, 7))
fig.suptitle(r'Evolution of $\zeta_\mathbf{TQG}$ and $\zeta_\mathbf{QG}$ : sum of '+str(nb_modes)+' modes', fontweight='bold')

vmax = np.nanmax(zeta_final)
vmin = -vmax


vmaxNT = np.nanmax(zeta_finalNT)
vminNT = -vmaxNT



for i in range(zeta_final.shape[0]):
	im1 = ax[0,i].contourf(x,y,zeta_final[i,:,:],25,cmap='RdBu_r',vmin=vmin,vmax=vmax)
	#im1 = ax[0,i].pcolormesh(x,y,zeta_final[i,:,:],cmap='RdBu_r',vmin=vmin,vmax=vmax)
	ax[0,i].set_title(str(timesteps[i]))
	ax[0,i].streamplot(x,y,u_s_final[i,:,:],v_s_final[i,:,:],color='k',linewidth=0.5,arrowsize=0.75,density=0.75)
	ax[0,i].set_xlim(np.min(x),np.max(x))
	ax[0,i].set_ylim(np.min(y),np.max(y))
	
	ax[0,i].set_title('t = '+str(timesteps[i]), fontweight="bold")
	ax[0,i].tick_params(top=True,right=True,labelbottom=False,direction='in',size=4,width=1)
	
	for spine in ax[0,i].spines.values():
		spine.set_linewidth(2)
	
	
	
	im2 = ax[1,i].contourf(x,y,zeta_finalNT[i,:,:],25,cmap='RdBu_r',vmin=vminNT,vmax=vmaxNT)
	#im2 = ax[1,i].pcolormesh(x,y,zeta_finalNT[i,:,:],cmap='RdBu_r',vmin=vminNT,vmax=vmaxNT)
	ax[1,i].streamplot(x,y,u_s_finalNT[i,:,:],v_s_finalNT[i,:,:],color='k',linewidth=0.5,arrowsize=0.75,density=0.75)
	ax[1,i].set_xlim(np.min(x),np.max(x))
	ax[1,i].set_ylim(np.min(y),np.max(y))
	ax[1,i].set_xlabel(r"x", fontweight="bold")
	ax[1,i].tick_params(top=True,right=True,direction='in',size=4,width=1)
	for tick in ax[1, i].get_xticklabels():
	    tick.set_fontweight('bold')

	for tick in ax[1, i].get_yticklabels():
	    tick.set_fontweight('bold')

	
	for spine in ax[1,i].spines.values():
		spine.set_linewidth(2)
	
	
ax[0,0].set_ylabel(r'y', fontweight="bold")
cbar_1 = fig.colorbar(im1, ax=ax[0,-1])
cbar_1.ax.yaxis.set_ticks_position('both')             # Ticks on both sides
cbar_1.ax.yaxis.set_tick_params(labelleft=False,       # Hide left labels
                               direction='in',    # Tick style
                               length=2,width=1)            # Length of ticks for visibility

# Set the border (spine) linewidth of the colorbar
for spine in cbar_1.ax.spines.values():
	spine.set_linewidth(1.5)  # You can set this to any float value



ax[0,1].tick_params(top=True,right=True,labelbottom=False,labelleft=False,direction='in',size=4,width=1)
ax[0,2].tick_params(top=True,right=True,labelbottom=False,labelleft=False,direction='in',size=4,width=1)
ax[0,3].tick_params(top=True,right=True,labelbottom=False,labelleft=False,direction='in',size=4,width=1)

ax[1,0].set_ylabel(r'y', fontweight="bold")
cbar_2 = fig.colorbar(im2, ax=ax[1,-1])
cbar_2.ax.yaxis.set_ticks_position('both')             # Ticks on both sides
cbar_2.ax.yaxis.set_tick_params(labelleft=False,       # Hide left labels
                               direction='in',    # Tick style
                               length=2,width=1)            # Length of ticks for visibility


# Set the border (spine) linewidth of the colorbar
for spine in cbar_2.ax.spines.values():
	spine.set_linewidth(1.5)  # You can set this to any float value


ax[1,1].tick_params(top=True,right=True,labelbottom=True,labelleft=False,direction='in',size=4,width=1)
ax[1,2].tick_params(top=True,right=True,labelbottom=True,labelleft=False,direction='in',size=4,width=1)
ax[1,3].tick_params(top=True,right=True,labelbottom=True,labelleft=False,direction='in',size=4,width=1)






plt.tight_layout()

	
plt.show()
