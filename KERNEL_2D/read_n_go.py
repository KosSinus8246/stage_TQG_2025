from KERNEL_TQG_solve_v3_bis import *
#import imageio.v2 as imageio
#import cmocean

from matplotlib.font_manager import FontProperties
# Define bold font
bold_font = FontProperties(weight='bold')




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
# 2) Find most unstables modes with : get_ix()
# 3) Computes PSI, zeta, u, v with : compute_variables()

'''
PARAMETERS

N : Size of the grid
Lmin, L : dimension of the grid

beta, F1star : beta effect and deformation radius
U0, Theta0_U0 : Mean velocity and ratio Theta0/U0
k0, l0 : wavelenght of the perturbation
Lstar : Shape of the temperature's gaussian
std : Standard deviation of the random perturbation only
for the thermal profile

BC : Boundary conditions : '' or 'activated'
crit : The program sort the modes following the most import
part of imaginary : 'imag' or 'real' or module : 'imag_real'
timesteps : the snapshots that you want to see on the plot
'''



N = 30
Lmin = 0.
epsilon = 0.001
#Lmin = 0.1
L = 2*np.pi
dh = L/N
x, y = np.linspace(Lmin,L,N), np.linspace(Lmin,L,N)

beta = 0.
F1star = 0.
U0 = 1.
Theta0_U0 = 3.
k0 = 1.
Lstar = 0.5
std = 0.


BC = ''
crit = 'imag'

timesteps = [0., 1., 2., 3.]
#timesteps = [0., 5., 10., 15.]


#lev_cont = [-0.75,-0.5, 0., 0.5, 0.75]
lev_cont = 6


#####
# compute eigenvalues and eigenvectors
x_l, y_l, xx, yy, c, c_NT, X, X_NT, Un, Thetabar = compute_TQG_2D(N, Lmin, L, beta, F1star, U0, Theta0_U0, k0, dh, BC, Lstar, std)


#####
# computing linear max time
sigma = np.max(np.imag(c)*k0)
time_linear = (1/sigma) * np.log(1/epsilon)

#sigmaNT = np.max(np.imag(c_NT)*k0)
#time_linearNT = (1/sigmaNT) * np.log(1/epsilon)

if timesteps[-1] > time_linear:
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print('WARNING : RANGE OF TIME OVER THE TIME MAX CRITERIA')
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


print('-----------------------------------------------------')
print('Maximum Linear Time TQG = ',time_linear)
print('-----------------------------------------------------')
#print('Maximum Linear Time QG = ',time_linearNT)


#####
# finding the modes that are important
ix_norm_c__, ix_norm_cNT__ = get_ix(c,c_NT,crit)
nb_modes = int(input('How many modes ? '))




ix_norm_c__2, ix_norm_cNT__2 = ix_norm_c__[:nb_modes], ix_norm_cNT__[:nb_modes]

# final list that stack each paramaters for each mode
zeta_list_2 = []
zeta_list_2NT = []
theta_list_2 = []

psi_list_2 = []
psi_listNT_2 = []


for i in tqdm(range(nb_modes)):

	theta, psi, psiNT  = compute_variables_prime(N,ix_norm_c__2[i], ix_norm_cNT__2[i], c, c_NT, X, X_NT,timesteps, k0, xx, yy, dh, Un, Thetabar,epsilon)

	theta_list_2.append(theta)

	psi_list_2.append(psi)
	psi_listNT_2.append(psiNT)






theta_list_2 = np.array(theta_list_2)
theta_final = np.nansum(theta_list_2,axis=0)
psi_list_2 = np.array(psi_list_2)
psi_final = np.nansum(psi_list_2,axis=0)
psi_listNT_2 = np.array(psi_listNT_2)
psi_finalNT = np.nansum(psi_listNT_2,axis=0)


#PSI = PSI - Un*yy
#PSI_NT = PSI_NT - Un*yy
#####
# Build background streamfunction psi0(y) = ∫ U(y) dy  (trapezoidal)
U1d = Un[:, 0]
psi0_y = np.zeros(N)
psi0_y[1:] = np.cumsum(0.5*(U1d[1:] + U1d[:-1])) * dh
psi_bg = psi0_y[:, None] * np.ones((1, N))


PSI = np.zeros_like(psi_final)
PSI_NT = np.zeros_like(psi_finalNT)
THETA = np.zeros_like(theta_final)

for i in range(len(timesteps)):

	PSI[i,:,:]    = psi_final[i,:,:]    + psi_bg
	PSI_NT[i,:,:] = psi_finalNT[i,:,:] + psi_bg
	THETA[i,:,:]  = theta_final[i,:,:] + Thetabar
	
	


ZETA, ZETANT = compute_zeta(N,len(timesteps),PSI, PSI_NT,dh)
ZETA, ZETANT = ZETA[0,:,:,:], ZETANT[0,:,:,:]






################
# spatial decomp

k_vals = np.arange(0,9,1)*2*np.pi
A_k, phi_k, A_bar_k, phi_bar_k = spatial_fourier_decomposition(psi_final, x_l, k_vals, dh)







fig, ax = plt.subplots(2, len(timesteps), figsize=(16, 6))
fig.suptitle(r'TQG : Evolution of ψ (top) and ζ (bottom) : sum of '+str(nb_modes)+' modes', fontweight='bold')


# figure for theta
fig2, ax2 = plt.subplots(1, len(timesteps), figsize=(16, 4))
fig2.suptitle(r'Evolution of Θ : sum of '+str(nb_modes)+' modes', fontweight='bold')

fig3, ax3 = plt.subplots(2, len(timesteps), figsize=(16, 6))
fig3.suptitle(r'QG : Evolution of ψ (top) and ζ (bottom) : QG) : sum of '+str(nb_modes)+' modes', fontweight='bold')






for i in range(ZETA.shape[0]):
	#im1 = ax[0,i].contourf(x,y,zeta_final[i,:,:],30,cmap='coolwarm',vmin=vmin,vmax=vmax)
	im1 = ax[0,i].contourf(x[1:-1],y[1:-1],PSI[i,1:-1,1:-1],30,cmap='coolwarm')

	ax[0,i].set_title(str(timesteps[i]))

	#cs = ax[0,i].contour(x,y,zeta_final[i,:,:],15,colors='k')
	cs = ax[0,i].contour(x[1:-1],y[1:-1],PSI[i,1:-1,1:-1],lev_cont,colors='k')
	lbl = ax[0,i].clabel(cs,colors='k')
	for lbl in lbl:
	        lbl.set_fontproperties(bold_font)

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


	im_theta = ax2[i].contourf(x[1:-1],y[1:-1],THETA[i,1:-1,1:-1],30,cmap='coolwarm')
	ax2[i].set_title(str(timesteps[i]))

	ax2[i].set_xlim(np.min(x[1:-1]),np.max(x[1:-1]))
	ax2[i].set_ylim(np.min(y[1:-1]),np.max(y[1:-1]))

	ax2[i].set_title('t = '+str(timesteps[i]), fontweight="bold")
	ax2[i].tick_params(top=True,right=True,direction='in',size=4,width=1)
	cs = ax2[i].contour(x[1:-1],y[1:-1],THETA[i,1:-1,1:-1],lev_cont,colors='k')
	lbl = ax2[i].clabel(cs,colors='k')
	for lbl in lbl:
		lbl.set_fontproperties(bold_font)

	ax2[i].set_xlabel('x', fontweight="bold")
	lbl = ax2[i].clabel(cs)
	for lbl in lbl:
		lbl.set_fontproperties(bold_font)
	for spine in ax2[i].spines.values():
		spine.set_linewidth(2)
	for tick in ax2[i].get_xticklabels():
		tick.set_fontweight('bold')

	for tick in ax2[i].get_yticklabels():
		tick.set_fontweight('bold')



	#im2 = ax[1,i].contourf(x,y,zeta_finalNT[i,:,:],30,cmap='coolwarm',vmin=vminNT,vmax=vmaxNT)
	im2 = ax[1,i].contourf(x[1:-1],y[1:-1],ZETA[i,1:-1,1:-1],30,cmap='coolwarm')
	#cs = ax[1,i].contour(x,y,zeta_finalNT[i,:,:],15,colors='k')
	cs = ax[1,i].contour(x[1:-1],y[1:-1],ZETA[i,1:-1,1:-1],lev_cont,colors='k')

	lbl = ax[1,i].clabel(cs,colors='k')
	for lbl in lbl:
		lbl.set_fontproperties(bold_font)
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




	ax3[0,i].contourf(x[1:-1],y[1:-1],PSI_NT[i,1:-1,1:-1],30,cmap='coolwarm')
	cs = ax3[0,i].contour(x[1:-1],y[1:-1],PSI_NT[i,1:-1,1:-1],lev_cont,colors='k')
	lbl = ax3[0,i].clabel(cs,colors='k')

	ax3[0,i].set_title('t = '+str(timesteps[i]),fontweight='bold')

	for tick in ax3[0, i].get_xticklabels():
		tick.set_fontweight('bold')
	for tick in ax3[0, i].get_yticklabels():
		tick.set_fontweight('bold')
	for spine in ax3[0,i].spines.values():
		spine.set_linewidth(2)
	for lbl in lbl:
		lbl.set_fontproperties(bold_font)


	ax3[0,i].set_xlim(np.min(x[1:-1]),np.max(x[1:-1]))
	ax3[0,i].set_ylim(np.min(y[1:-1]),np.max(y[1:-1]))




	ax3[1,i].contourf(x[1:-1],y[1:-1],ZETANT[i,1:-1,1:-1],30,cmap='coolwarm')
	cs = ax3[1,i].contour(x[1:-1],y[1:-1],ZETANT[i,1:-1,1:-1],lev_cont,colors='k')
	lbl = ax3[1,i].clabel(cs,colors='k')


	for tick in ax3[1, i].get_xticklabels():
		tick.set_fontweight('bold')
	for tick in ax3[1, i].get_yticklabels():
		tick.set_fontweight('bold')
	for spine in ax3[1,i].spines.values():
		spine.set_linewidth(2)
	for lbl in lbl:
		lbl.set_fontproperties(bold_font)

	ax3[1,i].set_xlabel('x',fontweight='bold')


	ax3[1,i].set_xlim(np.min(x[1:-1]),np.max(x[1:-1]))
	ax3[1,i].set_ylim(np.min(y[1:-1]),np.max(y[1:-1]))




ax[0,0].set_ylabel(r'y', fontweight="bold")
ax[1,0].set_ylabel(r'y', fontweight="bold")


ax2[0].set_ylabel('y', fontweight="bold")


ax3[0,0].set_ylabel('y',fontweight='bold')
ax3[1,0].set_ylabel('y',fontweight='bold')


for i in range(1,len(timesteps)):
	ax[0,i].tick_params(top=True,right=True,labelbottom=False,labelleft=False,direction='in',size=4,width=1)
	ax[1,i].tick_params(top=True,right=True,labelbottom=True,labelleft=False,direction='in',size=4,width=1)

	ax3[0,i].tick_params(top=True,right=True,labelbottom=False,labelleft=False,direction='in',size=4,width=1)
	ax3[1,i].tick_params(top=True,right=True,labelbottom=True,labelleft=False,direction='in',size=4,width=1)


ax3[0,0].tick_params(left=True,bottom=False)
ax3[1,0].tick_params(left=True)




###########################
# decomp

fig, ax = plt.subplots(1, 1)
import matplotlib.cm as cm
from matplotlib.ticker import LogLocator
import matplotlib.colors as mcolors
from matplotlib import colormaps

cmap = colormaps.get_cmap('plasma').resampled(len(k_vals)-1)

for i in range(1, len(k_vals)):
	color = cmap(i-1)
	ax.plot(timesteps,A_bar_k[i, :], color=color)

ax.set_yscale('log')

ax.tick_params(axis='y', which='both', direction='in', length=4, width=1, color='k', labelcolor='k')
ax.tick_params(top=True, right=True, direction='in', length=4, width=1)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
ax.yaxis.set_ticks_position('both')

ax.set_xlabel('Time', fontweight='bold')
ax.set_ylabel(r'Amplitude', fontweight='bold')

for tick in ax.get_yticklabels() + ax.get_xticklabels():
	tick.set_fontweight('bold')

for spine in ax.spines.values():
	spine.set_linewidth(2)

ax.set_xlim(0, timesteps[-1])
ax.set_title(r'ψ-decomposition : ' + str(len(k_vals)-1) + ' modes', fontweight='bold')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=len(k_vals)-1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r'Mode index', fontweight='bold')



cbar.ax.yaxis.set_ticks_position('both')
cbar.ax.yaxis.set_tick_params(labelleft=False,
                               direction='in',
                               length=2,width=1)

for spine in cbar.ax.spines.values():
	spine.set_linewidth(1.5)

for tick in cbar.ax.get_yticklabels():
	tick.set_fontweight('bold')

for spine in cbar.ax.spines.values():
	spine.set_linewidth(1.5)

# Log scale for y-axis
ax.set_yscale('log')

# Tick settings: major and minor ticks "in"
ax.tick_params(axis='y', which='both', direction='in', length=4, width=1, color='k', labelcolor='k')
ax.tick_params(top=True, right=True, direction='in', length=4, width=1)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
ax.yaxis.set_ticks_position('both')  # Ticks on both left and right








fig, ax = plt.subplots(1, 1)

cmap = colormaps.get_cmap('plasma').resampled(len(k_vals)-1)


for i in range(1, len(k_vals)):
	color = cmap(i-1)
	ax.plot(timesteps,phi_bar_k[i, :], color=color)


ax.tick_params(axis='y', which='both', direction='in', length=4, width=1, color='k', labelcolor='k')
ax.tick_params(top=True, right=True, direction='in', length=4, width=1)
ax.set_xlabel('Time', fontweight='bold')
ax.set_ylabel(r'Phase', fontweight='bold')



for tick in ax.get_yticklabels() + ax.get_xticklabels():
	tick.set_fontweight('bold')
for spine in ax.spines.values():
	spine.set_linewidth(2)

ax.set_title(r'ψ-decomposition : ' + str(len(k_vals)-1) + ' modes', fontweight='bold')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=len(k_vals)-1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r'Mode index', fontweight='bold')
cbar.ax.yaxis.set_ticks_position('both')
cbar.ax.yaxis.set_tick_params(labelleft=False,
                               direction='in',
                               length=2,width=1)
for spine in cbar.ax.spines.values():
	spine.set_linewidth(1.5)
for tick in cbar.ax.get_yticklabels():
	tick.set_fontweight('bold')

for spine in cbar.ax.spines.values():
	spine.set_linewidth(1.5)


# Log scale for y-axis
ax.set_yscale('log')

# Tick settings: major and minor ticks "in"
ax.tick_params(axis='y', which='both', direction='in', length=4, width=1, color='k', labelcolor='k')
ax.tick_params(top=True, right=True, direction='in', length=4, width=1)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
ax.yaxis.set_ticks_position('both')  # Ticks on both left and right




plt.show()
