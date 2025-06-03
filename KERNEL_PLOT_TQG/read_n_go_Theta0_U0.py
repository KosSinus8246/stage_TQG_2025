from KERNEL_TQG_solve_v2_bis import *
import imageio.v2 as imageio
import os
from io import BytesIO
from PIL import Image

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~TQG_SOLVE_2_BIS~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-----------------------------------------------------')


#########################
# To change the variable in the loop

# 1) Change the name of : var_title

# 2) Put a linspace or an arange in your parameter
#    that you want

# 3) Change the variable after the for var in range ...

# 4) Change the position of the variable inside the output
#    of the function : compute_sigmas

# 5) Change the name of : save_png (LaTeX format capable)




var_title = 'Theta0_U0'
config = 'conf_2'



Ny, Nk = 60, 51
dk = 0.1
ymin, kmin, Ly, Lk = 0.1, 0.1, np.pi, 0.1+dk*Nk
Lstar = 0.1

beta = 0 
F1star = 0
U0 = 1

Theta0_U0 = np.round(np.linspace(0., 2., 15), 3)



# stack maximas into a matrix
max_sigmas = np.zeros_like(Theta0_U0)
max_sigmas_NT = np.zeros_like(Theta0_U0)



# Ensure output dir exists
os.makedirs('output', exist_ok=True)

# List to store image bytes for the GIF
frames = []

i = 0

for var in Theta0_U0:
    Un, G12, fig, (ax), max_sigma, max_sigmaNT = compute_sigmas(Ny, Nk, dk, ymin, kmin, Ly, Lk, Lstar, beta, F1star, U0, var,config)
    
    save_png = r'$\Theta_0/U_0 =$'+str(var)
    ax.set_title(save_png)
    
    # Save figure to memory buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image = imageio.imread(buf)
    frames.append(image)
    buf.close()
    plt.close(fig)
    
    max_sigmas[i] = max_sigma
    max_sigmas_NT[i] = max_sigmaNT
    
    i = i+1



frames_pil = [Image.fromarray(frame) for frame in frames]
frames_pil[0].save('output/sigmas_animation_'+var_title+'_'+config+'.gif', save_all=True, append_images=frames_pil[1:], duration=600, loop=0)


print('GIF saved')







# Create full file path
file_path = os.path.join('output/', 'variables_used_'+var_title+'_'+config+'.txt')

# Open a file in write mode
with open('output/variables_used_'+var_title+'_'+config+'.txt', 'w') as file:
    file.write('Used variables for : '+var_title+' '+config+'\n')
    file.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    file.write(f"Ny = {Ny}\n")
    file.write(f"Nk = {Nk}\n")
    file.write(f"ymin = {ymin}\n")
    file.write(f"Ly = {Ly}\n")
    file.write(f"kmin = {kmin}\n")
    file.write(f"Lk = {Lk}\n")
    dy = (Ly-ymin)/Ny
    file.write(f"dy = {dy}\n")
    file.write(f"dk = {dk}\n")
    file.write(f"F1star = {F1star}\n")
    file.write(f"beta = {beta}\n")
    file.write(f"U0 = {U0}\n")
    file.write(f"ratio_Theta0_U0 = {Theta0_U0}\n")
    file.write(f"Lstar = {Lstar}\n")
    file.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')



y_l = np.linspace(ymin,Ly,Ny)
crit, borne =Un*G12, 0.25*(Un*y_l)**2

fig2, (ax2) = plt.subplots(1,2, figsize = (15,6),sharey=True)

ax2[0].plot(Un,y_l,'b')
ax2[0].tick_params(top=True,right=True,direction='in',size=4,width=1)
ax2[0].axhline(0, color='gray', linestyle=':')
ax2[0].axvline(0, color='gray', linestyle=':')
#ax2.set_ylim(-0.01, 0.5)
ax2[0].set_title('Velocity profile')
ax2[0].set_ylabel(r'$y$')
ax2[0].set_xlabel(r'$\overline{U}_n$')
	
	
for spine in ax2[0].spines.values():
    spine.set_linewidth(2)
    
    

ax2[1].plot(crit,y_l,'k',label=r'$\overline{U}_n.\frac{\mathrm{d}\overline{\Theta}}{\mathrm{d}y}$')
ax2[1].plot(borne,y_l,'k--',label=r'$\frac{1}{4}.\left(\overline{U}_n.y\right)^2$')
ax2[1].fill_betweenx(y_l,crit,borne,color='orange',alpha=0.3)
ax2[1].tick_params(top=True,right=True,direction='in',size=4,width=1)

ax2[1].axhline(0, color='gray', linestyle=':')
ax2[1].axvline(0, color='gray', linestyle=':')
ax2[1].set_title('Stability')
ax2[1].legend(fancybox=False,loc='upper left')
ax2[1].set_xlabel(r'$\propto~\left(\overline{U}_n.y\right)^2$')
#ax2.set_ylim(-0.01, 0.5)
	
	
for spine in ax2[1].spines.values():
    spine.set_linewidth(2)


plt.savefig('output/stab_'+var_title+'_'+config+'.png',dpi=300)




# derivative of sigmas_max
deriv = np.zeros_like(max_sigmas)
deriv_NT = np.zeros_like(max_sigmas)

dTheta0_U0 = (Theta0_U0[-1] - Theta0_U0[0])/(len(Theta0_U0))



for i in range(len(deriv)-1):
	deriv[i] = (max_sigmas[i+1] - max_sigmas[-1])/dTheta0_U0
	deriv_NT[i] = (max_sigmas_NT[i+1] - max_sigmas_NT[-1])/dTheta0_U0




fig3, (ax3) = plt.subplots(1,1)

ax3.axhline(0, color='gray', linestyle=':')
ax3.axvline(0, color='gray', linestyle=':')

ax3.plot(Theta0_U0, deriv, 'k--',label='TQG')
ax3.plot(Theta0_U0, deriv_NT, 'k-',label='QG')
ax3.tick_params(top=True,right=True,direction='in',size=4,width=1)
ax3.set_xlabel(r'$\Theta_0/U_0$')
ax3.legend(fancybox=False,loc='lower right')

ax3.set_ylabel(r'$\mathrm{d}\sigma_\mathbf{max}/\mathrm{d}(\Theta_0/U_0)$')

ax3.set_title('Behaviour of $\sigma_\mathbf{max}$')


for spine in ax3.spines.values():
    spine.set_linewidth(2)


plt.tight_layout()

plt.savefig('output/deriv_'+var_title+'_'+config+'.png',dpi=100)





print('Variables saved')


