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
#    that you want. Don't forget to replace the
#    initial value !

# 3) Change the variable after the for var in range ...

# 4) Change the position of the variable inside the output
#    of the function : compute_sigmas

# 5) Change the name of : save_png (LaTeX format capable)




var_title = 'Ly'
config = 'conf_2'


Ny, Nk = 60, 51
dk = 0.1
ymin, kmin, Ly, Lk = 0.1, 0.1, np.round(np.linspace(0.1, 4, 15), 3) , 0.1+dk*Nk

beta = 0
F1star = 0
U0 = 1

Theta0_U0 = 1


# Ensure output dir exists
os.makedirs('output', exist_ok=True)

# List to store image bytes for the GIF
frames = []

for var in Ly:
    Un, G12, fig, (ax) = compute_sigmas(Ny, Nk, dk, ymin, kmin, var, Lk, beta, F1star, U0, Theta0_U0, config)
    
    save_png = r'$L_y =$'+str(var)
    ax.set_title(save_png)
    
    # Save figure to memory buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image = imageio.imread(buf)
    frames.append(image)
    buf.close()
    plt.close(fig)



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
    file.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')







y_l = np.linspace(ymin,Ly,Ny)


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
    
    

ax2[1].plot(Un*G12,y_l,'k',label=r'$\overline{U}_n.\frac{\mathrm{d}\overline{\Theta}}{\mathrm{d}y}$')

for i in range(y_l.shape[1]):
	ax2[1].plot(0.25*(Un*y_l[:,i])**2,y_l[:,i],'k--',label=r'$\frac{1}{4}.\left(\overline{U}_n.y\right)^2$')
	#ax2[1].fill_betweenx(y_l,crit,borne,color='orange',alpha=0.3)
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




print('Variables saved')

