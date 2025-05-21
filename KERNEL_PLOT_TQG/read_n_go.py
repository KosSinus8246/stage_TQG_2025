'''from KERNEL_TQG_solve_v2_bis import *


Ny, Nk = 60, 51
dk = 0.1
ymin, kmin, Ly, Lk = 0.1, 0.1, np.pi, 0.1+dk*Nk


beta = 0 
F1star = 0 # 1/Rd**2

U0 = 1


# compute_sigmas(False, Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, 1)
# compute_sigmas(False, Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, 0.1)

Theta0_U0 = np.round(np.linspace(0., 1., 20),3)  # 5 values between 0.1 and 1.0

for theta in Theta0_U0:
    compute_sigmas(str(theta), Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, theta)
    plt.savefig('output/fig_ratio_'+str(theta)+'.png',dpi=300)

# vider la mémoire avec show et tout refermer
plt.show(block=False)
plt.close('all')'''


from KERNEL_TQG_solve_v2_bis import *
import imageio.v2 as imageio
import os
from io import BytesIO
from PIL import Image

Ny, Nk = 60, 51
dk = 0.1
ymin, kmin, Ly, Lk = 0.1, 0.1, np.pi, 0.1+dk*Nk

beta = 0 
F1star = 0
U0 = 1

Theta0_U0 = np.round(np.linspace(0., 1., 20), 3)

# Ensure output dir exists
os.makedirs('output', exist_ok=True)

# List to store image bytes for the GIF
frames = []

for theta in Theta0_U0:
    fig = compute_sigmas(r'$\Theta_0/U_0 =$'+str(theta), Ny, Nk, dk, ymin, kmin, Ly, Lk, beta, F1star, U0, theta)
    
    # Save figure to memory buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image = imageio.imread(buf)
    frames.append(image)
    buf.close()
    plt.close(fig)



frames_pil = [Image.fromarray(frame) for frame in frames]
frames_pil[0].save('output/sigmas_animation_pillow.gif', save_all=True, append_images=frames_pil[1:], duration=600, loop=0)


print("GIF saved to: output/sigmas_animation_pillow.gif")

