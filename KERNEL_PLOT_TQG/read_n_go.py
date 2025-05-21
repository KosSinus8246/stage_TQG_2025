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
frames_pil[0].save('output/sigmas_animation.gif', save_all=True, append_images=frames_pil[1:], duration=600, loop=0)


print("GIF saved to: output/sigmas_animation.gif")


# Create full file path
file_path = os.path.join('output/', 'variables_used.txt')

# Open a file in write mode
with open('output/variables_used.txt', 'w') as file:
    file.write('Used variables :\n')
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
    #file.write(f"Rd = {Rd}\n")
    file.write(f"U0 = {U0}\n")
    #file.write(f"Theta0 = {Theta0}\n")
    #file.write(f"ratio_Theta0_U0 = {Theta0_U0}\n")
    file.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

print('Variables stored into : output/variables_used.txt')


