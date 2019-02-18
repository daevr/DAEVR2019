import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import pywt
def add_noise(img, sigma):
    newimg = img.copy()
    noise = (np.random.normal(0, sigma, [img.shape[0], img.shape[1]]))
    return newimg+noise

image = np.array(Image.open('lenna.png').convert('L'))
image_noise = add_noise(image, 20)
image_noise = np.clip(image_noise,0,255)
Image.fromarray(image_noise).convert('RGB').save('noise.png')
image_noise = image_noise/255
rs = []
f_temp = np.fft.fft2(image_noise)
freq = np.fft.fftfreq(512,0.01)
f_temp = np.fft.fftshift(f_temp)
freq = np.fft.fftshift(freq)
for k in range(75,125,5):
    f = f_temp.copy()
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if np.sqrt(np.square(i - 255) + np.square(j - 255)) > k:
                f[i, j] = 0
    f = np.fft.ifftshift(f)
    image_new = np.abs(np.fft.ifft2(f))
    image_new = image_new * 255
    psnr = np.mean(np.square(image-image_new))
    psnr = 10*np.log10(255*255/psnr)
    rs.append(psnr)
print(rs)
x = np.arange(75,125,5)
x = x + 255
x = freq[x]
plt.plot(x,rs)
plt.xlabel('Hz')
plt.ylabel('PSNR')
plt.show()

Image.fromarray(image_new).convert('RGB').save('denoise.png')
print('finish')