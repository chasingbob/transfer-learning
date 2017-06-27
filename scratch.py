import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color, io, exposure
from scipy.misc import imresize

image_size=224
img = io.imread('images/grace.jpg')
new_img = imresize(img, (image_size, image_size, 3))
new_img = np.array(new_img) / 255.

flip_img = np.fliplr(new_img)
contrast_img_low = exposure.rescale_intensity(new_img, in_range=(0.1, 0.5))
contrast_img_high = exposure.rescale_intensity(new_img, in_range=(0.5, 0.9))

print('new: {} {}'.format(new_img.min(), new_img.max()))

print(new_img.shape)

plt.subplot(4, 1, 1)
plt.imshow(new_img)
plt.subplot(4, 1, 2)
plt.imshow(flip_img)
plt.subplot(4, 1, 3)
plt.imshow(contrast_img_low)
plt.subplot(4, 1, 4)
plt.imshow(contrast_img_high)
plt.show()
