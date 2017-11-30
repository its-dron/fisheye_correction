#####################################################
# Create sample images with known distortion models #
#####################################################
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import ipdb as pdb

IMAGES_DIR = 'images'

# Image of vertical stripes
h,w = (1000,1000)
stripe_w = 20

stripes = np.zeros((h,w))

mask = np.int_(np.arange(w) / stripe_w) % 2
idx = np.where(mask)
stripes[:, idx] = 255

cv2.imwrite(os.path.join(IMAGES_DIR, 'stripes_input.png'), stripes)

# Distortion Params
x_0 = w//2
y_0 = h//2
focal_length = 1000 # Just an arbitrary decent value for a (1000,1000) working image

# Construct Camera Intrinsic Matrix
K = np.array(
    [[focal_length,            0,     x_0],
     [           0, focal_length,     y_0],
     [           0,            0,       1]])
d = np.array([0., 0., 0., 0.]) # distortion coefficients that will later have to be optimized over

# This does some magic to have the output be a good size
K_inv = np.linalg.inv(K)

# Get pixel coordinates
X,Y = np.meshgrid(range(w),range(h))

# Get ideal image coordinates my multiplying by K_inv
pts = np.vstack((X.flatten(), Y.flatten(), np.ones(h*w))) # homo coordinates of size (3,n_pixels)
pts_c = np.dot(K_inv, pts)
#pts_c /= pts_c[2,:]
x_c = pts_c[0,:]
y_c = pts_c[1,:]

# Compute distortion amount
r = np.sqrt(x_c**2 + y_c**2)
th = np.arctan(r) # th is shape (n_pixels,)

th_2 = th*th
th_4 = th_2*th_2
th_6 = th_4*th_2
th_8 = th_6*th_2
th_d = th * (1 + d[0]*th_2 + d[1]*th_4 + d[2]*th_6 + d[3]*th_8)

# These apply undistortion
#scale = r
#scale[np.abs(scale) < 1e-5] = 1
#scale = th_d / scale

# These apply distortion
scale = th_d
scale[np.abs(scale) < 1e-5] = 1
scale = r / scale

# Reshape distortion scalar
scale = scale.reshape(h,w)

x_p = (X - x_0) * scale + K[0,2]
y_p = (Y - y_0) * scale + K[1,2]

# Convert to the right dtype for remap
m1 = x_p.astype(np.float32)
m2 = y_p.astype(np.float32)

distorted_im = cv2.remap(stripes,
                         m1,
                         m2,
                         interpolation=cv2.INTER_LINEAR)

cv2.imwrite(os.path.join(IMAGES_DIR, 'stripes_distorted.png'), distorted_im)

# Some debugging plots
#plt.scatter(x_p, y_p, marker='.')
#plt.axis('equal')
#plt.show()
plt.imshow(distorted_im)
plt.axis('equal')
plt.show()
