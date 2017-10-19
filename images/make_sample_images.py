#####################################################
# Create sample images with known distortion models #
#####################################################
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Image of vertical stripes
h,w = (1000,1000)
stripe_w = 20

stripes = np.zeros((h,w))

mask = np.int_(np.arange(w) / stripe_w) % 2
idx = np.where(mask)
stripes[:, idx] = 255

cv2.imwrite('stripes_input.png', stripes)

# Distortion Params
x_0 = w//2
y_0 = h//2
K = np.eye(3)
K[0,0] = x_0
K[1,1] = y_0
K[0,2] = x_0
K[1,2] = y_0
d = np.array([0., 0., 0., 0.])

# This does some magic to have the output be a good size
Knew, valid_roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 1)
K_inv = np.linalg.inv(K)

# Get pixel coordinates
X,Y = np.meshgrid(range(w),range(h))

# Get ideal image coordinates my multiplying by K_inv
pts = np.vstack((X.flatten(), Y.flatten(), np.ones(h*w)))
pts_c = np.dot(K_inv, pts)
x_c = pts_c[0,:]
y_c = pts_c[1,:]

# Compute distortion amount
r = np.sqrt(x_c**2 + y_c**2)
th = np.arctan(r)

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

# This is equivilant to K_new * diag(scale,scale,1) * K_inv * [x,y,1]
x_p = Knew[0,0] / K[0,0] * (X - x_0) * scale + Knew[0,2]
y_p = Knew[1,1] / K[1,1] * (Y - y_0) * scale + Knew[1,2]

# Convert to the right dtype for remap
m1 = x_p.astype(np.float32)
m2 = y_p.astype(np.float32)


distorted_im = cv2.remap(stripes, 
                         m1,
                         m2,
                         interpolation=cv2.INTER_LINEAR)

cv2.imwrite('stripes_distorted.png', distorted_im)

# Some debugging plots
#plt.scatter(x_p, y_p, marker='.')
#plt.axis('equal')
#plt.show()
plt.imshow(distorted_im)
plt.axis('equal')
plt.show()