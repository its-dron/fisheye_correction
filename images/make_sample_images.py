#####################################################
# Create sample images with known distortion models #
#####################################################
import numpy as np
import cv2


# Image of vertical stripes
h,w = (1000,1000)
n_stripes=10

stripes = np.zeros((h,w))

mask = np.int_(np.arange(w) / n_stripes) % 2
idx = np.where(mask)
stripes[:, idx] = 255


cv2.imwrite('stripes_input.png', stripes)

X,Y = np.meshgrid(range(w),range(h))
pts = np.vstack((X.reshape(1,-1), Y.reshape(1,-1)))

pts = pts.astype('float32')
pts = pts.reshape([1, h*w, 2])

# distort points
K = np.eye(3)
d = np.zeros(4)

distorted_pts = cv2.fisheye.distortPoints(pts, K, d)

x_dist = distorted_pts[0,:,0]
y_dist = distorted_pts[0,:,1]
distorted_im = cv2.remap(stripes, x_dist, y_dist, interpolation=cv2.INTER_LINEAR)

cv2.imwrite('stripes.png', distorted_im)

print distorted
print distorted.shape
