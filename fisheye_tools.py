#############################################
# Test bed for warping and unwarping images #
#############################################
'''
a more modularized version of make_sample_images.py
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMAGES_DIR = 'images'
WORKING_RES = (1000,1000)
FOCAL_LENGTH = 789
STRIPE_WIDTH = 20

def create_stripe_image(im_size=WORKING_RES, stripe_w=STRIPE_WIDTH):
    '''
    Creates a vertical striped image of size im_size.
    Each stripe will have width stripe_w
    '''
    stripes = np.zeros(im_size)

    mask = np.int_(np.arange(im_size[1]) / stripe_w) % 2
    idx = np.where(mask)
    stripes[:, idx] = 255
    return stripes

def construct_K(f_x, f_y, x_0, y_0):
    # Construct Camera Intrinsic Matrix
    K = np.array(
        [[ f_x,   0, x_0],
         [   0, f_y, y_0],
         [   0,   0,   1]])
    return K

def distort_im():
    pass

def plot(im, title='', axis='equal'):
    ''' Thin wrapper for plotting because I'm petty '''
    plt.imshow(im); plt.axis(axis); plt.title(title); plt.show()

def calibrated_im_coor(K, im_size):
    '''
    Creates Homogenous coordinates of all pixels in im_size

    Return of size (3, n_pixels)
    '''
    # Get pixel coordinates
    h,w = im_size
    X,Y = np.meshgrid(range(w),range(h))

    # Get ideal image coordinates my multiplying by K_inv
    pts = np.vstack((X.flatten(), Y.flatten(), np.ones(h*w))) # homo coordinates of size (3,n_pixels)
    pts_c = np.dot(np.linalg.inv(K), pts)
    return pts_c

def get_distortion_amount(pts_c, d):
    '''
    Gets the amount to distort each pixel
    '''
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

    return r, th_d

def construct_scale(num, den, im_size, thresh=1e-5):
    scale = den
    scale[np.abs(scale)<thresh] = 1
    scale = num / scale
    scale = scale.reshape(*im_size)
    return scale

def apply_scale(im, scale, K):
    '''
    Apply scale (distort/undistort) to im
    '''
    h,w = im.shape[0:2]
    x_0 = K[0,2]; y_0 = K[1,2]

    X,Y = np.meshgrid(range(w),range(h))

    x_p = (X - x_0) * scale + x_0
    y_p = (Y - y_0) * scale + y_0

    warped_im = cv2.remap(im,
            x_p.astype(np.float32), y_p.astype(np.float32),
            interpolation=cv2.INTER_LINEAR)
    return warped_im

def main():
    '''
    Demos the distorting and undistorting of an image
    '''
    # Center Principal Point Assumption
    x_0 = WORKING_RES[1] // 2
    y_0 = WORKING_RES[0] // 2

    stripe_im = create_stripe_image()
    plot(stripe_im, "Input Stripe Image")

    # distortion coefficients that will later have to be optimized over
    d = np.array([0., 0., 0., 0.])

    K = construct_K(FOCAL_LENGTH, FOCAL_LENGTH, x_0, y_0)
    pts_c = calibrated_im_coor(K, WORKING_RES)
    r, th_d = get_distortion_amount(pts_c, d)

    # distort image
    scale = construct_scale(r, th_d, WORKING_RES)
    fisheye_im = apply_scale(stripe_im, scale, K)
    plot(fisheye_im, "Fisheye Image")

    # undistort the fisheye image
    scale = construct_scale(th_d, r, WORKING_RES)
    corrected_im = apply_scale(fisheye_im, scale, K)
    plot(corrected_im, "Corrected Fisheye Image")

if __name__=="__main__":
    main()
