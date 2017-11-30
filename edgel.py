############################
# Compute an image's edgel #
############################
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.feature import structure_tensor, structure_tensor_eigvals
from skimage.transform import hough_line, hough_line_peaks
from scipy.optimize import minimize
from matplotlib import cm

import fisheye_tools as ft
import ipdb as pdb

IMAGES_DIR = 'images'
WORKING_RES = (1000,1000)
FOCAL_LENGTH = 1000 # Starting focal length guess
EDGEY_RATIO = 2
N_THETA = 180
eps = 1e-6

#INPUT_IM_PATH = os.path.join(IMAGES_DIR, 'smirkle.png')
INPUT_IM_PATH = os.path.join(IMAGES_DIR, 'stripes_distorted.png')
#INPUT_IM_PATH = os.path.join(IMAGES_DIR, 'stripes_input.png')

def objective_fn(fd, im):
    '''
    To be called:
        scipy.optimize.minimize(objective_fn, initial_guess,  args=(im_edge), method='Nelder-Mead')
    '''
    corrected_im = apply_fd(fd,im)
    h = hough_1d(corrected_im, n_theta=N_THETA)
    c = cost(h)
    print("Cost: %s" % c)
    return c

def apply_fd(fd, im):
    f = fd[0]
    d = fd[1:]
    h,w = im.shape[0:2]
    x_0 = w // 2; y_0 = h // 2
    K = ft.construct_K(f, f, x_0, y_0)
    pts_c = ft.calibrated_im_coor(K, im.shape[0:2])
    r, th_d = ft.get_distortion_amount(pts_c, d)

    # undistort the fisheye image
    scale = ft.construct_scale(th_d, r, im.shape[0:2])
    corrected_im = ft.apply_scale(im, scale, K)
    return corrected_im

def cost(h_1d):
    '''
    Compute the 1D hough entropy
    '''
    h_1d /= np.sum(h_1d) # Normalize the 1D hough
    c = -np.sum(h_1d * np.log2(h_1d + eps))
    return c

def hough_1d(im, n_theta=N_THETA):
    '''
    Takes in a grayscale image, and computes the 1D hough histogram of
    the computed edgel image
    '''
    edge_sal, eigvec = computeEdgeSaliency(im, edge_ratio = EDGEY_RATIO)

    # Get orientation
    orient = np.arctan2(eigvec[:,0], eigvec[:,1])
    orient %= np.pi

    theta_bins = np.linspace(-np.pi, np.pi, N_THETA+1);
    bin_ids = np.digitize(orient, theta_bins, right=True)

    hist = np.zeros(N_THETA)
    for i in range(N_THETA):
        hist[i] = np.sum((bin_ids == i) * (edge_sal > 0))

    hist /= np.sum(hist)
    return hist

def computeEdgeSaliency(im, sigma=1, edge_ratio=2):
    """
    Compute the edge saliency of the grayscale image `im`.

    Inputs:
        im         - grayscale image
        edge_ratio - minimum ratio of max to min eigenvlue to count as edge

    Returns:
        edge_sal   - edge saliency image. Same shape as `im`
        eig_vec    - vector of smaller eigen vectors. in [y,x] format.
    """
    h,w = im.shape

    # compute gradient images
    Ixx, Ixy, Iyy = structure_tensor(im, sigma=1.414, mode='reflect')
    data = np.array([[Ixx.flatten(), Ixy.flatten()], [Ixy.flatten(), Iyy.flatten()]])
    data = data.transpose()

    lamb, v = np.linalg.eig(data)

    # Sort ascending by eigenvalue
    sort_idx = np.argsort(lamb, axis=1)
    i = np.arange(im.size).reshape([-1,1])
    lamb = lamb[i, sort_idx]
    v = v[i, sort_idx]

    small_eigvec = v[:,:,0]

    # max_eig - e*min_eig
    phi = lamb[:,1] - (edge_ratio * lamb[:,0])
    edge_sal = phi#.reshape([h,w])
    edge_sal[edge_sal<0] = 0

    return edge_sal, small_eigvec

def main():
    im = cv2.imread(INPUT_IM_PATH, 0) # Read image as grayscale
    im = cv2.resize(im, WORKING_RES)
    h,w = im.shape
    ft.plot(im, "Input Image")

    # distortion coefficients that will later have to be optimized over
    # Because we also need to optimize the focal length, it is included
    df = np.array([FOCAL_LENGTH, 0., 0., 0., 0.])

    # Note that we also have to optimize FOCAL_LENGTH
    optim_fd = minimize(objective_fn, df,  args=(im), method='Nelder-Mead',
            options={'xtol': 0.005, 'disp': True})
    corrected_im = apply_fd(optim_fd.x, im)
    ft.plot(corrected_im, "Optimal Image")
    print(optim_fd.x)
    pdb.set_trace()


if __name__=='__main__':
    main()
