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

import ipdb as pdb

IMAGES_DIR = 'images'
WORKING_RES = (1000,1000)
#INPUT_IM_PATH = os.path.join(IMAGES_DIR, 'smirkle.png')
#INPUT_IM_PATH = os.path.join(IMAGES_DIR, 'stripes_distorted.png')
INPUT_IM_PATH = os.path.join(IMAGES_DIR, 'stripes_input.png')
EDGEY_RATIO = 2
N_THETA = 180

def objective_fn(vars_to_optimize, im_edge):
    '''
    To be called:
        scipy.optimize.minimize(objective_fn, initial_guess,  args=(im_edge), method='Nelder-Mead')
    '''

    # 1. Undistort im_edge by vars_to_optimize
    # 2. Compute cost of newly undistorted
    return cost(est_h_1d)

def cost(h_1d):
    '''
    Compute the 1D hough entropy
    '''
    h_1d /= np.sum(h_1d) # Normalize the 1D hough
    c = -np.sum(h_1d * np.log2(h_1d))
    return c


def hough_entropy(edges, n_theta=10):
    # Classic straight-line Hough transform
    theta = np.linspace(-np.pi/2, np.pi/2, n_theta);
    h,theta,d = hough_line(edges, theta=theta)

    plt.imshow(np.log(1+h),
            extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
            cmap=cm.gray)
    plt.title('Hough Space')
    plt.axis('equal')
    plt.xlabel('Angle')
    plt.ylabel('Radius')
    plt.show()

    h_1d = np.sum(h, axis=0)

    return h_1d


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

    edge_sal, eigvec = computeEdgeSaliency(im, edge_ratio = EDGEY_RATIO)

    #plt.imshow(edge_sal.reshape([h,w]))
    #plt.title('Edge Saliency')
    #plt.axis('equal')
    #plt.show()

    # TODO Perform Edgel Subsampling

    # Get orientation
    orient = np.arctan2(eigvec[:,0], eigvec[:,1])
    orient %= np.pi

    #plt.imshow(orient.reshape([h,w]))
    #plt.title('Orientation')
    #plt.axis('equal')
    #plt.show()

    #orient_hist, edges = np.histogram(orient, bins=N_THETA)
    #orient_hist /= np.sum(orient_hist)

    #theta_bins = np.linspace(-np.pi/2, np.pi/2, N_THETA);
    theta_bins = np.linspace(-np.pi, np.pi, N_THETA+1);
    bin_ids = np.digitize(orient, theta_bins, right=True)

    hist = np.zeros(N_THETA)
    for i in range(N_THETA):
        hist[i] = np.sum((bin_ids == i) * (edge_sal > 0))
        #hist[i] = np.sum(edge_sal[bin_ids == i])


    hist /= np.sum(hist)

    center = (theta_bins[:-1] + theta_bins[1:]) / 2
    width = 0.7*(theta_bins[1] - theta_bins[0])

    plt.bar(center, hist, align='center', width=width)
    plt.show()
    return

    h_1d = hough_entropy(edge_sal, N_THETA)
    print(h_1d)

    return

    plt.figure()
    plt.plot(h_1d)
    plt.show()

if __name__=='__main__':
    main()
