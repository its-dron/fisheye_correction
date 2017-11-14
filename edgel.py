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

import ipdb as pdb

IMAGES_DIR = 'images'
WORKING_RES = (1000,1000)
INPUT_IM_PATH = os.path.join(IMAGES_DIR, 'stripes_distorted.png')
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


def hough_entropy(edges):
    pass


def computeEdgeSaliency(im, edge_ratio=2):
    """
    Compute the edge saliency of the grayscale image `im`.

    Inputs:
        im         - grayscale image
        edge_ratio - minimum ratio of max to min eigenvlue to count as edge

    Returns:
        edgels     - edge saliency image. Same shape as `im`
    """

    # compute gradient images
    st = structure_tensor(im)
    eig_max, eig_min = structure_tensor_eigvals(*st)

    phi = eig_max - (edge_ratio * eig_min)
    edge_sal = phi
    edge_sal[edge_sal<0] = 0

    return edge_sal


def main():
    im = cv2.imread(INPUT_IM_PATH, 0) # Read image as grayscale
    im = cv2.resize(im, WORKING_RES)

    edge_sal = computeEdgeSaliency(im, edge_ratio = EDGEY_RATIO)

    #plt.imshow(edge_sal)
    #plt.title('Edge Saliency')
    #plt.axis('equal')
    #plt.show()

    # TODO Perform Edgel Subsampling

    # Classic straight-line Hough transform
    theta = np.linspace(-np.pi/2, np.pi/2, N_THETA);
    h, _, d = hough_line(edge_sal, theta=theta)

    plt.imshow(h)
    plt.title('Classic Hough (Intensity = Votes)')
    plt.xlabel('Angle')
    plt.ylabel('Radial Distance')
    plt.axes().set_aspect('equal', adjustable='box')
    plt.show()

    h_1d = np.sum(h, axis=0)
    return

    plt.figure()
    plt.plot(h_1d)
    plt.show()

if __name__=='__main__':
    main()
