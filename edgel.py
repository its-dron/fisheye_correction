############################
# Compute an image's edgel #
############################
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.feature import structure_tensor, structure_tensor_eigvals
from skimage.transform import hough_line, hough_line_peaks

import ipdb as pdb

IMAGES_DIR = 'images'
WORKING_RES = (1000,1000)
INPUT_IM_PATH = os.path.join(IMAGES_DIR, 'stripes_distorted.png')
EDGEY_RATIO = 2
N_THETA = 180

def objective_fn(h_1d):
    '''
    Compute the 1D hough entropy
    '''
    h_1d /= np.sum(h_1d) # Normalize the 1D hough
    c = -np.sum(h_1d * np.log2(h_1d))
    return c

def main():
    im = cv2.imread(INPUT_IM_PATH, 0) # Read image as grayscale
    im = cv2.resize(im, WORKING_RES)

    # compute gradient images
    st = structure_tensor(im)
    eig_max, eig_min = structure_tensor_eigvals(*st)

    phi = eig_max - (EDGEY_RATIO * eig_min)
    edge_sal = phi
    edge_sal[edge_sal<0] = 0

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
