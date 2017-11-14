############################
# Compute an image's edgel #
############################
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.feature import structure_tensor, structure_tensor_eigvals

import ipdb as pdb

IMAGES_DIR = 'images'
WORKING_RES = (640,640)
INPUT_IM_PATH = os.path.join(IMAGES_DIR, 'stripes_distorted.png')
EDGEY_RATIO = 2

def main():
    im = cv2.imread(INPUT_IM_PATH, 0) # Read image as grayscale
    im = cv2.resize(im, WORKING_RES)

    # compute gradient images
    st = structure_tensor(im)
    eig_max, eig_min = structure_tensor_eigvals(*st)

    phi = eig_max - (EDGEY_RATIO * eig_min)
    res_im = phi
    res_im[res_im<0] = 0

    plt.imshow(res_im)
    plt.axis('equal')
    plt.show()


if __name__=='__main__':
    main()
