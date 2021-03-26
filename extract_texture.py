from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import cv2
import os

# settings for LBP
radius = 1
n_points = 8 * radius

image_path = './super-resolved_images/'
images = os.listdir(image_path)
for image_name in images:
    image = cv2.imread(image_path+image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image, n_points, radius)

    save_path = './lbp_images/'
    plt.imsave(save_path+image_name[:-4]+'.jpg', lbp, cmap='gray')
