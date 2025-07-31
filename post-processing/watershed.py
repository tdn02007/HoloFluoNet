import os
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import dilation, square, erosion

file_name = "HoloFuloNet"
data_type = ["live", "dead"]

data_dir = f'../result_mask_{file_name}/split_data'
data_list = os.listdir(data_dir)

for item in data_list:
    for ty in data_type:
        boundary_dir = sorted(os.listdir(f"{data_dir}/{ty}/"))
        marker_dir = sorted(os.listdir(f"{data_dir}/distance/"))

        for i, f in enumerate(boundary_dir):
            img2 = cv2.imread(f"{data_dir}/{ty}/" + f)
            img3 = cv2.imread(f"{data_dir}/distance/" + marker_dir[i])

            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(gray2,125,255,cv2.THRESH_OTSU)
            sure_bg = thresh
            kernel = np.ones((3,3),np.uint8)
            sure_bg = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

            binary = dilation(sure_bg, square(3))
            boundary = sure_bg - binary

            ret, sure_fg = cv2.threshold(gray3,0.4*gray3.max(),255,0)
            sure_fg = dilation(sure_fg, square(3))
            sure_fg = erosion(sure_fg, square(3))
            sure_fg = np.uint8(sure_fg)

            unknown = cv2.subtract(sure_bg, sure_fg)
            unknown[boundary > 0] = 255

            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

            marker_results = markers

            img4 = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)

            markers = cv2.watershed(img4, markers)
            
            data = np.zeros_like(img2)
            
            if ty == "live":
                data[sure_bg == 255] = [0, 0, 255]
            else:
                data[sure_bg == 255] = [0, 255, 0]
            
            data[markers == -1] = [255, 255, 255]

            data = dilation(data, square(3))

            final_img = Image.fromarray(data.astype(np.uint8))

            os.makedirs(f"{data_dir}/watershed_{ty}_marker_results/", exist_ok=True)

            final_img.save(f"{data_dir}/watershed_{ty}_marker_results/{f}")

