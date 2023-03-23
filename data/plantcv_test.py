import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from DataLoading import get_df
import random
from plantcv import plantcv as pcv
from skimage.morphology import square
import cv2 as cv
from scipy import ndimage







def run_single_rgb(img, with_images = False):
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s') #get the saturation value of the image
    v = pcv.rgb2gray_hsv(rgb_img=img, channel='v') #get the value value of the image

    s_binary = pcv.threshold.binary(gray_img=s, threshold=125, max_value=255, object_type='light')
    s_e = pcv.erode(s_binary, 3,5)
    s_e = pcv.erode(s_binary, 5,5)
    s_d = pcv.dilate(s_e, 3, 10)

    s_last = s_d

    #Find the region of interest ROI
    roi_cm = ndimage.center_of_mass(s_last)
    roi_w, roi_h = 600, 800
    roi_x, roi_y  = int(max(0,(roi_cm[1]-roi_w/2))), int(max(0,(roi_cm[0]-roi_h/2)))
    roi_w += int(min(0, (s_last.shape[1] - roi_cm[1]-roi_w/2))) #Only diminish height or widht if it hgets out of the image
    roi_h += int(min(0, (s_last.shape[0] - roi_cm[0]-roi_h/2)))

    # print(f"roiX {roi_x} roiY {roi_y}")

    id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=s_last)
    roi1, roi_hierarchy = pcv.roi.rectangle(img=s_last, x=roi_x, y=roi_y, h=roi_h, w=roi_w)
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi1,
                                                                   roi_hierarchy=roi_hierarchy,
                                                                   object_contour=id_objects,
                                                                   obj_hierarchy=obj_hierarchy,
                                                                   roi_type='partial')
    obj, mask = pcv.object_composition(img=img, contours=roi_objects, hierarchy=hierarchy)
    try:
        analysis_image = pcv.analyze_object(img=img, obj=obj, mask=mask, label="default")
    except TypeError: #sometimes object returns nothing and len of nonetype gives TypeError
        analysis_image = img

    if with_images:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(s_last, cmap="gray")
        ax[1].imshow(analysis_image)
        plt.show()
    try:
        highest_x = np.amax(obj[:,:,0])
        highest_y = np.amax(obj[:,:,1])
        lowest_x = np.amin(obj[:,:,0])
        lowest_y = np.amin(obj[:,:,1])
    except TypeError: #sometimes object returns nothing and nonetype is not subscriptable TypeError
        highest_x = 0
        highest_y = 0
        lowest_x = 0
        lowest_y = 0
        w = s_last.shape[0]
        for i in range(w):
            pixel_n = np.sum(s_last[i,:])
            if not pixel_n == 0:
                highest_x = i
                break
        for i in range(w):
            pixel_n = np.sum(s_last[(w-i-1),:])
            if not pixel_n == 0:
                lowest_x = (w-i)
                break
        h = s_last.shape[1]
        for i in range(w):
            pixel_n = np.sum(s_last[:,i])
            if not pixel_n == 0:
                highest_y = i
                break
        for i in range(w):
            pixel_n = np.sum(s_last[:,(h-i-1)])
            if not pixel_n == 0:
                lowest_y = (w-i)
                break



    plant_width = highest_x - lowest_x
    plant_height = highest_y - lowest_y
    plant_area = obj_area
    plant_to_image_ratio = plant_area / (1280*960)
    plant_area_1 = np.sum(s_last)
    plant_to_image_ratio_1 = plant_area_1 / (1280*960*255)

    return plant_width, plant_height, plant_area

def run_single_gray(img, with_images=False):
    """This function takes an image (the side view of aa tomato in grayscale.
        It then isolates the planet from the other elements and estimates the height and weight of the plant.
        It returnss these values."""
    binary = pcv.threshold.binary(gray_img=img, threshold=100, max_value=255, object_type='dark')
    #Use blurr to clean the noise
    s_mblur = pcv.median_blur(gray_img=binary, ksize=3)
    gaussian_blur = pcv.gaussian_blur(img=binary, ksize=(3, 3))

    opened_mask = pcv.opening(gray_img=gaussian_blur, kernel=square(5))
    closed_mask = pcv.closing(gray_img=gaussian_blur, kernel=square(5))
    opened_closed_mask = pcv.closing(gray_img=opened_mask, kernel=square(5))

    # pixels of the original image are only used if they overlap with the mask
    masked_o = pcv.apply_mask(img=img, mask=opened_mask, mask_color='white')
    masked_c = pcv.apply_mask(img=img, mask=closed_mask, mask_color='white')
    masked_o_c = pcv.apply_mask(img=img, mask=opened_closed_mask, mask_color='white')

    stump_e = pcv.erode(opened_closed_mask, 10,15)
    stump_d = pcv.dilate(stump_e, 10, 20)
    only_plant = opened_closed_mask - stump_d
    only_plant[only_plant < 0] = 0
    only_plant = pcv.erode(only_plant, 3,3)


    #Find the region of interest ROI
    roi_cm = ndimage.center_of_mass(only_plant)
    w, h = 600, 800
    roi = [(roi_cm[1]-w/2), (roi_cm[0]-h/2),(roi_cm[1]+w/2), (roi_cm[0]+h/2)]
    roi_w, roi_h = 600, 800
    roi_x, roi_y  = max(0,(roi_cm[1]-roi_w/2)), max(0,(roi_cm[0]-roi_h/2))
    roi_w += min(0, (binary.shape[1] - roi_cm[1]-roi_w/2)) #Only diminish height or widht if it hgets out of the image
    roi_h += min(0, (binary.shape[0] - roi_cm[0]-roi_h/2))


    cropped = img[:, int(roi[0]): int(roi[2])]

    id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=only_plant)
    roi1, roi_hierarchy = pcv.roi.rectangle(img=only_plant, x=roi_x, y=roi_y, h=roi_h, w=roi_w)
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi1,
                                                                   roi_hierarchy=roi_hierarchy,
                                                                   object_contour=id_objects,
                                                                   obj_hierarchy=obj_hierarchy,
                                                                   roi_type='largest')
    obj, mask = pcv.object_composition(img=img, contours=roi_objects, hierarchy=hierarchy)
    analysis_image = pcv.analyze_object(img=img, obj=obj, mask=mask, label="default")


    if with_images:
        fig, ax = plt.subplots(4, 4)
        ax[0][0].imshow(img, cmap="gray")
        ax[0][0].title.set_text("original")
        ax[0][1].imshow(binary, cmap="gray")
        ax[0][1].title.set_text("binary")
        ax[0][2].imshow(s_mblur, cmap="gray")
        ax[0][2].title.set_text("s_mblur")
        ax[0][3].imshow(gaussian_blur, cmap="gray")
        ax[0][3].title.set_text("gaussian_blur")

        ax[1][0].imshow(opened_mask, cmap="gray")
        ax[1][0].title.set_text("opened_mask")
        ax[1][1].imshow(closed_mask, cmap="gray")
        ax[1][1].title.set_text("closed_mask")
        ax[1][2].imshow(masked_o, cmap="gray")
        ax[1][2].title.set_text("masked_o")
        ax[1][3].imshow(masked_c, cmap="gray")
        ax[1][3].title.set_text("masked_c")

        ax[2][0].imshow(masked_o_c, cmap="gray")
        ax[2][0].title.set_text("masked_o_c")
        ax[2][1].imshow(opened_closed_mask, cmap="gray")
        ax[2][1].title.set_text("opened_closed_mask")
        ax[2][2].imshow(cropped, cmap="gray")
        ax[2][2].title.set_text("cropped")
        ax[2][3].imshow(masked_c, cmap="gray")
        ax[2][3].title.set_text("masked_c")


        ax[3][0].imshow(analysis_image, cmap="gray")
        ax[3][0].title.set_text("analysis_image")
        ax[3][1].imshow(stump_e, cmap="gray")
        ax[3][1].title.set_text("stump_e")
        ax[3][2].imshow(stump_d, cmap="gray")
        ax[3][2].title.set_text("stump_d")
        ax[3][3].imshow(only_plant, cmap="gray")
        ax[3][3].title.set_text("only_plant")



        plt.show()

        plt.imshow(analysis_image)
        plt.show()


    highest_x = np.amax(obj[:,:,0])
    highest_y = np.amax(obj[:,:,1])
    lowest_x = np.amin(obj[:,:,0])
    lowest_y = np.amin(obj[:,:,1])

    plant_width = highest_x - lowest_x
    plant_height = highest_y - lowest_y
    plant_area = obj_area
    plant_to_image_ratio = plant_area / (1280*960)
    plant_area_1 = np.sum(only_plant)
    plant_to_image_ratio_1 = plant_area_1 / (1280*960*255)


    if with_images:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(analysis_image, cmap="gray")
        ax[1].imshow(only_plant, cmap="gray")
        plt.show()
    return plant_width, plant_height, plant_area

def main():
    df = get_df()
    pw_side = []  #plant width
    ph_side =[] #height
    pa_side = [] #and leaf area estimates from the side gray images
    pw_top = [] #and from the top rgb immages
    ph_top =[]
    pa_top = []

    for idx, row in df.iterrows():
        img_side = (plt.imread(row["side_cam_path"], format="png") * 255).astype(np.uint8)
        img_top = (plt.imread(row["color_cam_path"], format="png") * 255).astype(np.uint8)

        # visualize_original(images[0], images[1], images[2])
        plant_width_t, plant_height_t, plant_area_t = run_single_rgb(img_top, with_images =False)
        pw_top.append(plant_width_t)
        ph_top.append(plant_height_t)
        pa_top.append(plant_area_t)


        plant_width_s, plant_height_s, plant_area_s = run_single_gray(img_side, with_images=False)
        # plant_width_s, plant_height_s, plant_area_s = 0,0,0
        pw_side.append(plant_width_s)
        ph_side.append(plant_height_s)
        pa_side.append(plant_area_s)





    df["plant_width_side_view"] = pw_side
    df["plant_height_side_view"] = ph_side
    df["plant_area_side_view"] = pa_side
    df["plant_width_top_view"] = pw_top
    df["plant_height_top_view"] = ph_top
    df["plant_area_top_view"] = pa_top

    df.to_csv("seedling_labels_with_features.csv", index=False)

if __name__ == "__main__":
    main()
