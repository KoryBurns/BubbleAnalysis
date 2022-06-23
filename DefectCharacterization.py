# import the necessary packages
import cv2
import numpy as np
from matplotlib import font_manager, rc, rcParams
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, measure
from skimage.restoration import denoise_nl_means
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import pandas as pd
import argparse
import imutils
import os
from tqdm import tqdm
import warnings
import gc
warnings.simplefilter(action='ignore', category=FutureWarning)



def run_analysis(filename) :
    rcParams['figure.dpi'] = 1200
    rc('font',family='Times New Roman',size=15)
    rcParams['axes.titlepad'] = 15
    rcParams['mathtext.fontset'] = 'dejavuserif'
    ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal',
    size=15, weight='normal', stretch='normal')
    axis_font = {'fontname':'Times New Roman','fontsize':15}
    
    #Directory in which your images are stored in
    DIR = 'C:\\Users\\kdburns\\Documents\\LANL\\Nanoparticles\\'
    img = io.imread(DIR + 'Data\\' + filename)

    prefix = filename.split('.')[0]

    pixels_to_nm = (103.70/4096) #39.5 pixels = 1 nm

    #ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated= cv2.dilate(eroded, kernel, iterations=1)

    D = ndimage.distance_transform_edt(dilated)
    localMax = peak_local_max(D, indices=False, min_distance=10,
        labels=dilated)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=dilated)

    #print("[INFO] {} Helium Bubbles found in the micrograph".format(len(np.unique(labels)) - 1))



    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
    
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(img, (int(x), int(y)), int(r), (255, 0, 0), 2)
        cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output image
    #cv2.imshow("Output", image)
    cv2.imwrite(DIR + 'Processed\\IMG\\' + prefix + '_final.jpg', img)



    clusters = measure.regionprops(labels, img)

    propList = ['Area',
               'eccentricity']
    
    output_file = open(DIR + 'Processed\\CSV\\' + prefix + '_csv.csv', 'w')
    output_file.write(',' + ",".join(propList) + '\n')

    for cluster_props in clusters:
        #Output desired properties into excel
        output_file.write(str(cluster_props['Label']))
    
        for i,prop in enumerate(propList):
            if(prop.find('Area') <200):
                to_print = cluster_props[prop]*pixels_to_nm**2
            elif(prop.find('eccentricity') <1):
                to_print = cluster_props[prop]
            else:
                to_print = cluster_props[prop]
            output_file.write(',' + str(to_print))
        output_file.write('\n')
    output_file.close()




    rcParams['figure.dpi'] = 1200
    rc('font',family='Times New Roman',size=10)
    rcParams['axes.titlepad'] = 10
    rcParams['mathtext.fontset'] = 'dejavuserif'
    ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal',
    size=15, weight='normal', stretch='normal')
    axis_font = {'fontname':'Times New Roman','fontsize':10}

    data = pd.read_csv(DIR + 'Processed\\CSV\\' + prefix + '_csv.csv')
    index = data[(data['Area'] >= 40)|(data['Area'] <= 4)].index
    data.drop(index, inplace=True)
    data['Area'].describe()

    #counts = plt.legend(handles=[mx])

    num_bins = 20

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, color='purple', edgecolor='black', density=True)

    ax.set_xlabel('Area (nm$^2$)')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Size Distribution of He Bubbles')

    plt.savefig(DIR + 'Processed\\AREA\\' + prefix + '_Area_Histogram.png')
    plt.close()

    index = data[(data['Area'] >= 40)|(data['Area'] <= 4)].index
    data.drop(index, inplace=True)
    data['Area'].describe()
    
    x = (data.Area / pi) * 4
    x = x**(0.5)

    num_bins = 20

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, color='gold', edgecolor='silver', density=True)

    ax.set_xlabel('Diameter (nm)')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Size Distribution of He Bubbles')

    plt.savefig(DIR + 'Processed\\DIAMETER\\' + prefix + '_Diameter_Histogram.png')
    plt.close()

    x = data.eccentricity

    num_bins = 20

    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(x, num_bins, color='cyan', edgecolor='magenta', density=True)

    ax.set_xlabel('Eccentricity')
    ax.set_ylabel('Counts')
    ax.set_title(r'Circularity of He Bubbles')

    plt.savefig(DIR + 'Processed\\ECCENTRICITY\\' + prefix + '_Eccentricity_Histogram.png')
    plt.close()
    del data
    del img

def main_loop() :
    DIR = r'C:\\Users\\kdburns\\Documents\\LANL\\Nanoparticles\\Processed\\'
    for number in tqdm(range(0,4)) :
        filename =  str(number) + '.png'
        run_analysis(filename)
        gc.collect()

main_loop()
