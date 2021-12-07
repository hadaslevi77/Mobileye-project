#part 1
import pandas
import argparse
import cv2
import webcolors
from skimage import exposure

try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    import scipy
    import scipy.ndimage as ndimage
    from PIL import Image
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])

kernel2 = np.array([[-1, -1, -1, -1, -1],
                    [3, 2, 2, -1, -1],
                    [3, 2, -1, -1, -1],
                    [3, 2, 2, -1, -1],
                    [-1, -1, -1, -1, -1]])

kernel3 = np.array([[-1, -1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, -1, -1, -1, -1],
                    [4, 3, 1, 1, -1, -1, -1],
                    [4, 2, 1, -1, -1, -1, -1],
                    [4, 2, 1, -1, -1, -1, -1],
                    [4, 2, 1, -1, -1, -1, -1],
                    [4, 3, 1, 1, -1, -1, -1],
                    [-1, 1, 1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]])

kernel4 = np.array([[-1, -1, -1, -1, -1],
                    [-1, 2, 2, 2, -1],
                    [-2, 2, 3, 2, -2],
                    [-1, 2, 2, 2, -1],
                    [-1, -1, -1, -1, -1]])

kernel1 = np.array([
    [-1 / 256, -4 / 256, -6 / 256, -4 / 256, -1 / 256],
    [-4 / 256, 6 / 256, 6 / 256, 6 / 256, -4 / 256],
    [-6 / 256, 6 / 256, 12 / 256, 6 / 256, -6 / 256],
    [-4 / 256, 6 / 256, 6 / 256, 6 / 256, -4 / 256],
    [-1 / 256, -4 / 256, -6 / 256, -4 / 256, -1 / 256]
])


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    # 0 blue, 1 green, 2 red
    img_gray = rgb2gray(c_image)

    yellow_blue = c_image[:, :, 2]
    green_image = c_image.copy()  # Make a copy
    green_image[:, :, 0] = 0
    green_image[:, :, 2] = 0

    red_image = c_image.copy()  # Make a copy
    red_image[:, :, 2] = 0
    red_image[:, :, 1] = 0

    conv_im1 = sg.convolve2d(yellow_blue, kernel2)

    plt.figure()
    h1 = plt.subplot(121)
    plt.imshow(yellow_blue)
    plt.subplot(122, sharex=h1, sharey=h1)
    plt.imshow(conv_im1)

    # res = scipy.ndimage.maximum_filter(conv_im1, 155)
    xList = []
    yList = []

    neighborhood_size = 50
    threshold = 1500
    data_max = scipy.ndimage.maximum_filter(conv_im1, neighborhood_size)
    maxima = (conv_im1 == data_max)
    # data_min =  scipy.ndimage.minimum_filter(conv_im1, neighborhood_size)
    # diff = ((data_max - data_min) > threshold)
    diff = (data_max > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        y_center = (dy.start + dy.stop - 1) / 2
        if x_center >= 10 and x_center <= 2043 and y_center >= 10 and y_center <= 1021:
            xList.append(x_center)
            yList.append(y_center)

    print(xList)
    print(yList)

    indexes_line = []

    for i in range(len(xList)):
        flag = True
        for j in range(1, 28):
            if (int(yList[i]) - j) < 1032 and conv_im1[int(yList[i]) - j][int(xList[i])] < 2000:
                flag = False
        if flag == True:
            indexes_line.append(i)

    xList_good = []
    yList_good = []
    for i in range(len(xList)):
        if i not in indexes_line:
            xList_good.append(xList[i])
            yList_good.append(yList[i])
    print(xList_good, yList_good)

    xList_red = []
    yList_red = []
    xList_green = []
    yList_green = []

    for i in range(len(xList_good)):
        print(c_image[int(yList_good[i])][int(xList_good[i])])
        if np.argmax(c_image[int(yList_good[i])][int(xList_good[i])]) == 1:
            xList_green.append(xList_good[i])
            yList_green.append(yList_good[i])
            print(c_image[int(yList_good[i])][int(xList_good[i])])
            print("green")
        if np.argmax(c_image[int(yList_good[i])][int(xList_good[i])]) == 2:
            xList_red.append(xList_good[i])
            yList_red.append(yList_good[i])
            print(c_image[int(yList_good[i])][int(xList_good[i])])
            print("red")

    # print( xList_red, yList_red, xList_green, yList_green)

    # return xList_red, yList_red, xList_green, yList_green
    return xList_good, yList_good

    ### WRITE YOUR CODE HERE ###
    ### USE HELPER FUNCTIONS ###
    # return [500, 510, 520], [500, 500, 500], [700, 710], [500, 500]


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
        return objects
    show_image_and_gt(image, objects, fig_num)

    # red_x, red_y, green_x, green_y= find_tfl_lights(image)
    # plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    # plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
    red_x, red_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = r"C:\Users\aannr\Downloads\leftImg8bit_trainvaltest\leftImg8bit\train\aachen"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pandas.read_csv(r'C:\Users\aannr\Downloads\python-project-color-detection\colors.csv', names=index, header=None)


def check_color(points, img_path):
    cnames = []
    # index = ["color", "color_name", "hex", "R", "G", "B"]
    # csv = pandas.read_csv(r'C:\Users\aannr\Downloads\python-project-color-detection\colors.csv', names=index, header=None)
    img = cv2.imread(img_path)
    # for point in points:
    #     x, y = point
    #
    #     b, g, r = img[int(y), int(x)]
    #     b = int(b)
    #     g = int(g)
    #     r = int(r)
    #     minimum = 100000000
    #     for i in range(len(csv)):
    #         d = abs(r - int(csv.loc[i, "R"])) + abs(g - int(csv.loc[i, "G"])) + abs(b - int(csv.loc[i, "B"]))
    #         if (d <= minimum):
    #             minimum = d
    #             cname= csv.loc[i, "color_name"]
    #     cnames.append(cname)
    # return cnames
    colors = []
    for point in points:
        y = int(point[1])
        x = int(point[0])
        for y_top in range(y, y + 20):
            for x_right in range(x, x + 20):
                b, g, r = img[x_right, y_top]
                cnames.append(my_color(r, g, b))
        for y_top in range(y - 20, y):
            for x_right in range(x - 20, x):
                b, g, r = img[x_right, y_top]
                cnames.append(my_color(r, g, b))
        for y_top in range(y, y + 20):
            for x_right in range(x - 20, x):
                b, g, r = img[x_right, y_top]
                cnames.append(my_color(r, g, b))
        for y_top in range(y - 20, y):
            for x_right in range(x, x + 20):
                b, g, r = img[x_right, y_top]
                cnames.append(my_color(r, g, b))
        if cnames.count('g') > cnames.count('r'):
            colors.append('g')
        elif cnames.count('g') < cnames.count('r'):
            colors.append('r')


def my_color(r, g, b):
    cname = webcolors.rgb_to_name((r,g,b))
    return cname
    b = int(b)
    g = int(g)
    r = int(r)
    minimum = 10000
    for i in range(len(csv)):
        d = abs(r - int(csv.loc[i, "R"])) + abs(g - int(csv.loc[i, "G"])) + abs(b - int(csv.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            cname = csv.loc[i, "color_name"]
    if cname.find('Green') != -1:
        return 'g'
    elif cname.find('Red') != -1:
        return 'r'


if __name__ == '__main__':
    main()
