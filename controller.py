#part 4
import glob
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib._png as png
import numpy as np
from PIL import Image

import SFM
from part1 import test_find_tfl_lights, find_tfl_lights, check_color
from part3 import FrameContainer, visualize


path = r"C:/Users/aannr/PycharmProjects/mobileyeProject/"


# get image path and return 2 lists of suspicious_points and their color
def part1(image_path):
    # return np.array([[100, 250], [305, 379], [250, 400], [468, 777]]), ['r', 'g', 'r', 'r']
    image = np.array(Image.open(image_path))
    red_x, red_y = find_tfl_lights(image)
    points = []
    for i in range(len(red_y)):
        points.append([red_x[i], red_y[i]])

    return np.array(points), ['g']*len(points)


def part2(image_path, suspicious_points, points_color_list):
    return suspicious_points,points_color_list


def part3(prev_image_path, curr_image_path, prev_traffic_points, curr_traffic_points, prev_color_list, curr_color_list,
          EM, focal, pp):
    #return [1, 4, 8]
    prev_container = FrameContainer(prev_image_path)
    curr_container = FrameContainer(curr_image_path)
    prev_container.traffic_light=prev_traffic_points
    curr_container.traffic_light=curr_traffic_points
    curr_container.EM=EM
    curr_container = SFM.calc_TFL_dist(prev_container, curr_container, focal, pp)
    return visualize(prev_container,curr_container,focal,pp)

class TFL_Man:
    def __init__(self, frame_index, frame_path, EM):
        self.frame_index = frame_index
        self.frame_path = frame_path
        self.EM = EM

    def get_frame_index(self):
        return self.frame_index

    def get_frame_path(self):
        return self.frame_path

    def on_frame(self, prev_frame, focal, pp):
        self.distance_list = np.array([])
        self.suspicious_points, self.suspicious_points_colors = part1(self.frame_path)
        self.traffic_points, self.traffic_color_list = part2(self.frame_path, self.suspicious_points,
                                                             self.suspicious_points_colors)

        if prev_frame.frame_index != 0:
            self.distance_list,self.foe = part3(prev_frame.frame_path, self.frame_path, prev_frame.traffic_points,
                                       self.traffic_points,
                                       prev_frame.traffic_color_list, self.traffic_color_list, self.EM, focal, pp)

        return self.distance_list


def visualization(tfl_instance: TFL_Man):
    fig, (part1_sec, part2_sec, part3_sec) = plt.subplots(1, 3, figsize=(12, 6))
    part1_sec.set_title('part1')
    part1_sec.imshow(png.read_png_int(tfl_instance.frame_path))

    part1_sec.plot(tfl_instance.suspicious_points[:, 0], tfl_instance.suspicious_points[:, 1], 'ro', color='r',
                   markersize=4)
    part2_sec.set_title('part2')
    part2_sec.imshow(png.read_png_int(tfl_instance.frame_path))
    part2_sec.plot(tfl_instance.traffic_points[:, 0], tfl_instance.traffic_points[:, 1], 'ro', color='r',
                   markersize=4)
    part3_sec.set_title('part3')
    part3_sec.imshow(png.read_png_int(tfl_instance.frame_path))
    curr_p = tfl_instance.traffic_points
    # tx = tfl_instance.EM[0, 3]
    # ty = tfl_instance.EM[1, 3]
    tz = tfl_instance.EM[2, 3]
    foe = tfl_instance.foe
    for i in range(len(curr_p)):
        part3_sec.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
        if tz:
            part3_sec.text(curr_p[i, 0], curr_p[i, 1],
                           r'{0:.1f}'.format(tfl_instance.distance_list[i,2], color='r'))
    part3_sec.plot(foe[0], foe[1], 'r+')
    plt.show()

    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot(x, y)
    # ax1.set_title('Sharing Y axis')
    # ax2.scatter(x, y)


def controller(pls_file):
    f = open(pls_file, "r")
    EM = np.array([])
    prev_frame = TFL_Man(0, "", EM)
    lines = f.readlines()
    index = int(lines[1])
    with open(path + lines[0][:lines[0].rindex('\n')], 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')
    focal = data['flx']
    pp = data['principle_point']
    l = len(lines)
    for i in range(2, l):
        if prev_frame.frame_index != 0:
            EM = np.array(data['egomotion_' + str(prev_frame.frame_index) + '-' + str(index)])
        curr_frame = TFL_Man(index, path + lines[i][:lines[i].rindex('\n')], EM)
        curr_frame.on_frame(prev_frame, focal, pp)
        prev_frame = curr_frame
        index += 1
        if prev_frame.frame_index != 24:
            visualization(curr_frame)


controller(r"C:\Users\aannr\Downloads\play_list.txt")
