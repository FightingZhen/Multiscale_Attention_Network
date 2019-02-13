# 利用给定三维中心点坐标找出原CT图像中对应的图像

import SimpleITK as sitk
import time
import os
import operator

ROI_WIDTH = 32
mode = 'TEST'

image_dir = 'D:/Source_DB/Lung_DB/AnaData/'

if mode == 'TRAIN':
    out_dir = 'D:/New_Lung_Data/train/'
elif mode == 'TEST':
    out_dir = 'D:/New_Lung_Data/test/'
else:
    out_dir = ''

if mode == 'TRAIN':
    guide_file_dir = 'D:/Experiment Data/Multi-scale-residual-network data/train.txt'
elif mode == 'TEST':
    guide_file_dir = 'D:/Experiment Data/Multi-scale-residual-network data/test.txt'
else:
    guide_file_dir = ''

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)


def gather_guide_information(guide_file_dir):
    guide = open(guide_file_dir, 'r')
    guide_file_list = []
    for line in guide:
        line_list = line.split(' ')
        gf_name = line_list[0].split('\\')[-1]
        category = line_list[1]
        index_x = line_list[2]
        index_y = line_list[3]
        index_z = line_list[4].strip('\n')
        guide_file_list.append([gf_name, category, index_x, index_y, index_z])

    return guide_file_list


def gather_same_list(guide_file_list):
    file_lists = get_training_file_lists(guide_file_list)
    total_index_list = []

    for f in file_lists:
        index_list = find_same_list(f, guide_file_list)
        total_index_list.append(index_list)
    return file_lists, total_index_list


def find_same_list(target_str, str_list):
    index_list = []
    for i in range(len(str_list)):
        if operator.eq(target_str, str_list[i][0]) == 1:
            index_list.append(i)
    return index_list


def get_training_file_lists(guide_file_lists):
    file_lists = []
    for i in range(len(guide_file_lists)):
        if guide_file_lists[i][0] not in file_lists:
            file_lists.append(guide_file_lists[i][0])
    return file_lists


def list2str(pixel_list):
    str_list = list(map(str, pixel_list))
    return ','.join(str_list) + '\n'


def save2file(save_dir, file_name, image_list, category):
    file = open(save_dir + file_name[:-7] + '.txt', 'a+')
    for line in image_list:
        str_line = list2str(line)
        file.write(str_line)
    file.write(category + '\n')
    file.close()


def extract_from_ct_cube(cube_dir, out_dir, guide_information_list):
    file_lists, index_lists = gather_same_list(guide_information_list)

    for i in range(len(file_lists)):
        start_time = time.time()

        file_name = file_lists[i]

        index_set = index_lists[i]

        image = sitk.ReadImage(cube_dir + file_name)

        image_array = sitk.GetArrayFromImage(image)

        for j in range(len(index_set)):
            image_list = []
            line_list = []

            category = guide_information_list[index_set[j]][1]
            index_x = int(guide_information_list[index_set[j]][2])
            index_y = int(guide_information_list[index_set[j]][3])
            index_z = int(guide_information_list[index_set[j]][4])

            start_x = index_x - ROI_WIDTH // 2
            start_y = index_y - ROI_WIDTH // 2

            for n in range(start_y, start_y + ROI_WIDTH):
                for m in range(start_x, start_x + ROI_WIDTH):
                    line_list.append(image_array[index_z][n][m])
                image_list.append(line_list)
                line_list = []

            save2file(out_dir, file_name, image_list, category)

        end_time = time.time()
        print('File:' + file_name + ' uses time: ' + str(end_time - start_time) + ' s')


if __name__ == '__main__':
    information = gather_guide_information(guide_file_dir)
    # get_training_file_lists(information)
    # gather_same_list(information)
    extract_from_ct_cube(image_dir, out_dir, information)
