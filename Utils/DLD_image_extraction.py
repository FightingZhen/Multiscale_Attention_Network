# 从CT数据块中按特定条件抽取图像

import SimpleITK as sitk
import os
import numpy as np

source_image_dir = 'D:/Source_DB/Lung_DB/AnaData/'
source_label_dir = 'D:/Source_DB/Lung_DB/NII-files/'
out_dir = 'D:/DLD_origin_o8_s4/'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)


def list2str(value_list):
    str_list = list(map(str, value_list))
    return ','.join(str_list) + '\n'


def generate_source_data_file_list(source_dir):
    return os.listdir(source_dir)


def get_overlap_percentage(start_x, start_y, start_z, image_array, roi_x, roi_y):
    value_in_label = []
    for c in range(start_y, start_y + roi_y):
        for r in range(start_x, start_x + roi_x):
            value = image_array[start_z][c][r]
            if value != 0:
                value = 1
            else:
                value = 0
            value_in_label.append(value)
    percentage = np.mean(value_in_label)

    return percentage


def get_image(start_x, start_y, start_z, image_array, roi_x, roi_y):
    CENTER = -650
    STRIDE = 750

    MIN_PIXEL = CENTER - STRIDE
    MAX_PIXEL = CENTER + STRIDE

    image_list = []
    for c in range(start_y, start_y + roi_y):
        for r in range(start_x, start_x + roi_x):
            value = image_array[start_z][c][r]
            image_list.append(value)

    for i in range(len(image_list)):
        if image_list[i] < MIN_PIXEL:
            image_list[i] = MIN_PIXEL
        if image_list[i] > MAX_PIXEL:
            image_list[i] = MAX_PIXEL

    reshaped = np.reshape(image_list, (roi_x, roi_y)).tolist()

    return reshaped


def save2file(image_list, label, out_dir, file_name):
    file = open(out_dir + file_name.strip('.nii.gz') + '.txt', 'a+')
    for item in image_list:
        file.write(list2str(item))
    file.write(str(label) + '\n')
    file.close()


def save_index(out_dir, file_name, center_point, label):
    file = open(out_dir + file_name.strip('.nii.gz') + '.txt', 'a+')
    information = out_dir + file_name + ' ' + str(label) + ' ' + str(center_point[0]) + ' ' + str(
        center_point[1]) + ' ' + str(center_point[2]) + '\n'
    file.write(information)
    file.close()


def extract_image_label(file_name):
    ROI_X = 32
    ROI_Y = 32
    OVERLAP_PERCENTAGE = 0.8

    if file_name[:-11] == 'DIF_NOD':
        STRIDE = 4
    elif file_name[:-11] == 'EMP':
        STRIDE = 12
    elif file_name[:-11] == 'HCM':
        STRIDE = 4
    elif file_name[:-11] == 'Mul_CON':
        STRIDE = 4
    elif file_name[:-11] == 'Mul_GGO':
        STRIDE = 4
    elif file_name[:-11] == 'NOR':
        STRIDE = 14
    elif file_name[:-11] == 'Ret_GGO':
        STRIDE = 4
    else:
        print('File name error !')
        return

    # print(file_name)
    image = sitk.ReadImage(source_image_dir + file_name)
    label = sitk.ReadImage(source_label_dir + file_name)

    size = image.GetSize()
    # print(origin)
    # print(size)

    image_arr = sitk.GetArrayFromImage(image)
    label_arr = sitk.GetArrayFromImage(label)

    start_x = 0
    start_y = 0
    start_z = 0

    cube_x = int(size[0])
    cube_y = int(size[1])
    cube_z = int(size[2])

    for z in range(start_z, start_z + cube_z):
        for y in range(start_y, start_y + cube_y - ROI_Y, STRIDE):
            for x in range(start_x, start_x + cube_x - ROI_X, STRIDE):
                center_point = [x + ROI_X // 2, y + ROI_Y // 2, z]
                center_point_label = label_arr[center_point[2]][center_point[1]][center_point[0]]
                if center_point_label != 0:
                    overlap_percentage = get_overlap_percentage(x, y, z, label_arr, ROI_X, ROI_Y)
                    if overlap_percentage >= OVERLAP_PERCENTAGE:
                        # print(str(x), str(y), str(z), str(overlap_percentage))
                        ext_image = get_image(x, y, z, image_arr, ROI_X, ROI_Y)
                        save_index(out_dir, file_name, center_point, center_point_label)
                        save2file(ext_image, center_point_label, out_dir, file_name)

    print('File: ' + file_name + ' finished , stride : ' + str(STRIDE))


def main():
    source_file_list = generate_source_data_file_list(source_image_dir)
    for f in source_file_list:
        extract_image_label(f)


def extract_index_image(image_root_dir, mode):
    files_list = generate_source_data_file_list(image_root_dir + mode + '/')
    index_dir = 'D:/DLD_o8_s4_index/'
    image_dir = 'D:/DLD_o8_s4/' + mode + '/'

    if not os.path.isdir(index_dir):
        os.makedirs(index_dir)
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    for file in files_list:
        f = open(out_dir + file, 'r')
        w_index = open(index_dir + mode + '_index.txt', 'a+')
        w_image = open(image_dir + file, 'a+')

        for line in f:
            if line[0] == 'D':
                w_index.write(line)
            else:
                w_image.write(line)
        f.close()


if __name__ == '__main__':
    # main()
    extract_index_image('D:/DLD_split_o8_s4/', 'test')
