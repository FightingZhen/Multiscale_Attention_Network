import os
import operator

# files_dir = 'D:/Experiment Data/Jornel_DLD_multiscale_attention_network/DLD_o8_s4/test/'
files_dir = 'D:/Experiment Data/Jornel_DLD_multiscale_attention_network/DLD_origin_o8_s4/'

file_list = os.listdir(files_dir)
IMAGE_WIDTH = 32


def check_image_num(file_name):
    f = open(files_dir + file_name, 'r')
    number = 0

    for line in f:
        if line.index('\n') != 1 and line[0] != 'D':
            number += 1

    return number / IMAGE_WIDTH


def main():
    name = 'DIF_NOD'
    counter = 0
    for file in file_list:
        if operator.eq(file[:-8], name) != 1:
            counter = 0
            name = file[:-8]
        image_num = check_image_num(file)
        counter += image_num
        print("File : %s has %g images, total : %g" % (file, image_num, counter))


def search_min_pixel_value():
    min_value = 0
    max_value = 0
    for file in file_list:
        f = open(files_dir + file, 'r')
        for line in f:
            if line.index('\n') != 1:
                line_pixel = line.strip(',\n').split(',')
                line_pixel_float = list(map(float, line_pixel))
                for p in line_pixel_float:
                    if p < min_value:
                        print(p)
                        min_value = p
                    if p > max_value:
                        max_value = p
        print('File %s finished' % file)

    print('Min_value: %g' % min_value)
    print('Max_value: %g' % max_value)

if __name__ == '__main__':
    main()
    # search_min_pixel_value()
