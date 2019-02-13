import os

Training_period = True
Test_period = True

l1_training_dir = 'D:/dld_eigen_value/train/ev1/'
l2_training_dir = 'D:/dld_eigen_value/train/ev2/'
l3_training_dir = 'D:/dld_eigen_value/train/ev3/'

l1_validation_dir = 'D:/dld_eigen_value/validation/ev1/'
l2_validation_dir = 'D:/dld_eigen_value/validation/ev2/'
l3_validation_dir = 'D:/dld_eigen_value/validation/ev3/'

l1_test_dir = 'D:/dld_eigen_value/test/ev1/'
l2_test_dir = 'D:/dld_eigen_value/test/ev2/'
l3_test_dir = 'D:/dld_eigen_value/test/ev3/'

out_training_dir = 'D:/DLD_Eigens/train/'
out_validation_dir = 'D:/DLD_Eigens/validation/'
out_test_dir = 'D:/DLD_Eigens/test/'

training_file_list = os.listdir(l1_training_dir)
validation_file_list = os.listdir(l1_validation_dir)
test_file_list = os.listdir(l1_test_dir)

if not os.path.isdir(out_training_dir):
    os.makedirs(out_training_dir)
if not os.path.isdir(out_validation_dir):
    os.makedirs(out_validation_dir)
if not os.path.isdir(out_test_dir):
    os.makedirs(out_test_dir)


# def read():
#     for file_name in training_file_list:
#         print(file_name)
#         f = open(out_training_dir+file_name,'r')
#         for line in f:
#             line = line[:-2]
#             line = line.split(',')
#             print(len(line))

def gather_information(list1, list2, list3):
    result = []

    for i in range(len(list1)):
        result.append(list1[i])
        result.append(list2[i])
        result.append(list3[i])

    result_str = list(map(str, result))
    result_str = ','.join(result_str)
    return result_str


def str_to_list(str):
    str = str.strip(',\n')
    str = str.split(',')
    for i in range(len(str)):
        str[i] = float(str[i])
    return str


def main(mode):
    l1_dir = ''
    l2_dir = ''
    l3_dir = ''
    out_dir = ''

    if mode == 'TRAIN':
        l1_dir = l1_training_dir
        l2_dir = l2_training_dir
        l3_dir = l3_training_dir
        out_dir = out_training_dir
        file_list = training_file_list
    if mode == 'VALIDATION':
        l1_dir = l1_validation_dir
        l2_dir = l2_validation_dir
        l3_dir = l3_validation_dir
        out_dir = out_validation_dir
        file_list = validation_file_list
    if mode == 'TEST':
        l1_dir = l1_test_dir
        l2_dir = l2_test_dir
        l3_dir = l3_test_dir
        out_dir = out_test_dir
        file_list = test_file_list

    for file_name in file_list:
        f_l1 = open(l1_dir + file_name, 'r')
        f_l2 = open(l2_dir + file_name, 'r')
        f_l3 = open(l3_dir + file_name, 'r')
        f_out = open(out_dir + file_name, 'w')

        for line_l1, line_l2, line_l3 in zip(f_l1, f_l2, f_l3):
            if line_l1.index('\n') != 1:
                f1_list = str_to_list(line_l1)
                f2_list = str_to_list(line_l2)
                f3_list = str_to_list(line_l3)

                gather = gather_information(f1_list, f2_list, f3_list)
                f_out.write(gather + '\n')
            else:
                f_out.write(line_l1)
        f_out.close()


if __name__ == '__main__':
    main(mode='TRAIN')
    main(mode='VALIDATION')
    main(mode='TEST')
