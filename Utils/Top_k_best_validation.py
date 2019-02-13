import copy
import os


def get_validation_test_message(file):
    val_list = []
    tst_list = []
    f_score_list = []
    epoch_list = []

    with open(file, 'r') as f:
        for line in f:
            l = line.lower()
            if l.find('epoch') == 0 and l.find('validation') > -1:
                val_start = l.index('validation')
                val_end = l.index(' , loss')
                val = float(l[val_start:val_end].split('[')[1][:-1])
                val_list.append(val)

                ep_start = l.index('epoch')
                ep_end = l.index(' , training')
                ep = float(l[ep_start:ep_end].split('[')[1][:-1])
                epoch_list.append(ep)

            if l.find('test accuracy') == 0:
                tst = float(l.split(':')[1][1:])
                tst_list.append(tst)

            if l.find('f_value') == 0:
                f_score = float(l.split(':')[1][1:])
                f_score_list.append(f_score)

    return val_list, tst_list, f_score_list, epoch_list


def top_k_val_tst_index(val_list, tst_list, f_score_list, epoch_list, k):
    var_list_bak = copy.deepcopy(val_list)
    indices = []
    for i in range(k):
        max_index = val_list.index(max(val_list))
        indices.append(max_index + 1)
        val_list[max_index] = 0

    topk_val = []
    topk_tst = []
    topk_f_score = []
    topk_eps = []

    for i in range(len(indices)):
        topk_val.append(var_list_bak[indices[i] - 1])
        topk_tst.append(tst_list[indices[i] - 1])
        topk_f_score.append(f_score_list[indices[i] - 1])
        topk_eps.append(epoch_list[indices[i] - 1])

    return indices, topk_val, topk_tst, topk_f_score, topk_eps


def top_k_tst_val_index(val_list, tst_list, f_score_list, epoch_list, k):
    tst_list_bak = copy.deepcopy(tst_list)
    indices = []
    for i in range(k):
        max_index = tst_list.index(max(tst_list))
        indices.append(max_index + 1)
        tst_list[max_index] = 0

    topk_val = []
    topk_tst = []
    topk_f_score = []
    topk_eps = []

    for i in range(len(indices)):
        topk_tst.append(tst_list_bak[indices[i] - 1])
        topk_val.append(val_list[indices[i] - 1])
        topk_f_score.append(f_score_list[indices[i] - 1])
        topk_eps.append(epoch_list[indices[i] - 1])

    return indices, topk_val, topk_tst, topk_f_score, topk_eps


def show_results_val_tst(file_name, k, indices, val_list, tst_list, f_score_list, epoch_list):
    length = len(indices)
    print('File name : %s' % file_name)
    print('-' * 10 + 'Top %d validation results and correspoding test results' % k + '-' * 10)
    for i in range(1, length + 1):
        print('Top %d epoch: [%d], validation accuracy: [%.4f], test accuracy: [%.4f], f_score: [%.4f]' % (
            i, epoch_list[i - 1], val_list[i - 1], tst_list[i - 1], f_score_list[i - 1]))


def show_results_tst_val(file_name, k, indices, val_list, tst_list, f_score_list, epoch_list):
    length = len(indices)
    print('File name : %s' % file_name)
    print('-' * 10 + 'Top %d validation results and correspoding test results' % k + '-' * 10)
    for i in range(1, length + 1):
        print('Top %d epoch: [%d], test accuracy: [%.4f], validation accuracy: [%.4f], f_score: [%.4f]' % (
            i, epoch_list[i - 1], tst_list[i - 1], val_list[i - 1], f_score_list[i - 1]))


def calculate_last_k_test_average_accuracy(test_list, k):
    sum = 0.0
    for i in range(1, k + 1):
        sum += test_list[-i]

    return sum / k


def main():
    top_k = 20
    # file_dir = 'D:/1/Test_Round0/'
    # file_dir = 'D:/1/repeatExp/'
    file_dir = 'D:/1/DLD_Extra_Experiments/'
    file_name_list = os.listdir(file_dir)
    for file in file_name_list:
        val_list, tst_list, f_score_list, epoch_list = get_validation_test_message(file_dir + file)

        # # top k validation-test -mode
        # ind, val, tst, f_score, eps = top_k_val_tst_index(val_list, tst_list, f_score_list, epoch_list, top_k)
        # show_results_val_tst(file, top_k, ind, val, tst, f_score, eps)

        # top k test-validation -mode
        ind, val, tst, f_score, eps = top_k_tst_val_index(val_list, tst_list, f_score_list, epoch_list, top_k)
        show_results_tst_val(file, top_k, ind, val, tst, f_score, eps)


if __name__ == '__main__':
    main()
