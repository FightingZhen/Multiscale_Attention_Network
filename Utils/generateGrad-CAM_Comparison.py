from PIL import Image
import numpy as np
import os
from PIL import ImageDraw, ImageFont


originImagePath = 'D:/Workspace/DLD_Classification/visualTestImage/'

Grad_CAM_ImagePath_scale1 = 'D:/Workspace/DLD_Classification/Grad_CAM_Split/scale1/'
Grad_CAM_ImagePath_scale2 = 'D:/Workspace/DLD_Classification/Grad_CAM_Split/scale2/'
Grad_CAM_ImagePath_scale3 = 'D:/Workspace/DLD_Classification/Grad_CAM_Split/scale3/'
Grad_CAM_ImagePath_sum = 'D:/Workspace/DLD_Classification/Grad_CAM_Split/sum/'

softmaxScoreFile = 'D:/Workspace/DLD_Classification/model_results_ROC/after-softmax/score/fc_attention_y_score.txt'
realLabelFile = 'D:/Workspace/DLD_Classification/model_results_ROC/after-softmax/label/fc_attention_test_label.txt'
savedPath = 'D:/Workspace/DLD_Classification/Grad_CAM_Split_Comparison/'


def getFileList(filePath):
    l_dir = os.listdir(filePath)
    l_dir.sort(key=lambda x: int(x[:-4]))
    return l_dir


def getTargetLine(fileName, lineNumber):
    with open(fileName, 'r') as f:
        info = f.readlines()

    return info[lineNumber]


def getRealLabel(line):
    l = line[:-2]
    l = l.split(' ')
    l = list(map(float, l))

    return str(list(np.where(l == np.max(l)))[0][0])


def generateFusedImages(fileLists, rowsPerImage=10):
    pattern_name = {0: 'CON',
                    1: 'M-GGO',
                    2: 'HCM',
                    3: 'R-GGO',
                    4: 'EMP',
                    5: 'NOD',
                    6: 'NOR'}

    img_counter = 0
    saved_counter = 0
    total_counter = 0
    file_name_lib = []
    softmax_score_lib = []
    real_label_lib = []
    origin_img_lib = []
    cam_img_lib_sc1 = []
    cam_img_lib_sc2 = []
    cam_img_lib_sc3 = []
    cam_img_lib_sum = []

    for f in fileLists:
        img_number = int(f.split('.')[0])
        softmax_score = getTargetLine(softmaxScoreFile, img_number)[:-1]
        real_label = getRealLabel(getTargetLine(realLabelFile, img_number))
        softmax_score_lib.append(softmax_score)
        real_label_lib.append(real_label + '(' + pattern_name[int(real_label)] + ')')
        origin_img_lib.append(Image.open(originImagePath + f))
        cam_img_lib_sc1.append(Image.open(Grad_CAM_ImagePath_scale1 + f))
        cam_img_lib_sc2.append(Image.open(Grad_CAM_ImagePath_scale2 + f))
        cam_img_lib_sc3.append(Image.open(Grad_CAM_ImagePath_scale3 + f))
        cam_img_lib_sum.append(Image.open(Grad_CAM_ImagePath_sum + f))
        file_name_lib.append(f)
        img_counter += 1
        total_counter += 1

        if img_counter == rowsPerImage:
            saved_counter += 1
            img_width, img_height = origin_img_lib[0].size
            img_width += 3
            img_height += 3
            newImg = Image.new('RGB', (img_width * 26, img_height * rowsPerImage))
            draw = ImageDraw.Draw(newImg)
            for i in range(rowsPerImage):
                newImg.paste(origin_img_lib[i], box=(img_width * 2, 0 + img_height * i))
                newImg.paste(cam_img_lib_sc1[i], box=(img_width * 3, 0 + img_height * i))
                newImg.paste(cam_img_lib_sc2[i], box=(img_width * 4, 0 + img_height * i))
                newImg.paste(cam_img_lib_sc3[i], box=(img_width * 5, 0 + img_height * i))
                newImg.paste(cam_img_lib_sum[i], box=(img_width * 6, 0 + img_height * i))
                draw.text((0, 0 + img_height * i), file_name_lib[i])
                draw.text((img_width * 7, 0 + img_height * i), softmax_score_lib[i])
                draw.text((img_width * 24, 0 + img_height * i), real_label_lib[i])
            if not os.path.isdir(savedPath):
                os.makedirs(savedPath)
            newImg.save(savedPath + str(saved_counter) + '.png')
            print('File ' + str(saved_counter) + '.png has been saved !')

            img_counter = 0
            softmax_score_lib = []
            real_label_lib = []
            origin_img_lib = []
            cam_img_lib_sc1 = []
            cam_img_lib_sc2 = []
            cam_img_lib_sc3 = []
            cam_img_lib_sum = []
            file_name_lib = []

        if total_counter == len(fileLists) and len(fileLists) % rowsPerImage != 0:
            saved_counter += 1
            remainImgNum = len(fileLists) % rowsPerImage
            img_width, img_height = origin_img_lib[0].size
            img_width += 3
            img_height += 3
            newImg = Image.new('RGB', (img_width * 26, img_height * remainImgNum))
            draw = ImageDraw.Draw(newImg)
            for i in range(remainImgNum):
                newImg.paste(origin_img_lib[i], box=(img_width * 2, 0 + img_height * i))
                newImg.paste(cam_img_lib_sc1[i], box=(img_width * 3, 0 + img_height * i))
                newImg.paste(cam_img_lib_sc2[i], box=(img_width * 4, 0 + img_height * i))
                newImg.paste(cam_img_lib_sc3[i], box=(img_width * 5, 0 + img_height * i))
                newImg.paste(cam_img_lib_sum[i], box=(img_width * 6, 0 + img_height * i))
                draw.text((0, 0 + img_height * i), file_name_lib[i])
                draw.text((img_width * 7, 0 + img_height * i), softmax_score_lib[i])
                draw.text((img_width * 24, 0 + img_height * i), real_label_lib[i])
            if not os.path.isdir(savedPath):
                os.makedirs(savedPath)
            newImg.save(savedPath + str(saved_counter) + '.png')
            print('File ' + str(saved_counter) + '.png has been saved !')


if __name__ == '__main__':
    fList = getFileList(Grad_CAM_ImagePath_scale1)
    # print(fList)
    generateFusedImages(fList)
