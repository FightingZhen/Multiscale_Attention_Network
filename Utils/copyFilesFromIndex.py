import shutil
import os

src_dir = 'D:/dld_eigens/'
tgt_dir = 'D:/dld_eigens_cp/'
mode = 'test/'

src_fileNames = os.listdir(src_dir)

tgt_dir = tgt_dir + mode
if not os.path.isdir(tgt_dir):
    os.makedirs(tgt_dir)

guide_dir = 'D:/DLD_split_o8_s4/' + mode
guide_fileNames = os.listdir(guide_dir)

for g in guide_fileNames:
    shutil.copy(src_dir + g, tgt_dir + g)
