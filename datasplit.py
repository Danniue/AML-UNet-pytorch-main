import os
import argparse
import random
import shutil
from shutil import copyfile


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)


def main(config):
    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)

    filenames = os.listdir(config.origin_data_path)
    masknames = os.listdir(config.origin_GT_path)
    data_list = []
    GT_list = []

    for filename in filenames:
        data_list.append(filename)
    for maskname in masknames:
        GT_list.append(maskname)

    data_list = sorted(data_list)
    GT_list = sorted(GT_list)

    num_total = len(data_list)
    num_train = int((config.train_ratio / (config.train_ratio +
                                           config.valid_ratio + config.test_ratio)) * num_total)
    num_valid = int((config.valid_ratio / (config.train_ratio +
                                           config.valid_ratio + config.test_ratio)) * num_total)
    num_test = num_total - num_train - num_valid

    print('\nNum of train set : ', num_train)
    print('\nNum of valid set : ', num_valid)
    print('\nNum of test set : ', num_test)

    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train):
        idx = Arange.pop()
        print(idx)
        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.train_path, data_list[idx])
        copyfile(src, dst)
        print(src, dst)
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.train_GT_path, GT_list[idx])
        copyfile(src, dst)
        print(src, dst)


    for i in range(num_valid):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.valid_path, data_list[idx])
        copyfile(src, dst)

        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.valid_GT_path, GT_list[idx])
        copyfile(src, dst)


    for i in range(num_test):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.test_path, data_list[idx])
        copyfile(src, dst)

        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.test_GT_path, GT_list[idx])
        copyfile(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0)

    # data path
    parser.add_argument('--origin_data_path', type=str,
                        default='E:/code/DeformableDualUnet/data/ACSA/or_files/images')
    parser.add_argument('--origin_GT_path', type=str,
                        default='E:/code/DeformableDualUnet/data/ACSA/or_files/masks')

    parser.add_argument('--train_path', type=str, default='E:/code/DeformableDualUnet/data/ACSA/train/images')
    parser.add_argument('--train_GT_path', type=str,
                        default='E:/code/DeformableDualUnet/data/ACSA/train/masks')
    parser.add_argument('--valid_path', type=str, default='E:/code/DeformableDualUnet/data/ACSA/test/images')
    parser.add_argument('--valid_GT_path', type=str,
                        default='E:/code/DeformableDualUnet/data/ACSA/test/masks')
    parser.add_argument('--test_path', type=str, default='E:/code/DeformableDualUnet/data/ACSA/test/images')
    parser.add_argument('--test_GT_path', type=str,
                        default='E:/code/DeformableDualUnet/data/ACSA/test/masks')

    config = parser.parse_args()
    print(config)
    main(config)