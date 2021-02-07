import os
import gdal
import numpy as np
from tqdm import tqdm


#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


#  保存tif文件函数
def writeTiff(im_data, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    # if (dataset != None):
    #     dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    #     dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    # del dataset


def sample_is_invalid(mask_array):
    '''
    判断样本是否无效
    :return:
    '''
    # area_useful = np.sum(mask_array // 255)
    area_useful = np.sum(mask_array)
    # h, w = mask_array.shape
    return area_useful == 0


'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''


def tif_crop(
        tif_path,
        mask_path,
        tif_croped_save_path,
        mask_croped_save_path,
        crop_size,
        repetition_rate=0
):
    dataset_img = readTif(tif_path)
    dataset_mask = readTif(mask_path)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
    mask = dataset_mask.ReadAsArray(0, 0, width, height)  # 获取数据

    print("++++++++++", img.shape)

    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    if not os.path.exists(tif_croped_save_path):
        os.makedirs(tif_croped_save_path)
    if not os.path.exists(mask_croped_save_path):
        os.makedirs(mask_croped_save_path)
    new_name = len(os.listdir(tif_croped_save_path)) + 1
    #  裁剪图片,重复率为RepetitionRate

    for i in range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        for j in range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
            # mask为单波段，shape只有2维
            mask_cropped = mask[
                           int(i * crop_size * (1 - repetition_rate)): int(
                               i * crop_size * (1 - repetition_rate)) + crop_size,
                           int(j * crop_size * (1 - repetition_rate)): int(
                               j * crop_size * (1 - repetition_rate)) + crop_size]

            # if np.argmin(img.shape) == 2:
            # cropped = img[
            #           int(i * crop_size * (1 - repetition_rate)): int(
            #               i * crop_size * (1 - repetition_rate)) + crop_size,
            #           int(j * crop_size * (1 - repetition_rate)): int(
            #               j * crop_size * (1 - repetition_rate)) + crop_size,:]
            # else:
            cropped = img[:,
                      int(i * crop_size * (1 - repetition_rate)): int(
                          i * crop_size * (1 - repetition_rate)) + crop_size,
                      int(j * crop_size * (1 - repetition_rate)): int(
                          j * crop_size * (1 - repetition_rate)) + crop_size]

            if sample_is_invalid(mask_cropped):  # 如果是无效样本，则不保存
                continue
            # 写图像
            writeTiff(cropped, tif_croped_save_path + "/%d.tif" % new_name)
            writeTiff(mask_cropped, mask_croped_save_path + "/%d.tif" % new_name)
            #  文件名 + 1
            new_name = new_name + 1
    #  向前裁剪最后一列
    for i in range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        mask_cropped = mask[int(i * crop_size * (1 - repetition_rate)): int(
            i * crop_size * (1 - repetition_rate)) + crop_size,
                       (width - crop_size): width]
        cropped = img[:,
                  int(i * crop_size * (1 - repetition_rate)): int(
                      i * crop_size * (1 - repetition_rate)) + crop_size,
                  (width - crop_size): width]

        if sample_is_invalid(mask_cropped):
            continue
        #  写图像
        writeTiff(cropped, tif_croped_save_path + "/%d.tif" % new_name)
        writeTiff(mask_cropped, mask_croped_save_path + "/%d.tif" % new_name)
        new_name = new_name + 1
    #  向前裁剪最后一行
    for j in range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        mask_cropped = mask[(height - crop_size): height,
                       int(j * crop_size * (1 - repetition_rate)): int(
                           j * crop_size * (1 - repetition_rate)) + crop_size]

        cropped = img[:,
                  (height - crop_size): height,
                  int(j * crop_size * (1 - repetition_rate)): int(
                      j * crop_size * (1 - repetition_rate)) + crop_size]

        if sample_is_invalid(mask_cropped):
            continue
        writeTiff(cropped, tif_croped_save_path + "/%d.tif" % new_name)
        writeTiff(mask_cropped, mask_croped_save_path + "/%d.tif" % new_name)
        #  文件名 + 1
        new_name = new_name + 1
    #  裁剪右下角
    mask_cropped = mask[(height - crop_size): height,
                   (width - crop_size): width]
    cropped = img[:,
              (height - crop_size): height,
              (width - crop_size): width]

    if not sample_is_invalid(mask_cropped):
        writeTiff(cropped, tif_croped_save_path + "/%d.tif" % new_name)
        writeTiff(mask_cropped, mask_croped_save_path + "/%d.tif" % new_name)
    print(f'当前共有{new_name}张训练图片。')


def tif_crop_all(
        tifs_dir,
        masks_dir,
        crop_size,
        tifs_croped_save_dir,
        masks_croped_save_dir,
        repetition_rate=0
):
    '''
    批量将原始图像和mask进行裁剪
    :param tifs_dir:
    :param masks_dir:
    :param tifs_croped_save_dir:
    :param masks_croped_save_dir:
    :return:
    '''
    for root, dirs, files in os.walk(tifs_dir):
        for tif_file in tqdm(files):
            tif_file_path = os.path.join(root, tif_file)
            mask_file_path = os.path.join(masks_dir, tif_file)
            tif_crop(tif_file_path, mask_file_path, tifs_croped_save_dir, masks_croped_save_dir, crop_size,
                     repetition_rate)


if __name__ == '__main__':
    tifs_dir = 'E:\workspace\sandiao\data\Multi-class/total_imgs'
    masks_dir = 'E:\workspace\sandiao\data\Multi-class/total_masks'
    tifs_croped_save_dir = 'E:\workspace\sandiao\data\Multi-class/train_data/imgs_train'
    masks_croped_save_dir = 'E:\workspace\sandiao\data\Multi-class/train_data\imgs_mask'
    tif_crop_all(tifs_dir, masks_dir, 512, tifs_croped_save_dir, masks_croped_save_dir)
