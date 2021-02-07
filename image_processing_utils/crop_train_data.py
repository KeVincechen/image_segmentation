import os
import cv2
from tqdm import tqdm


def tif_crop(
        img_path,
        mask_path,
        tif_croped_save_dir,
        mask_croped_save_dir,
        crop_size,
        repetition_rate=0
):
    '''
    滑动窗口裁剪函数
    :param img_path: 影像路径
    :param mask_path:
    :param tif_croped_save_dir: 裁剪后保存目录
    :param mask_croped_save_dir:
    :param crop_size: 裁剪尺寸
    :param repetition_rate: 重复率
    :return:
    '''
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape

    print("++++++++++\n", img.shape, mask.shape)

    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    if not os.path.exists(tif_croped_save_dir):
        os.makedirs(tif_croped_save_dir)
    if not os.path.exists(mask_croped_save_dir):
        os.makedirs(mask_croped_save_dir)
    new_name = len(os.listdir(tif_croped_save_dir)) + 1
    #  裁剪图片,重复率为RepetitionRate

    for i in range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        for j in range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
            # mask为单波段，shape只有2维
            mask_cropped = mask[
                           int(i * crop_size * (1 - repetition_rate)): int(
                               i * crop_size * (1 - repetition_rate)) + crop_size,
                           int(j * crop_size * (1 - repetition_rate)): int(
                               j * crop_size * (1 - repetition_rate)) + crop_size]

            cropped = img[
                      int(i * crop_size * (1 - repetition_rate)): int(
                          i * crop_size * (1 - repetition_rate)) + crop_size,
                      int(j * crop_size * (1 - repetition_rate)): int(
                          j * crop_size * (1 - repetition_rate)) + crop_size,:]

            # 写图像
            cv2.imwrite(tif_croped_save_dir + "/%d_sat.jpg" % new_name, cropped)
            cv2.imwrite(mask_croped_save_dir + "/%d_mask.png" % new_name, mask_cropped)
            #  文件名 + 1
            new_name = new_name + 1
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
            index = tif_file.split('_')[0]
            mask_file_path = os.path.join(masks_dir, f'{index}_mask.png')
            tif_crop(tif_file_path, mask_file_path, tifs_croped_save_dir, masks_croped_save_dir, crop_size,
                     repetition_rate)


if __name__ == '__main__':
    tifs_dir = 'E:\workspace\sandiao\data\Road/train\imgs'
    masks_dir = 'E:\workspace\sandiao\data\Road/train\masks'
    tifs_croped_save_dir = 'E:\workspace\sandiao\data\Road/train/imgs_temp'
    masks_croped_save_dir = 'E:\workspace\sandiao\data\Road/train/masks_temp'
    tif_crop_all(tifs_dir, masks_dir, 512, tifs_croped_save_dir, masks_croped_save_dir)
