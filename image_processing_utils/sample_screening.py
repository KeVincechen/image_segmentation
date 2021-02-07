import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm
import numpy as np
import tifffile


def sample_screening(img_dir, label_dir):
    '''
    删掉有效面积不合格的样本
    :param img_dir:
    :param label_dir:
    :return:
    '''
    name_list = os.listdir(label_dir)
    cn = 0

    for file in tqdm(name_list):
        label_path = os.path.join(label_dir, file)
        img_path = os.path.join(img_dir, file)
        img = tifffile.imread(label_path)
        img[img == 255] = 0

        # img1 = np.where(img==0,1,0)
        # img2 = np.where(img==1,1,0)
        # img3 = np.where(img ==2,1,0)
        area_useful = np.sum(img)
        if area_useful == 0:
            try:
                os.remove(img_path)
                os.remove(label_path)
            except Exception as e:
                print(e)
            else:
                cn = cn + 1
                print(f'{file} 样本不合格,已删除该样本！')

    print(f'共删除{cn}组不合格样本。')


def remove_some_imgs(imgs_dir, masksdir, set_files_index):
    '''
    同步删除img和mask文件夹下的某些图片
    :param imgs_dir:
    :param masksdir:
    :param set_files_index: 将要删除的图片的序号集合
    :return:
    '''
    count = 0
    for index in tqdm(set_files_index):
        img_file_path = os.path.join(imgs_dir, f'4207042019BJ2DOM-{index}.tif')
        mask_file_path = os.path.join(masksdir, f'4207042019BJ2DOM-{index}.tif')
        try:
            os.remove(mask_file_path)
            os.remove(img_file_path)
            count += 1

        except Exception as e:
            print(e)
    print(f'成功删除{count}组训练图片')


def batch_rename(imgs_dir, masks_dir, suffix=''):
    '''
    批量给imgs和masks文件夹下的图片同步的按顺序重命名
    :param imgs_dir:
    :param masks_dir:
    :return:
    '''
    index = 0
    for img_file in tqdm(os.listdir(imgs_dir)):
        img_file_path = os.path.join(imgs_dir, img_file)
        img_file_path_new = os.path.join(imgs_dir, f'{index + 1}{suffix}.tif')
        mask_file_path = os.path.join(masks_dir, img_file)
        mask_file_path_new = os.path.join(masks_dir, f'{index + 1}{suffix}.tif')
        try:
            os.rename(img_file_path, img_file_path_new)
            os.rename(mask_file_path, mask_file_path_new)
            index += 1
        except Exception as e:
            print(e)
    print(f'共修改{index}组图片的文件名')


def batch_copy(imgs_dir, masks_dir, imgs_train, masks_train, set_img_index):
    '''
    批量将imgs和masks文件夹下的某些图片同步移动到训练集文件夹下，并且重命名
    :param imgs_dir:
    :param masks_dir:
    :param imgs_train:
    :param masks_train:
    :param set_img_index: 图片文件名中的编号
    :return:
    '''
    new_file_index = len(os.listdir(imgs_train))
    count = 0
    for index in tqdm(set_img_index):
        img_path = os.path.join(imgs_dir, f'420702GF2DOM-{index}.tif')
        img_path_new = os.path.join(imgs_train, f'{new_file_index + 1}.tif')
        mask_path = os.path.join(masks_dir, f'420702GF2DOM-{index}.tif')
        mask_path_new = os.path.join(masks_train, f'{new_file_index + 1}.tif')
        try:
            shutil.copyfile(img_path, img_path_new)
            shutil.copyfile(mask_path, mask_path_new)
            new_file_index += 1
            count += 1
        except Exception as e:
            print(e)
    print(f'成功复制{count}组训练图片')


def split_img_mask(train_data_dir, imgs_train_dir, masks_train_dir):
    '''

    :param train_data_dir:
    :param imgs_train_dir:
    :param masks_train_dir:
    :return:
    '''
    if not os.path.exists(imgs_train_dir):
        os.makedirs(imgs_train_dir)
    if not os.path.exists(masks_train_dir):
        os.makedirs(masks_train_dir)

    count = 0
    for file in tqdm(os.listdir(train_data_dir)):

        file_path = os.path.join(train_data_dir, file)
        try:
            if file.endswith('.png'):
                shutil.copy(file_path, masks_train_dir)
            elif file.endswith('.tif'):
                shutil.copy(file_path, imgs_train_dir)
            count += 1
        except Exception as e:
            print(e)
    print(f'共复制{count}组训练图片')


def find_difference(imgs_dir, masks_dir):
    '''
    快速找出img和mask文件夹下的名字不对应的图片，并删除
    :param imgs_dir:
    :param masks_dir:
    :return:
    '''
    count1, count2 = 0, 0
    for img_file in tqdm(os.listdir(imgs_dir)):
        if not os.path.exists(os.path.join(masks_dir, img_file)):
            img_path = os.path.join(imgs_dir, img_file)
            os.remove(img_path)
            count1 += 1
            print(f'删除img图片：{img_path}')
    for mask_file in tqdm(os.listdir(masks_dir)):
        if not os.path.exists(os.path.join(imgs_dir, mask_file)):
            mask_path = os.path.join(masks_dir, mask_file)
            os.remove(mask_path)
            count2 += 1
            print(f'删除mask图片：{mask_path}')
    print(f'共删除{count1}张img图片，{count2}张mask图片')


def mask2render(masks_dir, renders_dir, class_num):
    '''
    将mask图片转换成灰度渲染图片，方便检查样本标注错误
    :param masks_dir: mask图片所在目录
    :param renders_dir: 灰度渲染图片输出目录
    :param class_num: 总分类数（不包括背景）
    :return:
    '''
    if not os.path.exists(renders_dir):
        os.makedirs(renders_dir)
        print(f'创建{renders_dir}成功')
    for mask_file in tqdm(os.listdir(masks_dir)):
        mask_file_path = os.path.join(masks_dir, mask_file)
        mask = tifffile.imread(mask_file_path)
        mask[mask == 255] = 0  # 原始mask中背景值为255，转换成0
        interval = 255 // class_num  # 灰度值间隔
        mask = mask * interval
        mask_render_path = os.path.join(renders_dir, mask_file)
        tifffile.imwrite(mask_render_path, mask)


def merge_img_render(imgdir, maskdir, dist_dir):
    '''
    将img与maks渲染后的render图片进行水平拼接，方便检查样本标注错误
    :param all_dirs: img和render所在目录
    :param dist_dir: 拼接后输出目录
    :return:
    '''
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    imglists = os.listdir(imgdir)
    for imgname in tqdm(imglists):
        imgpath = os.path.join(imgdir, imgname)
        maskpath = os.path.join(maskdir, imgname)

        img = cv2.imread(imgpath)
        mask = cv2.imread(maskpath)
        hstack_list = []
        hstack_list.append(img)
        hstack_list.append(mask)
        merge_img = np.hstack(hstack_list)
        dstpath = os.path.join(dist_dir, imgname)
        cv2.imwrite(dstpath, merge_img)


def find_false_mask(images_dir, renders_dir):
    '''
    找出无法读取的mask图片
    :param renders_dir:
    :return:
    '''
    images_edit_dir = os.path.join(os.path.dirname(images_dir), 'toedit/images')
    if not os.path.exists(images_edit_dir):
        os.makedirs(images_edit_dir)
    masks_edit_dir = os.path.join(os.path.dirname(images_dir), 'toedit/masks')
    if not os.path.exists(masks_edit_dir):
        os.makedirs(masks_edit_dir)
    error_list = []
    for file in os.listdir(renders_dir):
        image_path = os.path.join(images_dir, file)
        render_path = os.path.join(renders_dir, file)
        mask_path = os.path.join(os.path.dirname(images_dir), 'masks', file)

        try:
            mask = cv2.imdecode(np.fromfile(render_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            print(f'{file}:{mask.shape}\n{np.unique(mask)}')
        except Exception as e:
            print(e)
            error_list.append(file)
            shutil.copy(mask_path, masks_edit_dir)
            shutil.copy(image_path, images_edit_dir)
            os.remove(image_path)
            os.remove(render_path)
    print(error_list)


def sample_is_valid(mask_array):
    '''
    判断样本是否有效
    :return:
    '''
    area_useful = np.sum(mask_array // 255)
    # area_useful = np.sum(mask_array)
    return area_useful > 0


def create_one_class_data(raw_image_dir, new_image_dir, raw_mask_dir, new_mask_dir, class_id=4):
    """
    根据class_id值，从多分类标签中提取单个分类作为标签
    :param raw_image_dir:
    :param new_image_dir:
    :param raw_mask_dir:
    :param new_mask_dir:
    :param class_id:
    :return:
    """
    os.makedirs(new_image_dir, exist_ok=True)
    os.makedirs(new_mask_dir, exist_ok=True)
    raw_mask_dir, new_mask_dir = Path(raw_mask_dir), Path(new_mask_dir)
    raw_image_dir, new_image_dir = Path(raw_image_dir), Path(new_image_dir)
    for filename in tqdm(os.listdir(raw_mask_dir)):
        mask_array = cv2.imdecode(np.fromfile(raw_mask_dir / filename, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask_array = np.where(mask_array == 4, 255, 0)
        if sample_is_valid(mask_array):
            shutil.copy(raw_image_dir / filename.replace('.png', '.tif'), new_image_dir)
            cv2.imencode('.png', mask_array)[1].tofile(new_mask_dir / filename)


if __name__ == '__main__':
    raw_image_dir = r'G:\dataset\天池比赛数据集\images'
    new_image_dir = r'G:\dataset\天池比赛数据集/road/images'
    raw_mask_dir = r'G:\dataset\天池比赛数据集\masks'
    new_mask_dir = r'G:\dataset\天池比赛数据集/road/masks'
    create_one_class_data(raw_image_dir, new_image_dir, raw_mask_dir, new_mask_dir)
    # split_img_mask(data_dir, imgs_dir, masks_dir)
    # renders_dir = 'H:\dataset\sandiao/newdata/12.19\冯小敏-1218XW\冯小敏-1218XW\mask/renders'
    # mask2render(masks_dir, renders_dir, 8)

    # dist_dir = 'H:\dataset\sandiao/newdata/1215ecq-fxm\ecq\mask/render2check'
    # batch_rename(imgs_dir,masks_dir)
    # merge_img_render(imgs_dir,renders_dir,dist_dir)
    # find_difference(imgs_dir, masks_dir)

    # set_img_index = {2,4,5,7,9,11,16,}
    # remove_some_imgs(imgs_dir, masks_dir, set_img_index)
    # batch_copy(imgs_dir, renders_dir,
    #            'E:\workspace\sandiao\data\Multi-class/train_data\eight-class\images',
    #            'E:\workspace\sandiao\data\Multi-class/train_data\eight-class/masks',
    #            set_img_index)

    # find_false_mask(imgs_dir, masks_dir)
