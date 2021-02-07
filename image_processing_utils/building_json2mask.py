import os
import cv2
import numpy as np
import json
import tifffile
from tqdm import tqdm

dict_labels = {
    'background': 0,
    'DL': 64,
    'JMD': 128,
    'SX': 192,
    'ZB': 255
}


def read_DataStru_json(path):
    with open(path, 'r', encoding='utf-8') as load_f:
        strF = load_f.read()
        try:
            datas = json.loads(strF)
        except Exception as e:
            print(e)
            datas = {}
    return datas


def poly_mask(myjson, mask, j):
    a = myjson['features'][j]['geometry']['coordinates'][0]  # a是标记的点
    label = myjson['features'][j]['properties']['OBJECTID_2']
    a = np.array([[int(c[0]), int(c[1])] for c in a])
    b = np.array(a, dtype=np.int32)
    roi_t = []
    for i in range(len(a)):
        roi_t.append(b[i])
    roi_t = np.asarray(roi_t)
    roi_t = np.expand_dims(roi_t, axis=0)
    cv2.polylines(mask, roi_t, True, dict_labels.get(label, 0))
    cv2.fillPoly(mask, roi_t, dict_labels.get(label, 0))


def json2mask(json_file, save_mask, img):
    if np.argmin(img.shape) == 0:  # 图片的通道在第一个维度，如(3,xx,xx)
        mask = np.zeros((img.shape[1], img.shape[2]), dtype="uint8")
    else:  # 图片通道在最后一个维度，如(xx,xx,3)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    print("mask:", mask.shape)
    myjson = read_DataStru_json(json_file)
    print('********', type(myjson))
    print('myjson:', len(myjson['features']))
    for j in range(len(myjson['features'])):
        poly_mask(myjson, mask, j)

    basename = os.path.basename(json_file)
    mask_path = os.path.join(save_mask, basename).replace('.json', '.tif')
    cv2.imencode('.tif', mask)[1].tofile(mask_path)  # 解决写文件名字中文乱码问题
    # cv2.imwrite(mask_path, mask)
    # tifffile.imwrite(mask_path, mask)
    print('json to mask complete!')


def json2mask_all(imgs_dir, jsons_dir, masks_save_dir):
    '''
    批量将json转成mask
    :param imgs_dir:
    :param jsons_dir:
    :param masks_save_dir:
    :return:
    '''
    if not os.path.exists(masks_save_dir):
        os.makedirs(masks_save_dir)
    for root, dirs, files in os.walk(imgs_dir):
        for img_file in tqdm(files):
            img_file_path = os.path.join(root, img_file)
            json_file_path = os.path.join(jsons_dir, img_file.split('.')[0] + '.json')
            img = tifffile.imread(img_file_path)
            print(img.shape)
            json2mask(json_file_path, masks_save_dir, img)


if __name__ == '__main__':
    imgs_dir = 'E:\workspace\sandiao\data\Multi-class/total_imgs'
    jsons_dir = 'E:\workspace\sandiao\data\Multi-class\jsons'
    save_path = 'E:\workspace\sandiao\data\Multi-class/total_masks'
    json2mask_all(imgs_dir, jsons_dir, save_path)
