import json
import os
from functools import reduce
import cv2
import numpy as np
import tifffile
from skimage.measure import find_contours
import torch
from flask import Flask, request
from werkzeug.utils import secure_filename
from osgeo import gdal, ogr
from urllib.parse import unquote
import logging

# from model.unet import UNet
# from model.dinknet import DinkNet34
from model.deeplabv3 import DeepLabV3

logging.basicConfig(level=logging.INFO)
model_dir = os.path.join(os.path.dirname(__file__), 'models')
logging.info(model_dir)
image_dir = os.path.join(os.path.dirname(__file__), 'images')
class_names_dict = {
    # 'sandiao-jmd.pt': ['背景', '建筑物'],
    # 'net.pt': ['无分类','道路'],
    # 'net': ['background', 'water'],
    # 'net': ['background', 'shadow', 'cloud', 'snow'],
    # 'net': ['背景','沙滩', '滩涂', '红树林', '岛体', '水体']
    'net.pt': ['无分类', '建筑', '道路', '耕地', '林地', '植被', '湿地', '水域'],

}
labels = set(reduce(lambda x, y: x + y, class_names_dict.values()))
# class_names = None
class_names = class_names_dict.get('net.pt')

app = Flask(__name__)
# net = None  # 已加载权重的分割模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = DeepLabV3.load_from_checkpoint(
    r'../lightning_logs\version_5\checkpoints\sandiao-multi_class-deeplabv3-os8-epoch=0002-val_acc=0.9102.ckpt')
net.to(device)
net.eval()
logging.info('net1 加载成功')
net2 = torch.load(r'../models\road_rgb_dinknet34_0111.pt')
net2.to(device)
net2.eval()
logging.info('net2 加载成功')


def json_desc_instances(mask, class_names):
    features = []
    for j in range(len(class_names)):
        label = class_names[j]
        mask_single = np.where(mask == j, 1, 0)
        padded_mask = np.zeros(
            (mask_single.shape[0] + 2, mask_single.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask_single
        contours = find_contours(padded_mask, 0.5)
        for coordinates in contours:
            geometry = {
                'type': 'Polygon',
                'coordinates': np.expand_dims(coordinates, 0).tolist(),
                'class': label
            }
            feature = dict(type='Feature', geometry=geometry)
            features.append(feature)

    return json.dumps(dict(features=features), ensure_ascii=False)


def adjust_json(feature):
    class_id = feature.get('properties').get('class_id')
    feature['geometry']['class'] = class_names[class_id]
    return feature


def mask2json(mask):
    memdrv = gdal.GetDriverByName('MEM')
    src_ds = memdrv.Create('', mask.shape[1], mask.shape[0], 1)
    geom = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    src_ds.SetGeoTransform(geom)
    band = src_ds.GetRasterBand(1)
    band.WriteArray(mask)
    dst_layername = "results"
    drv = ogr.GetDriverByName("geojson")
    dst_ds = drv.CreateDataSource('results.json')
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=None)
    fieldName = 'class_id'
    fd = ogr.FieldDefn(fieldName, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 0
    gdal.Polygonize(band, None, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    logging.info(f'results.json已保存')
    with open('results.json', 'r') as f:
        res = f.read()
        res = json.loads(res)
        features = res.get('features')
        res['features'] = list(map(adjust_json, features))
        logging.info(res)
    return json.dumps(res, ensure_ascii=False)


def detect_image(image_path=None):
    # image = cv2.imdecode(np.fromfile(file=image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = tifffile.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.float32) / 255
    print(image.shape)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.to(device=device, dtype=torch.float32)

    if len(image.shape) == 2:  # 图片为单通道
        image = image.unsqueeze(0)
    elif len(image.shape) == 3:  # 图片为多通道
        image = image.permute((2, 0, 1))

    image = image.unsqueeze(0)
    print(image.shape)
    global net
    if not net:
        raise Exception("Model is not loaded")
    try:
        r = torch.argmax(net(image), dim=1, keepdim=True)
        r2 = torch.sigmoid(net2(image))
        r2 = torch.where(r2 > 0.5, 1, 0)
        mask = (r2 == 1)
        r[mask] = 2
    except Exception as e:
        print(e)
        raise Exception('Model does not match the picture')
    r = r.squeeze(0).squeeze(0)
    r = r.cpu().detach().numpy()
    # r = np.transpose(r,(1, 0))  # h,w 转换成w,h
    print(r.shape)
    print(class_names)
    # ret = json_desc_instances(r, class_names)
    ret = mask2json(r)
    return ret


@app.route('/api', methods=['GET', 'POST'])
def request_api():
    print('start')
    if request.method == 'GET':
        image_path = request.args.get("image", type=str, default='')
        print(image_path)

    if request.method == 'POST':
        f = request.files['file']
        print(f)
        # fname = secure_filename(f.filename)
        # ext = fname.rsplit('.', 1)[1]
        # new_filename = Pic_str().create_uuid() + '.' + ext
        # image_path = os.path.join(image_dir, new_filename)
        # if not os.path.exists(image_dir):
        #     os.makedirs(image_dir)
        # f.save(image_path)
        # print(image_path)
        # if os.path.exists(image_path):
        # try:
        ret_json = detect_image(f)
        print(ret_json)
        return ret_json
        # except Exception as e:
        #     print(e)
        #     return str(e)
    return "Invalid request"


@app.route('/model', methods=['GET', 'POST'])
def request_model():
    print('开始上传模型：')
    global model_path
    if request.method == 'GET':
        model_path = request.args.get("models", type=str, default='')
    if request.method == 'POST':
        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                logging.info(f'创建目录{model_dir}成功')
            f = request.files['file']
            fname = secure_filename(unquote(f.filename, encoding='gbk'))
            logging.info(fname)
            global class_names
            class_names = class_names_dict.get(fname)
            model_path = os.path.join(model_dir, fname)
            f.save(model_path)
            logging.info(f'文件：{model_path}保存成功')
            status = 1
            info = '上传成功！'
        except Exception as e:
            status = 0
            info = f'上传失败，错误信息：{e}'

    if os.path.exists(model_path):
        global net
        # net = torch.load(model_path)
        # net = UNet.load_from_checkpoint(model_path)
        # net = DinkNet34.load_from_checkpoint(model_path)
        net = DeepLabV3.load_from_checkpoint(model_path)
        net.to(device)
        net.eval()
        logging.info('加载模型完成！')
    return json.dumps({'status': status, 'info': info}, ensure_ascii=False)


@app.route('/connect')
def connect_server():
    status, model_list, info = 0, [], '服务端连接失败！'
    if request.method == 'GET':
        status = 1
        info = '服务端连接成功！'
        logging.info(info)
        if os.path.exists(model_dir):
            for model_file in os.listdir(model_dir):
                if model_file.endswith(('.pt', '.pth')):
                    model_list.append(model_file)

    res = {
        'status': status,
        'info': info,
        'model_list': model_list
    }
    res_json = json.dumps(res, ensure_ascii=False)
    logging.info(f'返回数据：{res_json}')
    return res_json


@app.route('/load_model', methods=['GET', 'POST'])
def load_model():
    status = 0
    if request.method == 'GET':
        try:
            model_name = request.args.get('model_name')
            global class_names
            class_names = class_names_dict.get(model_name, [])
            logging.info(class_names)
            model_path = os.path.join(model_dir, model_name)
            if os.path.exists(model_path):
                global net
                net = torch.jit.load(model_path)
                # net = UNet.load_from_checkpoint(model_path)
                # net = DinkNet34.load_from_checkpoint(model_path)
                # net = DeepLabV3.load_from_checkpoint(model_path)
                net.to(device)
                net.eval()
                logging.info(f'模型：{model_name} 加载完成')
                status = 1
                info = f'模型：{model_name} 加载完成'
        except Exception as e:
            logging.info(e)
            info = f'模型加载失败，错误代码：{e}'

    return json.dumps({'status': status, 'info': info, 'class_list': class_names}, ensure_ascii=False)


@app.route('/', methods=['GET', 'POST'])
def index():
    return 'KanqGis AI cloud'


############################################################

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8088, threaded=False)
