import json
import os
from osgeo import gdal, ogr
import cv2
import numpy as np
import tifffile
import torch
from flask import Flask, request
from werkzeug.utils import secure_filename
from urllib.parse import unquote
import logging

logging.basicConfig(level=logging.INFO)
model_dir = os.path.join(os.path.dirname(__file__), 'models')
class_names_dict = {
    'sandiao-jmd.pt': ['背景', '建筑物'],
    'sandiao-dl.pt': ['背景', '道路'],
    'xuexian.pt': ['background', 'shadow', 'cloud', 'snow'],
    'haiyu.pt': ['背景', '沙滩', '滩涂', '红树林', '岛体', '水体'],
    'sandiao-multi.pt': ['无分类', '建筑', '道路', '耕地', '林地', '植被', '湿地', '水域'],

}
class_names = None

app = Flask(__name__)
net = None  # 已加载权重的分割模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    if class_names == class_names_dict.get('xuexian.pt') or class_names == class_names_dict.get('haiyu.pt'):
        image = tifffile.imread(image_path)
    elif class_names == class_names_dict.get('sandiao-jmd.pt'):
        image = cv2.imdecode(np.fromfile(file=image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imdecode(np.fromfile(file=image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = image.astype(np.float32) / 255
    logging.info(image.shape)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.to(device=device, dtype=torch.float32)

    if len(image.shape) == 2:  # 图片为单通道
        image = image.unsqueeze(0)
    elif len(image.shape) == 3:  # 图片为多通道
        image = image.permute((2, 0, 1))
    image = image.unsqueeze(0)
    logging.info(image.shape)
    global net
    if not net:
        raise Exception("Model is not loaded")
    try:
        r = net(image)
        if r.shape[1] == 1:
            r = torch.sigmoid(r)
            r = torch.where(r > 0.5, 1, 0)
        else:
            r = torch.argmax(r, dim=1, keepdim=True)
    except Exception as e:
        logging.info(e)
        raise Exception('Model does not match the picture')
    r = r.squeeze(0).squeeze(0)
    r = r.cpu().detach().numpy()  # mask:(512,512)
    # r = cv2.resize(r.astype(np.uint8),(514,514))
    logging.info(r.shape)
    logging.info(class_names)
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
