import logging
import os

import torch
from PIL import Image
from tqdm import tqdm
from osgeo import gdal, ogr
import numpy as np

Image.MAX_IMAGE_PIXELS = None


class Sandiao_Test(object):
    model1_path = r'models\sandiao-no-dl.pt'
    model2_path = r'models\sandiao-dl.pt'

    color_list = [
        0, 0, 0,  # 黑色，背景
        255, 0, 0,  # 红色，建筑
        255, 255, 255,  # 白色，道路
        0, 255, 0,  # 绿色，耕地
        32, 64, 0,  # 深绿色，林地
        32, 64, 0,  # 深绿色，植被
        0, 0, 255,  # 蓝色，湿地
        0, 0, 255,  # 蓝色，水域
    ]

    def __init__(self):
        self.logger = self.get_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model1 = torch.jit.load(self.model1_path).to(self.device)
        self.logger.info(f'模型1：{self.model1_path}加载完成')
        self.model2 = torch.jit.load(self.model2_path).to(self.device)
        self.logger.info(f'模型2：{self.model2_path}加载完成')

    def __call__(self, image_path, output_dir):
        img_array = self.read_image(image_path)
        self.sliding_window_detection(img_array, output_dir)

    def read_image(self, img_path):
        """
        读取原始大图像，返回图像数组darray
        :param img_path:
        :return:
        """
        try:
            img = gdal.Open(img_path)
            img_width = img.RasterXSize
            img_height = img.RasterYSize
            img = img.ReadAsArray(0, 0, img_width, img_height)
            self.filename = os.path.basename(img_path)  # 图片文件名
            self.logger.info(f'{img_path} 图片读取成功,图片尺寸：{img.shape}')
            return img
        except Exception as e:
            self.logger.info(f'{img_path} 图片无法读取,错误信息：{str(e)}')

    def sliding_window_detection(self, image_array, output_dir, window_size=512):
        """
        滑动窗口对原始图像进行剪裁，并检测，将预测结果保存到指定文件夹
        :param image_array:
        :param window_size:
        :param output_dir:
        :return:
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        height, width = image_array.shape[1:]
        mask = np.zeros([height, width], dtype=np.uint8)
        self.logger.info(f'顺序扫描检测检测中。。。')
        # 按照从左往右，从上往下，逐行扫描
        for i in tqdm(range(height // window_size)):
            for j in range(width // window_size):
                img_cropped = image_array[
                              :,
                              i * window_size: (i + 1) * window_size,
                              j * window_size: (j + 1) * window_size,
                              ]
                img_pred = self.predict_img(img_cropped)  # 得到预测图
                mask[i * window_size: (i + 1) * window_size,
                j * window_size: (j + 1) * window_size] = img_pred  # 拼接到mask
        self.logger.info(f'扫描最后一列中。。。')
        # 扫描最后一列
        for i in tqdm(range(height // window_size)):
            img_cropped = image_array[:, i * window_size:(i + 1) * window_size, (width - window_size):width]
            img_pred = self.predict_img(img_cropped)
            mask[i * window_size:(i + 1) * window_size, (width - window_size):width] = img_pred
        self.logger.info('扫描最后一行中。。。')
        # 扫描最后一行
        for j in tqdm(range(width // window_size)):
            img_cropped = image_array[:, (height - window_size):height, j * window_size:(j + 1) * window_size]
            img_pred = self.predict_img(img_cropped)
            mask[(height - window_size):height, j * window_size:(j + 1) * window_size] = img_pred
        self.logger.info('扫描右下角')
        # 扫描右下角
        img_cropped = image_array[:, (height - window_size):height, (width - window_size):width]
        img_pred = self.predict_img(img_cropped)
        mask[(height - window_size):height, (width - window_size):width] = img_pred
        self.logger.info(f'检测完成，预测图像保存中。。。')
        img_name, img_type = self.filename.split('.')
        mask_filename = img_name + '_predict.' + img_type  # 与原始图像保持类型一致
        save_path = os.path.join(output_dir, mask_filename)
        mask2color = Image.fromarray(mask, mode='L')
        mask2color.putpalette(self.color_list)
        mask2color.save(save_path)
        self.logger.info(f'预测图像已保存到：{save_path}')
        # self.logger.info(f'mask转shape中。。。')
        # self.mask2shape(mask, output_dir)
        # self.logger.info('检测结束！')

    def predict_img(self, img_cropped):
        """
        将当前扫描的图片输入到模型进行预测，返回预测结果
        :param img_cropped:
        :return:
        """
        image = self.img2tensor(img_cropped)
        r = torch.argmax(self.model1(image), dim=1, keepdim=True)
        r2 = torch.sigmoid(self.model2(image))
        r2 = torch.where(r2 > 0.5, 1, 0)
        mask = (r2 == 1)
        r[mask] = 2
        r = r.squeeze(0).squeeze(0)
        r = r.cpu().detach().numpy()
        return r

    def img2tensor(self, image):
        """
        模型输入图像预处理
        :param image:
        :return:
        """
        image = image / 255
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = image.to(device=self.device, dtype=torch.float32)
        image = image.unsqueeze(0)
        return image

    def get_logger(self):
        """
        创建日志对象
        :return:
        """
        logger = logging.getLogger()
        fh = logging.FileHandler('logs/test.log', mode='w', encoding='utf-8')
        ch = logging.StreamHandler()
        logger.addHandler(fh)
        logger.addHandler(ch)
        format = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        fh.setFormatter(format)
        logger.setLevel(logging.DEBUG)
        return logger

    def mask2shape(self, mask, output_dir):
        """
        mask转shape文件
        :param mask: mask数组darray
        :param output_dir: 输出目录
        :return:
        """
        mask = mask.astype(np.int16)
        row = mask.shape[0]  # 行数
        columns = mask.shape[1]  # 列数
        dim = 1  # 通道数
        geoTrans = [0.0, 1.0, 0.0, float(row), 0.0, -1.0]
        # 创建驱动
        driver = gdal.GetDriverByName('MEM')
        # 创建文件
        img_name, img_type = self.filename.split('.')
        # mask_filename = img_name + '_mask.' + img_type
        # mask_path = os.path.join(output_dir, mask_filename)
        dst_ds = driver.Create('', columns, row, dim)
        # 设置几何信息
        dst_ds.SetGeoTransform(geoTrans)
        # 将数组写入
        dst_ds.GetRasterBand(1).WriteArray(mask)
        # 转shape
        srcband = dst_ds.GetRasterBand(1)
        # maskband = srcband.GetMaskBand()
        drv = ogr.GetDriverByName('ESRI Shapefile')
        shape_filename = img_name + '.shape'
        shape_path = os.path.join(output_dir, shape_filename)
        if os.path.exists(shape_path):
            drv.DeleteDataSource(shape_path)
        dst_shpDs = drv.CreateDataSource(shape_path)
        srs = None
        dst_layername = img_name
        dst_layer = dst_shpDs.CreateLayer(dst_layername, srs=srs)
        dst_fieldname = 'class'
        fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
        dst_layer.CreateField(fd)
        dst_field = 0
        prog_func = 0
        options = []
        # 参数输入栅格图像波段\掩码图像波段、矢量化后的矢量图层、需要将DN值写入矢量字段的索引、算法选项、进度条回调函数、进度条参数
        gdal.Polygonize(srcband, None, dst_layer, dst_field, options, callback=prog_func)
        self.logger.info(f'shape文件已保存到：{shape_path}')


if __name__ == '__main__':
    test = Sandiao_Test()
    test(
        image_path=r'H:\资料\总结\图片\道路测试原图.png',
        output_dir=r'H:\资料\总结\图片'
    )
