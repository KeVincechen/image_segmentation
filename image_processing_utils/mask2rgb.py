import numpy as np
import tifffile


def colormap():
    cmap = []
    cmap.append([0, 0, 0])
    cmap.append([224, 64, 64])  # 红色
    cmap.append([96, 96, 96])  # 灰色
    cmap.append([96, 160, 32])  # 浅绿色
    cmap.append([32, 64, 0])  # 深绿色
    cmap.append([192, 192, 64])  # 橙黄色
    cmap.append([128, 164, 164])  # 灰绿色
    cmap.append([32, 224, 224])  # 浅蓝色
    cmap.append([255, 255, 32])  # 金黄色
    return cmap


def mask2rgb(gray_image):
    cmap = colormap()
    shape = gray_image.shape
    color_image = np.zeros((shape[0], shape[1], 3)).astype(np.uint8)

    for label in range(len(cmap)):
        mask = gray_image == label
        color_image[mask] = cmap[label]

    return color_image

def rgb2mask(rgb_imgge):
    cmap = colormap()
    for label in range(len(cmap)):
        mask = (rgb_imgge == cmap[label]).all(2)
        rgb_imgge[mask] = label
    gray_image = rgb_imgge[:, :, 0]
    return gray_image

if __name__ == '__main__':
    mask = tifffile.imread('../test.tif')
    mask[mask == 255] = 0
    rgb_img = mask2rgb(mask)
    tifffile.imwrite('aa.tif',rgb_img)

    # rgb_img = tifffile.imread('aa.tif')
    # gray_img = rgb2mask(rgb_img)
    # tifffile.imwrite('aa_gray.tif',gray_img)


