from osgeo import gdal
from osgeo import gdalconst
import sys

# def setImageInfo(datasetSrc,datasetDst):
#     geotansform = datasetSrc.GetGeoTransform()
#     sProj = datasetSrc.GetProjection()
#
#     datasetDst.SetProjection(sProj)
#     datasetDst.SetGeoTransform(geotansform)
#
# if __name__ == '__main__':
#     gdal.AllRegister()
#     fnSrc = r'C:\Users\yi\Desktop\shui\12.tif'
#     fnDst = r'C:\Users\yi\Desktop\shui\pred_shui.tif'
#     datasetSrc = gdal.Open(fnSrc, gdalconst.GA_ReadOnly)
#     datasetDst = gdal.Open(fnDst,gdalconst.GA_Update)
#
#     del datasetSrc
#     del datasetDst

sPathSrc = r'f:/dataset\三调\多分类\测试\130102长安区\13010201长安区\test.tif'
sPathDst = r'f:/dataset\三调\多分类\测试\130102长安区_result\ClassResult.tif'
dataSrc = gdal.Open(sPathSrc,gdalconst.GA_ReadOnly)
dataDst = gdal.Open(sPathDst,gdalconst.GA_Update)
dGeoTrans = dataSrc.GetGeoTransform()
sProj = dataSrc.GetProjectionRef()
dataDst.SetGeoTransform(dGeoTrans)
dataDst.SetProjection(sProj)

del dataSrc
del dataDst
