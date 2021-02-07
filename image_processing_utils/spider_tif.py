import requests
import re
import os
from multiprocessing import Pool


def download(url, save_dir):
    conetent = requests.get(url).content
    filename = url.split('/')[-1]
    file_path = os.path.join(save_dir, filename)
    if os.path.exists(file_path):
        print(f'{filename}已经下载过')
        return
    with open(os.path.join(save_dir, filename), 'wb') as f:
        try:
            print(f'下载{filename}中。。。')
            f.write(conetent)
            print(f'{filename}下载完成')
        except Exception as e:
            print(e)


def main():
    html = requests.get('http://data.ess.tsinghua.edu.cn/fromglc10_2017v01.html').content.decode()
    urls = re.findall(r'.tif</td><td><a href="(.*)">from', html)
    save_dir = 'E:/spider_download'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pool = Pool(8)
    for url in urls:
        pool.apply_async(func=download, args=(url, save_dir))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
