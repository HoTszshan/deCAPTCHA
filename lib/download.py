# Download CAPTCHAS
# -*- coding:utf-8 -*-

import urllib
import urllib2
import requests
import re
import os
from multiprocessing.dummy import Pool as ThreadPool

class CaptchaSpider:

    targetDir = None
    userAgent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = { 'User-Agent' : userAgent }

    def __init__(self, captchaURL, folder=None, fileType='.jpg', sys='XOS'):
        self.siteURL = captchaURL
        self.fileType = fileType
        #self.userAgent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
        #self.headers = { 'User-Agent' : self.userAgent }
        if sys == 'Windows':
            self.sys_split = '\\'
        else:
            self.sys_split = '/'
        self.set_target_folder(self.__get_library_name() if not folder else folder)

    def __get_library_name(self):
        pattern = re.compile('://(.*?)\.(.*?)\.(.*?)/')
        name = re.findall(pattern, self.siteURL)[0]
        return name[1]

    def set_target_folder(self, folder):
        data_folder = os.path.abspath('..') + self.sys_split + 'data'
        if not os.path.isdir(data_folder):
            os.mkdir(data_folder)
        CaptchaSpider.targetDir = data_folder + self.sys_split + folder
        if not os.path.isdir(CaptchaSpider.targetDir):
            os.mkdir(CaptchaSpider.targetDir)

    def get_page_content(self):
        try:
            request = urllib2.Request(self.siteURL, headers=self.headers)
            response = urllib2.urlopen(request)
            return response.read()
        except urllib2.HTTPError, e:
            raise ValueError(e.code)
        except urllib2.URLError, e:
            raise ValueError(e.reason)

    def __save_img(self, fileName):
        data = self.get_page_content()
        filePath = CaptchaSpider.targetDir + self.sys_split + str(fileName) + self.fileType
        f = open(filePath, "wb")
        f.write(data)
        f.close()

    def download_images(self, number=100):
        pool = ThreadPool(8)
        pool.map(self.__save_img, range(1, number+1))
        pool.close()
        pool.join()


class WebPageSpider(CaptchaSpider):

    def __init__(self, siteURL, folder=None, fileType='.jpg', sys='XOS'):
        self.siteURL = siteURL
        self.fileType = fileType
        if sys == 'Windows':
            self.sys_split = '\\'
        else:
            self.sys_split = '/'
        self.set_target_folder(self.__get_web_library_name() if not folder else folder)

    def __get_web_library_name(self):
        pattern = re.compile('/(\w+)')#('/(.*?)')
        name = re.findall(pattern, self.siteURL)
        return name[-1] + '-' + name[-2] if len(name) > 1 else name[-1]

    def __interpret_content(self, pattern, content):
        items = re.findall(pattern, content)
        return items

    def __get_image_content(self, url):
        imageURL = self.siteURL + url
        try:
            request = urllib2.Request(imageURL, headers=CaptchaSpider.headers)
            response = urllib2.urlopen(request)
            return response.read()
        except urllib2.HTTPError, e:
            raise ValueError(e.code)
        except urllib2.URLError, e:
            raise ValueError(e.reason)

    def __download_a_image(self, info):
        imgURL, imgName = info
        data = self.__get_image_content(imgURL)
        filePath = CaptchaSpider.targetDir + self.sys_split + str(imgName) + self.fileType
        files_list = [os.path.join(self.targetDir, f) for f in os.listdir(self.targetDir)
                      if f.endswith(str(imgName)+'.jpg') or f.split('_')[0] == str(imgName)]
        filePath = CaptchaSpider.targetDir + self.sys_split + str(imgName) + '_' \
                   + str(len(files_list)) + self.fileType if len(files_list) > 0 else filePath
        f = open(filePath, "wb")
        f.write(data)
        f.close()

    def download_captcha_images(self, number=100, pattern=None):
        webPattern = re.compile(r'<p><img src=\"(.*?)\">\n'
                             r'<p>Final choice of words: (\S*)') if not pattern else pattern
        content = self.get_page_content()
        imgInfoList = self.__interpret_content(webPattern, content)
        imgInfoList = imgInfoList[:number] if number < len(imgInfoList) else imgInfoList
        pool = ThreadPool(8)
        pool.map(self.__download_a_image, imgInfoList)
        pool.close()
        pool.join()


"""
spider = CaptchaSpider(captchaURL='https://www.oschina.net/action/user/captcha')
spider.download_images(number=500)
#"""
"""
spider = WebPageSpider('http://www2.cs.sfu.ca/~mori/research/gimpy/ez/')
spider.download_captcha_images(number=200)
#"""



