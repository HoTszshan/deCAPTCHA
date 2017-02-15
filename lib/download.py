# Download CAPTCHAS
# -*- coding:utf-8 -*-

import urllib
import urllib2
import requests
import re
import os
from multiprocessing.dummy import Pool as ThreadPool

class CaptchaSpider:

    def __init__(self, captchaURL, folder=None, fileType='.jpg', sys='XOS'):
        self.siteURL = captchaURL
        self.fileType = fileType
        self.userAgent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
        self.headers = { 'User-Agent' : self.userAgent }
        if sys == 'Windows':
            self.sys_split = '\\'
        else:
            self.sys_split = '/'
        self.__set_target_folder(self.__get_library_name() if not folder else folder)

    def __get_library_name(self):
        pattern = re.compile('://(.*?)\.(.*?)\.(.*?)/')
        name = re.findall(pattern, self.siteURL)[0]
        return name[1]

    def __set_target_folder(self, folder):
        self.targetDir = os.path.abspath('..') + self.sys_split + 'data' + self.sys_split + folder
        print self.targetDir
        if not os.path.isdir(self.targetDir):
            os.mkdir(self.targetDir)

    def __get_page_content(self):
        request = urllib2.Request(self.siteURL, headers=self.headers)
        response = urllib2.urlopen(request)
        return response.read()

    def __save_img(self, fileName):
        data = self.__get_page_content()
        filePath = self.targetDir + self.sys_split + str(fileName) + self.fileType
        f = open(filePath, "wb")
        f.write(data)
        f.close()

    def download_images(self, number=100):
        pool = ThreadPool(8)
        pool.map(self.__save_img, range(1, number))
        pool.close()
        pool.join()




spider = CaptchaSpider(captchaURL='https://www.oschina.net/action/user/captcha')
spider.download_images()