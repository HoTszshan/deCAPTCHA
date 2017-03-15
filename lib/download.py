# Download CAPTCHAS
# -*- coding:utf-8 -*-


import urllib2
import httplib
import re
import os
import sys
import time
from multiprocessing.dummy import Pool as ThreadPool

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.action_chains import ActionChains

sys.setrecursionlimit(10000)

FOLDER = 'data'

class CaptchaSpider:

    targetDir = None
    userAgent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers_ = { 'User-Agent' : userAgent }

    def __init__(self, captchaURL, folder=None, fileType='.jpg', curdir_save=True):
        self.siteURL_ = captchaURL
        self.fileType_ = fileType
        self.set_target_folder(self.__get_library_name() if not folder else folder, curdir=curdir_save)

    def __get_library_name(self):
        pattern = re.compile('://(.*?)\.(.*?)\.(.*?)/')
        name = re.findall(pattern, self.siteURL_)[0]
        return name[1]

    def set_target_folder(self, folder, curdir=False):
        if not curdir:
            data_folder = os.path.join(os.path.abspath('..'), FOLDER)
        else:
            data_folder = FOLDER
        if not os.path.isdir(data_folder):
            os.mkdir(data_folder)
        CaptchaSpider.targetDir = os.path.join(data_folder , folder)
        if not os.path.isdir(CaptchaSpider.targetDir):
            os.mkdir(CaptchaSpider.targetDir)

    def get_page_content(self):
        try:
            request = urllib2.Request(self.siteURL_, headers=self.headers_)
            response = urllib2.urlopen(request)
            return response.read()
        except urllib2.HTTPError, e:
            raise ValueError(e.code)
        except urllib2.URLError, e:
            raise ValueError(e.reason)

    def __save_img(self, fileName):
        start_time = time.time()
        data = self.get_page_content()
        while len(data) < 100:
            print("Not a image: %s!" % data)
            time.sleep(10)
            data = self.get_page_content()
        filePath = os.path.join(CaptchaSpider.targetDir, str(fileName) + self.fileType_)
        f = open(filePath, "wb")
        f.write(data)
        f.close()
        finish_time = time.time()
        print('It takes %.4f s to save %s ' % ((finish_time - start_time), os.sep.join(filePath.split('/')[-3:])))
        #print('Save image: ' + filePath)

    def download_images(self, number=100):
        start_time = time.time()
        try:
            pool = ThreadPool(4)
            pool.map(self.__save_img, range(1, number + 1))
            pool.close()
            pool.join()
        except httplib.BadStatusLine:
            print httplib.BadStatusLine
        except ValueError:
            print ValueError
        # except UnboundLocalError:
        #     print UnboundLocalError
        finish_time = time.time()
        print('It takes %.4f s to save %d ' % (finish_time - start_time, number))

class WebPageSpider(CaptchaSpider):

    def __init__(self, siteURL, folder=None, fileType='.jpg', curdir_save=True):
        self.siteURL_ = siteURL
        self.fileType_ = fileType
        self.set_target_folder(self.__get_web_library_name() if not folder else folder, curdir=curdir_save)

    def __get_web_library_name(self):
        pattern = re.compile('/(\w+)')#('/(.*?)')
        name = re.findall(pattern, self.siteURL_)
        return name[-1] + '-' + name[-2] if len(name) > 1 else name[-1]

    def __interpret_content(self, pattern, content):
        items = re.findall(pattern, content)
        return items

    def __get_image_content(self, url):
        imageURL = self.siteURL_ + url
        try:
            request = urllib2.Request(imageURL, headers=CaptchaSpider.headers_)
            response = urllib2.urlopen(request)
            return response.read()
        except urllib2.HTTPError, e:
            raise ValueError(e.code)
        except urllib2.URLError, e:
            raise ValueError(e.reason)

    def __download_a_image(self, info):
        imgURL, imgName = info
        data = self.__get_image_content(imgURL)
        filePath = os.path.join(CaptchaSpider.targetDir, str(imgName) + self.fileType_)
        files_list = [os.path.join(self.targetDir, f) for f in os.listdir(self.targetDir)
                      if f.endswith(str(imgName)+'.jpg') or f.split('_')[0] == str(imgName)]
        filePath = os.path.join(CaptchaSpider.targetDir, str(imgName) + '_'
                   + str(len(files_list)) + self.fileType_) if len(files_list) > 0 else filePath
        f = open(filePath, "wb")
        f.write(data)
        f.close()

    def download_captcha_images(self, number=100, pattern=None):
        webPattern = re.compile(r'<p><img src=\"(.*?)\">\n'
                             r'<p>Final choice of words: (\S*)') if not pattern else pattern
        content = self.get_page_content()
        imgInfoList = self.__interpret_content(webPattern, content)
        imgInfoList = imgInfoList[:number] if number < len(imgInfoList) else imgInfoList
        pool = ThreadPool(4)
        pool.map(self.__download_a_image, imgInfoList)
        pool.close()
        pool.join()


# name_url = 'https://ticket.urbtix.hk/internet/login/memberLogin'
# #     #'https://zc.reg.163.com/cp?channel=2&id=06E55BEA79780998EC66F9128786C4212F65A82B52469155A18FF01FBC8BFA000970E3B6C5698F3FC6E02C19D3DEB825B0A8BC52123B6D6144C3F5CADC321DE4CD9C1F88078EA8F6B7EFF6DC14F80A6B&nocache=1488594174019'
# #driver = webdriver.Chrome()#Firefox()
# import time
# from selenium import webdriver
#
# #driver = webdriver.Chrome('/path/to/chromedriver')  # Optional argument, if not specified will search path.
#
# driver = webdriver.Chrome('/Users/hezishan/Downloads/chromedriver')
# driver.get(name_url)
# print driver.page_source

"""
spider = CaptchaSpider(captchaURL='https://zc.reg.163.com/cp?channel=2&id=06E55BEA79780998EC66F9128786C4212F65A82B52469155A18FF01FBC8BFA0062F71E77C77BBB4EFFAD403A74189BF29136C041EA71FBFF814CE64DAD430C37E4C157E7A5E2FBFE938D0FCA9EACB2F3&nocache=1489315430710')
print spider.get_page_content(), len(spider.get_page_content())
#spider.download_images(number=1000)
finish_time = time.time()
#"""
"""
#https://www.zhihu.com/captcha.gif
spider = WebPageSpider('http://www2.cs.sfu.ca/~mori/research/gimpy/ez/')
spider.download_captcha_images(number=180)
#"""

## QQ: https://ssl.captcha.qq.com/getimage?uin=309392121@qq.com&aid=522005705&cap_cd=SvRGhBB3gorxquzCqv_zjMWG9QewrtdK2v3OFiP21YuyCtdropACGQ**&0.22700696171043822

# Gdgs:  http://www.gdgs.gov.cn/
#               (http://www.gdgs.gov.cn/sofpro/loginimg.ucap)

# hongxiu http://login.sns.hongxiu.com/reg.aspx
#       (http://login.sns.hongxiu.com/reg/CheckCode.aspx?r=631)

# xiaoxiang http://www.xxsy.net/user/Reg.aspx
#       (http://www.xxsy.net/showCode.aspx?rnd=0.5803604107709841)

# hkgolden: https://www.hkgolden.com/members/join2015.aspx?type=0
#       (https://www.hkgolden.com/members/CheckImageCode.aspx)

# PCOnline: http://my.pconline.com.cn/passport/mobileRegister.jsp
#               (http://captcha.pconline.com.cn/captcha/v.jpg)

# HKU portal : https://extranet.hku.hk/itpwdpol/servlet/identifyUser
#               (https://extranet.hku.hk/itpwdpol/servlet/Kaptcha3)
# HKBU issue: https://iss.hkbu.edu.hk/buam/signForm.seam
#              (https://iss.hkbu.edu.hk/buam/KaptchaFour.jpg)

# CSDN: https://passport.csdn.net/account/mobileregister?action=mobileRegisterView&service=https%3A%2F%2Fwww.google.com.hk%2F
#           (https://passport.csdn.net/ajax/verifyhandler.ashx)

# 17173:  http://passport.17173.com/register
#           (http://passport.17173.com/register/captcha?v=58ba35e8b6e8b)

# jiayuan: http://reg.jiayuan.com/signup/fillbasic.php?bd=5411&sex=m&year=1992&month=18&day=1802&province=44&degree=30&marriage=1&height=170&degree=30
#           (http://reg.jiayuan.com/antispam_v5.php?v=1488598141)
#
#
# qiannvyouhun http://xqn.163.com/reg/
#           (https://zc.reg.163.com/cp?channel=2&id=06E55BEA79780998EC66F9128786C4212F65A82B52469155A18FF01FBC8BFA000970E3B6C5698F3FC6E02C19D3DEB825B0A8BC52123B6D6144C3F5CADC321DE4CD9C1F88078EA8F6B7EFF6DC14F80A6B&nocache=1488594174019)
#
# baidu     https://passport.baidu.com/?getpassindex&tt=1489247099547&gid=93F9875-4E31-44AA-A1FA-19DC629F80D6&tpl=pp&u=https%3A%2F%2Fpassport.baidu.com%2F
#           (https://passport.baidu.com/cgi-bin/genimage?njG0206e28ea88ce28302d514f5de016214ccf5de063e04137c)

# 163 email:   http://reg.email.163.com/unireg/call.do?cmd=register.entrance&from=163mail_right
#           (http://reg.email.163.com/unireg/call.do?cmd=register.verifyCode&v=common/verifycode/vc_en&vt=mobile_acode&t=1488596379849)

# 360: http://openapi.360.cn/page/reg?destUrl=https%3A%2F%2Fopenapi.360.cn%2Foauth2%2Fauthorize%3Fclient_id%3Dc8528cec7650180c3dbc5ee67b9c265d%26response_type%3Dcode%26redirect_uri%3Dhttps%253A%252F%252Fpassport.sohu.com%252Fopenlogin%252Fcallback%252Fqihoo360&m=e8f27d
#           (http://passport.360.cn/captcha.php?m=create&app=i360&scene=reg&userip=&level=default&sign=f67462&r=1488596475&_=1488596475518)

# TPO: http://passport.zhan.com/Users/register.html
#           (http://passport.zhan.com/Users/changImg?time=0.9684228332821088&id=phone_register)

# SINA blog:  https://login.sina.com.cn/signup/signup
# (https://login.sina.com.cn/cgi/pin.php?r=1488596902468&lang=zh&type=hollow)
# https://passport.yuewen.com/reg.html?appid=36&areaid=1&target=iframe&ticket=1&auto=1&autotime=30&returnUrl=https%3A%2F%2Fwww.readnovel.com%2FloginSuccess
# https://nisp-captcha.nosdn.127.net/1489212887760_751682016
# https://ebanks.cgbchina.com.cn/perbank/
# https://perbank.abchina.com/EbankSite/startup.do

# https://zc.reg.163.com/cp?channel=2&id=06E55BEA79780998EC66F9128786C4212F65A82B52469155A18FF01FBC8BFA0062F71E77C77BBB4EFFAD403A74189BF29136C041EA71FBFF814CE64DAD430C37E4C157E7A5E2FBFE938D0FCA9EACB2F3&nocache=1489315430711

# Websites:  http://123.lvse.com/testcaptchas