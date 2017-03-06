# import captcha.audio
from third_party.captcha.image import ImageCaptcha
import random
import os


class CaptchaGenerator:

    def __init__(self, path, number, length):
        if not os.path.exists(path):
            os.mkdir(path)
        self.path = path
        self.number = number
        self.length = length
        self.fonts = ['/Library/Fonts/Arial.ttf']
        #self.fonts = ['/Library/Fonts/Arial.ttf', '/Library/Fonts/Brush Script.ttf', '/Library/Fonts/Phosphate.ttc']
        self.dictionary = []
        for index in range(48,58):
            self.dictionary.append(chr(index))
        for index in range(65,91):
            self.dictionary.append(chr(index))
        for index in range(97, 123):
            self.dictionary.append(chr(index))


    def captchaGeneration(self):
        generator = ImageCaptcha(fonts=self.fonts)
        for num in range(0, self.number):
            label = ''
            for length in range(0, self.length):
                label += self.dictionary[random.choice(range(len(self.dictionary)))]
            #print label
            generator.generate(label)
            generator.write(label,os.path.join(self.path, label+'.png'))




# captchaGenerator = CaptchaGenerator('data',100,5)
# captchaGenerator.captchaGeneration()

"""
image = ImageCaptcha(fonts=['/Library/Fonts/Arial.ttf'])

for num in range(0, 5):
    data = image.generate('Sally')
    image.write('Sally', 'out' + str(num) +'.png')
"""



