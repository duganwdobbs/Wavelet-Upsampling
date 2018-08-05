# build list of images from Oir


# For image in Resized, resize using all interpolation tech
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

def str_remove(things_to_replace,str):
  for thing in things_to_replace:
    str.replace(thing,'')

def calc_error(im1,im2):
  im1 = np.asarray(im1)
  im2 = np.asarray(im2)

  err = np.abs(im1-im2)
  avg = np.average(err)

  return avg

THINGS_TO_REPLACE = ('_waifu2x_photo_noise1_scale_tta_1','-magic','.png')

base_dir     = 'E:/Wavelet-Upsampling/UpsampleTesting/'
origin_dir   = base_dir + 'Ori/'
resize_dir   = base_dir + 'Resized/'
waifu2x_dir  = base_dir + 'Waifu2x/'
letsenh_dir  = base_dir + 'LetsEnhance/'
wavsamp_dir  = base_dir + 'WavResult/'
nearest_dir  = base_dir + 'Nearest/'
bilinear_dir = base_dir + 'Bilinear/'
bicubic_dir  = base_dir + 'Bicubic/'
lanczos_dir  = base_dir + 'Lanczos/'

ori_list = [f.replace('.png','') for f in os.listdir(origin_dir )]
res_list = [f.replace('.png','') for f in os.listdir(resize_dir )]
wai_list = [f.replace('.png','') for f in os.listdir(waifu2x_dir)]
wai_list = [f.replace('_waifu2x_photo_noise1_scale_tta_1','') for f in wai_list]
let_list = [f.replace('.png','') for f in os.listdir(letsenh_dir)]
let_list = [f.replace('-magic','') for f in let_list]
wav_list = [f.replace('.png','') for f in os.listdir(wavsamp_dir)]

# Reduce everything to just the files we have.
compare_list = [f for f in res_list if f in let_list and f in wai_list]
print(compare_list)


for id in compare_list:
  sm_img = Image.open(resize_dir + id + '.png')
  img_copy = np.asarray(sm_img)
  imgH,imgW,imgC = img_copy.shape
  upsize = (imgW *2, imgH * 2)
  img_bilinear = sm_img.resize(upsize,Image.BILINEAR)
  img_bilinear.save(bilinear_dir + id + '.png')
  img_bicubic  = sm_img.resize(upsize,Image.BICUBIC )
  img_bicubic.save(bicubic_dir + id + '.png')
  img_lanczos  = sm_img.resize(upsize,Image.LANCZOS )
  img_lanczos.save(lanczos_dir + id + '.png')
  img_nearest  = sm_img.resize(upsize,Image.NEAREST )
  img_nearest.save(nearest_dir + id + '.png')
  img_origin   = Image.open(origin_dir  + id.replace('2_Resized','1_Origional') + '.png'  )
  img_wavsamp  = Image.open(wavsamp_dir + id.replace('2_Resized','1_WavResult') + '.png'  )
  img_waifu2x  = Image.open(waifu2x_dir + id + '_waifu2x_photo_noise1_scale_tta_1.png')
  img_letsenh  = Image.open(letsenh_dir + id + '-magic.png'                           )
  img_letsenh  = img_letsenh.resize(upsize,Image.NEAREST )

  errors = {}
  errors['bilinear_err'] = calc_error(img_origin,img_bilinear)
  errors['bicubic_err']  = calc_error(img_origin,img_bicubic )
  errors['lanczos_err']  = calc_error(img_origin,img_lanczos )
  errors['nearest_err']  = calc_error(img_origin,img_nearest )
  errors['wavsamp_err']  = calc_error(img_origin,img_wavsamp )
  errors['waifu2x_err']  = calc_error(img_origin,img_waifu2x )
  errors['letsenh_err']  = calc_error(img_origin,img_letsenh )

  keys = list(errors.keys())
  for key in keys:
    print("%s: %.3f"%(key,errors[key]),end= ' ')
  print()
