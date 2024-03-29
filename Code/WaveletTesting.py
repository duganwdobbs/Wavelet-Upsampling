# Wavelet TESTING

from DataLoaders.GoogleWebGetter import DataGenerator
import pywt, numpy as np, cv2

# internal_generator = DataGenerator('train','E:/Wavelet-Upsampling/')
# imgs, batch_list = internal_generator.get_next_batch(300)
# img = imgs[0]
wavelet_types = ['haar','db1','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20']
wavelet_types = ['db2']
#
# def normalize(img):
#   # img = img + min(0,np.amin(img))
#   img = img / np.amax(img)
#   return img

# maxval = 0
# for img in imgs:
#   for type in wavelet_types:
#     _avg = []
#     _low_w = []
#     _low_h = []
#     _detail = []
#     for x in range(3):
#       avg,(low_w,low_h,detail) = pywt.dwt2(img[:,:,x],type)
#       _avg.append(avg)
#       _low_w.append(low_w)
#       _low_h.append(low_h)
#       _detail.append(detail)
#
#
#     maxval = max(maxval,np.amax(_avg) - np.amin(_avg))
#     print("%s: %.2f, this ext: (%.2f/%.2f), from: (%d/%d)"%(type,maxval,np.amax(_avg),np.amin(_avg),np.amax(img),np.amin(img)))
#
#     _avg = normalize(np.stack(_avg,-1))
#     _low_w = normalize(np.stack(_low_w,-1))
#     _low_h = normalize(np.stack(_low_h,-1))
#     _detail = normalize(np.stack(_detail,-1))
#
#     results = [_avg,_low_w,_low_h,_detail]

    # internal_generator.test(results)

wavelet = pywt.Wavelet('db2')
# print(wavelet)
wavelet = wavelet.filter_bank
print(wavelet[0])
print()
print(np.sum(wavelet[0]),np.sum(np.abs(wavelet[0])))
print(np.sum(wavelet[0])**2,np.sum(np.abs(wavelet[0]))**2)
print()
print(np.sum(wavelet[1]),np.sum(np.abs(wavelet[1])))
print(np.sum(wavelet[1])**2,np.sum(np.abs(wavelet[1]))**2)
print()
print(np.sum(wavelet[2]),np.sum(np.abs(wavelet[2])))
print(np.sum(wavelet[2])**2,np.sum(np.abs(wavelet[2]))**2)
print()
print(np.sum(wavelet[3]),np.sum(np.abs(wavelet[3])))
print(np.sum(wavelet[3])**2,np.sum(np.abs(wavelet[3]))**2)
print()
