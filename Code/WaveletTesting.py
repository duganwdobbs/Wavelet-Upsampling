# Wavelet TESTING

from DataLoaders.GoogleWebGetter import DataGenerator
import pywt, numpy as np, cv2

internal_generator = DataGenerator('train','E:/Wavelet-Upsampling/')
imgs, batch_list = internal_generator.get_next_batch(20)
img = imgs[0]
wavelet_types = ['haar','db1','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20']

for type in wavelet_types:
  dwt = [[pywt.dwt(channel,type) for channel in img] for img in imgs]
  # [Imgs [ Channels (Tuple of results)]]
  results = []
  for imgs in dwt:
    img = np.zeros((4,600,912,3))
    c = 0
    for channels in imgs:
      d = 0
      for decomp in channels:
        img[d,:,:,c] = decomp
        d += 1
      c += 1
    for x in range(4):
      results.append(img[x])

  internal_generator.test(results)


  maxval = np.amax(results)
  print("%s: %.7f"%(type,maxval))
