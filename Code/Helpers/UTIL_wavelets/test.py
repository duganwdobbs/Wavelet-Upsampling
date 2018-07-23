# Purpose:
#     performs dwt and idwt in pywavelets and tensorflow for comparison

import tensorflow as tf
import wavelets
import numpy as np
from scipy import misc
import pywt

pywt_wavelet = "db20"
wavelet = eval("wavelets." + pywt_wavelet)


# ██       ██████   █████  ██████      ██████   █████  ████████  █████
# ██      ██    ██ ██   ██ ██   ██     ██   ██ ██   ██    ██    ██   ██
# ██      ██    ██ ███████ ██   ██     ██   ██ ███████    ██    ███████
# ██      ██    ██ ██   ██ ██   ██     ██   ██ ██   ██    ██    ██   ██
# ███████  ██████  ██   ██ ██████      ██████  ██   ██    ██    ██   ██


# the images are 2 scales of the same face
image_size = 200
inputs = misc.imread('s0_r0_t0male.png', flatten = True)
inputs = misc.imresize(inputs, (image_size, image_size))
inputs = np.reshape(inputs/255.0, [1, image_size, image_size, 1])


# ██████  ██    ██ ██     ██  █████  ██    ██ ███████ ██      ███████ ████████
# ██   ██  ██  ██  ██     ██ ██   ██ ██    ██ ██      ██      ██         ██
# ██████    ████   ██  █  ██ ███████ ██    ██ █████   ██      █████      ██
# ██         ██    ██ ███ ██ ██   ██  ██  ██  ██      ██      ██         ██
# ██         ██     ███ ███  ██   ██   ████   ███████ ███████ ███████    ██

inputs = inputs[0, :, :, 0]

for mode in pywt.Modes.modes:
    wp = pywt.WaveletPacket2D(data=inputs, wavelet=pywt_wavelet, mode=mode)
    tf.gfile.MakeDirs("pywt/" + mode)
    misc.imsave("pywt/" + mode + "/pywt_a_result.png", wp['a'].data)
    misc.imsave("pywt/" + mode + "/pywt_h_result.png", wp['h'].data)
    misc.imsave("pywt/" + mode + "/pywt_v_result.png", wp['v'].data)
    misc.imsave("pywt/" + mode + "/pywt_d_result.png", wp['d'].data)


# ███    ██ ███████ ████████
# ████   ██ ██         ██
# ██ ██  ██ █████      ██
# ██  ██ ██ ██         ██
# ██   ████ ███████    ██

inputs = np.reshape(inputs, [1, 200, 200, 1])
x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
net = x

for mode in wavelets.modes:
    # ██████  ██     ██ ████████
    # ██   ██ ██     ██    ██
    # ██   ██ ██  █  ██    ██
    # ██   ██ ██ ███ ██    ██
    # ██████   ███ ███     ██
    tf.gfile.MakeDirs("tfwt/" + mode)
    dwt = wavelets.dwt(net, wavelet, padding_mode = mode)
    dwt_result = tf.Session().run(dwt, feed_dict = {x: inputs})
    # SAVE
    misc.imsave("tfwt/" + mode + "/dwt_a_result.png", dwt_result[0, 0, 0, :, :, 0])
    misc.imsave("tfwt/" + mode + "/dwt_h_result.png", dwt_result[0, 1, 0, :, :, 0])
    misc.imsave("tfwt/" + mode + "/dwt_v_result.png", dwt_result[1, 0, 0, :, :, 0])
    misc.imsave("tfwt/" + mode + "/dwt_d_result.png", dwt_result[1, 1, 0, :, :, 0])

    # ██ ██████  ██     ██ ████████
    # ██ ██   ██ ██     ██    ██
    # ██ ██   ██ ██  █  ██    ██
    # ██ ██   ██ ██ ███ ██    ██
    # ██ ██████   ███ ███     ██

    idwt = wavelets.idwt(dwt, wavelet, padding_mode = mode)
    w_x, idwt_result = tf.Session().run(idwt, feed_dict = {x: inputs})
    # SAVE
    misc.imsave("tfwt/" + mode + "/w_x_a_result.png", w_x[:,:,0])
    misc.imsave("tfwt/" + mode + "/w_x_h_result.png", w_x[:,:,1])
    misc.imsave("tfwt/" + mode + "/idwt_result.png", idwt_result[0,:,:,0])
