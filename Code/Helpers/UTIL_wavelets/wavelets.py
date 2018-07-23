import tensorflow as tf

# purpose of this version is to introduce other wavelets and generalize the kernels
# high/low and decomp/reconstruct in terms of the wavelet
# resource for wavelet weights http://wavelets.pybytes.com/wavelet/db4/
# Decomposition low-pass filter = reverse(wavelet) * [1, -1]
# decomposition high-pass filter = wavelet
# Reconstruction low-pass filter = wavelet * [-1, 1]
# Reconstruction high-pass filter = reverse(wavelet)
# This implementation has the reverse relationship with the wavelet
# in order to match pywavelets (I think it uses a different coordinate system)


# TODO:
    # do not switch to storing the reverse of the wavelet
    # support accepting tensorflow variable as wavelet
    # support swt, (stride = 1)
    # support more wavelets
    # support more modes

modes = ["periodization", "zero", "reflect", "symmetric"]

def make_decomposition_filter(wavelet):
    wavelet_length = len(wavelet)
    decomposition_high_pass_filter = tf.reverse(tf.constant(wavelet, dtype = tf.float32), [0])
    decomposition_low_pass_filter = tf.reshape(tf.multiply(tf.reshape(wavelet, [-1, 2]),
                                                           [-1, 1]),
                                               [-1])
    filter = tf.concat([tf.reshape(decomposition_low_pass_filter, [wavelet_length, 1, 1]),
                        tf.reshape(decomposition_high_pass_filter, [wavelet_length, 1, 1])],
                        axis = 2)

    return filter

# assumed [batch, h, w, c]
# padding_mode: One of "periodization", "zero", "reflect", or "symmetric"
# resource for modes http://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html
def dwt(inputs, wavelet, padding_mode = "periodization"):
    wavelet_length = len(wavelet)
    wavelet_pad = wavelet_length-2
    channels    = tf.shape(inputs)[3]
    height      = tf.shape(inputs)[1]
    width       = tf.shape(inputs)[2]
    filter      = make_decomposition_filter(wavelet)

    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    # now [batch, channel, height, width]

    inputs = tf.reshape(inputs, [-1, width, 1])
    # now [batch channel height, width, 1]

    if padding_mode is "periodization": # manually wrap
        p_wavelet_pad = int(wavelet_pad/2)
        left_pad_vals = inputs[:, -p_wavelet_pad:, :]
        right_pad_vals = inputs[:, :p_wavelet_pad, :]
        inputs = tf.concat([left_pad_vals, inputs, right_pad_vals], axis = 1)
        w_x = tf.nn.conv1d(inputs, filters = filter, stride = 2, padding = 'VALID')
        # [b c h, w, pass]
        w_x = tf.reshape(w_x, [-1, height, tf.cast(width/2, tf.int32), 2])
        # [b c, h, w, pass]
        w_x = tf.transpose(w_x, [0, 2, 3, 1])
        # [b c, w, pass, height]
        w_x = tf.reshape(w_x, [-1, height, 1])
        # [b c w pass, height, 1]
        left_pad_vals = w_x[:, -p_wavelet_pad:, :]
        right_pad_vals = w_x[:, :p_wavelet_pad, :]
        w_x = tf.concat([left_pad_vals, w_x, right_pad_vals], axis = 1)
        w_x_y = tf.nn.conv1d(w_x, filters = filter, stride = 2, padding = 'VALID')
        # [b c w w_pass, height, h_pass]
        w_x_y = tf.reshape(w_x_y, [-1, channels, tf.cast(width/2, tf.int32), 2, tf.cast(height/2, tf.int32), 2])
        # [b, c, w, w_pass, height, h_pass]
    elif padding_mode is "zero": # constant
        inputs = tf.pad(inputs, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "CONSTANT")
        w_x = tf.nn.conv1d(inputs, filters = filter, stride = 2, padding = 'VALID')
        # [b c h, w, pass]
        w_x = tf.reshape(w_x, [-1, height, tf.cast((width+wavelet_pad)/2, tf.int32), 2])
        # [b c, h, w, pass]
        w_x = tf.transpose(w_x, [0, 2, 3, 1])
        # [b c, w, pass, height]
        w_x = tf.reshape(w_x, [-1, height, 1])
        # [b c w pass, height, 1]
        w_x = tf.pad(w_x, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "CONSTANT")
        w_x_y = tf.nn.conv1d(w_x, filters = filter, stride = 2, padding = 'VALID')
        # [b c w w_pass, height, h_pass]
        w_x_y = tf.reshape(w_x_y, [-1, channels, tf.cast((width+wavelet_pad)/2, tf.int32), 2, tf.cast((height+wavelet_pad)/2, tf.int32), 2])
        # [b, c, w, w_pass, height, h_pass]
    elif padding_mode is "reflect": # reflect
        inputs = tf.pad(inputs, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "REFLECT")
        w_x = tf.nn.conv1d(inputs, filters = filter, stride = 2, padding = 'VALID')
        # [b c h, w, pass]
        w_x = tf.reshape(w_x, [-1, height, tf.cast((width+wavelet_pad)/2, tf.int32), 2])
        # [b c, h, w, pass]
        w_x = tf.transpose(w_x, [0, 2, 3, 1])
        # [b c, w, pass, height]
        w_x = tf.reshape(w_x, [-1, height, 1])
        # [b c w pass, height, 1]
        w_x = tf.pad(w_x, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "REFLECT")
        w_x_y = tf.nn.conv1d(w_x, filters = filter, stride = 2, padding = 'VALID')
        # [b c w w_pass, height, h_pass]
        w_x_y = tf.reshape(w_x_y, [-1, channels, tf.cast((width+wavelet_pad)/2, tf.int32), 2, tf.cast((height+wavelet_pad)/2, tf.int32), 2])
        # [b, c, w, w_pass, height, h_pass]
    elif padding_mode is "symmetric": # symmetric
        inputs = tf.pad(inputs, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "SYMMETRIC")
        w_x = tf.nn.conv1d(inputs, filters = filter, stride = 2, padding = 'VALID')
        # [b c h, w, pass]
        w_x = tf.reshape(w_x, [-1, height, tf.cast((width+wavelet_pad)/2, tf.int32), 2])
        # [b c, h, w, pass]
        w_x = tf.transpose(w_x, [0, 2, 3, 1])
        # [b c, w, pass, height]
        w_x = tf.reshape(w_x, [-1, height, 1])
        # [b c w pass, height, 1]
        w_x = tf.pad(w_x, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "SYMMETRIC")
        w_x_y = tf.nn.conv1d(w_x, filters = filter, stride = 2, padding = 'VALID')
        # [b c w w_pass, height, h_pass]
        w_x_y = tf.reshape(w_x_y, [-1, channels, tf.cast((width+wavelet_pad)/2, tf.int32), 2, tf.cast((height+wavelet_pad)/2, tf.int32), 2])
        # [b, c, w, w_pass, height, h_pass]

    w_x_y = tf.transpose(w_x_y, [3, 5, 0, 4, 2, 1])
    # [w_pass, h_pass, b, height, width, channel]

    return w_x_y

# inputs assumed [w_pass, h_pass, b, height, width, channel]
def idwt(inputs, wavelet, padding_mode = "periodization"):

    wavelet_length = len(wavelet)
    wavelet_pad = wavelet_length-2
    examples = tf.shape(inputs)[2]
    channels = tf.shape(inputs)[5]
    width = tf.shape(inputs)[4]
    height = tf.shape(inputs)[3]
    dest_width = (tf.shape(inputs)[4] * 2) - wavelet_pad
    dest_height = (tf.shape(inputs)[3] * 2) - wavelet_pad

    inputs = tf.transpose(inputs, [2, 5, 4, 0, 3, 1])
    # [b, channel, width, w_pass, height, h_pass]
    inputs = tf.reshape(inputs, [-1, tf.cast(height, tf.int32), 2])
    # [b channel width w_pass, height, h_pass]

    filter = make_decomposition_filter(wavelet)

    if padding_mode is "periodization":
        dest_width = width*2
        dest_height = height*2
        p_wavelet_pad = int(wavelet_pad/2)

        left_pad_vals = inputs[:, -p_wavelet_pad:, :]
        right_pad_vals = inputs[:, :p_wavelet_pad, :]
        inputs = tf.concat([left_pad_vals, inputs, right_pad_vals], axis = 1)

        w_x = tf.contrib.nn.conv1d_transpose(inputs,
                                             filter = filter,
                                             output_shape = [examples*channels*width*2, dest_height+(3*wavelet_pad), 1],
                                             stride = 2,
                                             padding = "VALID")
        # [b channel width w_pass, height+2pad, 1]
        if wavelet_pad > 0:
            w_x = w_x[:, 3*p_wavelet_pad:-3*p_wavelet_pad, 0]
        w_x = tf.reshape(w_x, [-1, width, 2, dest_height])
        # [b channel, width, w_pass, height]
        w_x = tf.transpose(w_x, [0, 3, 1, 2])
        # [b channel, height, width, width_pass]
        w_x = tf.reshape(w_x, [-1, tf.cast(width, tf.int32), 2])
        # [b channel height, width, width_pass]

        left_pad_vals = w_x[:, -p_wavelet_pad:, :]
        right_pad_vals = w_x[:, :p_wavelet_pad, :]
        w_x = tf.concat([left_pad_vals, w_x, right_pad_vals], axis = 1)

        spatial = tf.contrib.nn.conv1d_transpose(w_x,
                                                 filter = filter,
                                                 output_shape = [examples*channels*dest_height, dest_width+(3*wavelet_pad), 1],
                                                 stride = 2,
                                                 padding = "VALID")
        # [b channels height, width, 1]
        if wavelet_pad > 0:
            spatial = spatial[:, 3*p_wavelet_pad:-3*p_wavelet_pad, 0]
        spatial = tf.reshape(spatial, [-1, channels, dest_height, dest_width])
        # [b, channels, height, width]
    elif padding_mode is "zero":
        w_x = tf.contrib.nn.conv1d_transpose(inputs,
                                             filter = filter,
                                             output_shape = [examples*channels*width*2, dest_height+(2*wavelet_pad), 1],
                                             stride = 2,
                                             padding = "VALID")
        # [b channel width w_pass, height+2pad, 1]
        if wavelet_pad > 0:
            w_x = w_x[:, wavelet_pad:-wavelet_pad, 0]
        w_x = tf.reshape(w_x, [-1, tf.cast(width, tf.int32), 2, dest_height])
        # [b channel, width, w_pass, height]
        w_x = tf.transpose(w_x, [0, 3, 1, 2])
        # [b channel, height, width, width_pass]
        w_x = tf.reshape(w_x, [-1, tf.cast(width, tf.int32), 2])
        # [b channel height, width, width_pass]

        spatial = tf.contrib.nn.conv1d_transpose(w_x,
                                                 filter = filter,
                                                 output_shape = [examples*channels*dest_height, dest_width+(2*wavelet_pad), 1],
                                                 stride = 2,
                                                 padding = "VALID")
        # [b channels height, width, 1]
        if wavelet_pad > 0:
            spatial = spatial[:, wavelet_pad:-wavelet_pad, 0]
        spatial = tf.reshape(spatial, [-1, channels, dest_height, dest_width])
        # [b, channels, height, width]
    elif padding_mode is "reflect":
        w_x = tf.contrib.nn.conv1d_transpose(inputs,
                                             filter = filter,
                                             output_shape = [examples*channels*width*2, dest_height+(2*wavelet_pad), 1],
                                             stride = 2,
                                             padding = "VALID")
        # [b channel width w_pass, height+2pad, 1]
        if wavelet_pad > 0:
            w_x = w_x[:, wavelet_pad:-wavelet_pad, 0]
        w_x = tf.reshape(w_x, [-1, tf.cast(width, tf.int32), 2, dest_height])
        # [b channel, width, w_pass, height]
        w_x = tf.transpose(w_x, [0, 3, 1, 2])
        # [b channel, height, width, width_pass]
        w_x = tf.reshape(w_x, [-1, tf.cast(width, tf.int32), 2])
        # [b channel height, width, width_pass]

        spatial = tf.contrib.nn.conv1d_transpose(w_x,
                                                 filter = filter,
                                                 output_shape = [examples*channels*dest_height, dest_width+(2*wavelet_pad), 1],
                                                 stride = 2,
                                                 padding = "VALID")
        # [b channels height, width, 1]
        if wavelet_pad > 0:
            spatial = spatial[:, wavelet_pad:-wavelet_pad, 0]
        spatial = tf.reshape(spatial, [-1, channels, dest_height, dest_width])
        # [b, channels, height, width]
    elif padding_mode is "symmetric":
        w_x = tf.contrib.nn.conv1d_transpose(inputs,
                                             filter = filter,
                                             output_shape = [examples*channels*width*2, dest_height+(2*wavelet_pad), 1],
                                             stride = 2,
                                             padding = "VALID")
        # [b channel width w_pass, height+2pad, 1]
        if wavelet_pad > 0:
            w_x = w_x[:, wavelet_pad:-wavelet_pad, 0]
        w_x = tf.reshape(w_x, [-1, tf.cast(width, tf.int32), 2, dest_height])
        # [b channel, width, w_pass, height]
        w_x = tf.transpose(w_x, [0, 3, 1, 2])
        # [b channel, height, width, width_pass]
        w_x = tf.reshape(w_x, [-1, tf.cast(width, tf.int32), 2])
        # [b channel height, width, width_pass]

        spatial = tf.contrib.nn.conv1d_transpose(w_x,
                                                 filter = filter,
                                                 output_shape = [examples*channels*dest_height, dest_width+(2*wavelet_pad), 1],
                                                 stride = 2,
                                                 padding = "VALID")
        # [b channels height, width, 1]
        if wavelet_pad > 0:
            spatial = spatial[:, wavelet_pad:-wavelet_pad, 0]
        spatial = tf.reshape(spatial, [-1, channels, dest_height, dest_width])
        # [b, channels, height, width]

    spatial = tf.transpose(spatial, [0, 2, 3, 1])
    # [b, height, width, channels]

    return w_x, spatial


#stationary wavelet transform
# assumed [batch, h, w, c]
# padding_mode: One of "periodization", "zero", "reflect", or "symmetric"
# resource  http://pywavelets.readthedocs.io/en/latest/ref/swt-stationary-wavelet-transform.html
def swt(inputs, wavelet, level, padding_mode = "periodization"):
    wavelet_length = len(wavelet)
    wavelet_pad = wavelet_length-2
    channels = tf.shape(inputs)[3]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    filter = make_decomposition_filter(wavelet)

    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    # now [batch, channel, height, width]

    inputs = tf.reshape(inputs, [-1, width, 1])
    # now [batch channel height, width, 1]

    if padding_mode is "periodization": # manually wrap
        p_wavelet_pad = int(wavelet_pad/2)
        left_pad_vals = inputs[:, -p_wavelet_pad:, :]
        right_pad_vals = inputs[:, :p_wavelet_pad, :]
        inputs = tf.concat([left_pad_vals, inputs, right_pad_vals], axis = 1)
        w_x = tf.nn.conv1d(inputs, filters = filter, stride = 2, padding = 'VALID')
        # [b c h, w, pass]
        w_x = tf.reshape(w_x, [-1, height, tf.cast(width/2, tf.int32), 2])
        # [b c, h, w, pass]
        w_x = tf.transpose(w_x, [0, 2, 3, 1])
        # [b c, w, pass, height]
        w_x = tf.reshape(w_x, [-1, height, 1])
        # [b c w pass, height, 1]
        left_pad_vals = w_x[:, -p_wavelet_pad:, :]
        right_pad_vals = w_x[:, :p_wavelet_pad, :]
        w_x = tf.concat([left_pad_vals, w_x, right_pad_vals], axis = 1)
        w_x_y = tf.nn.conv1d(w_x, filters = filter, stride = 2, padding = 'VALID')
        # [b c w w_pass, height, h_pass]
        w_x_y = tf.reshape(w_x_y, [-1, channels, tf.cast(width/2, tf.int32), 2, tf.cast(height/2, tf.int32), 2])
        # [b, c, w, w_pass, height, h_pass]
    elif padding_mode is "zero": # constant
        inputs = tf.pad(inputs, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "CONSTANT")
        w_x = tf.nn.conv1d(inputs, filters = filter, stride = 2, padding = 'VALID')
        # [b c h, w, pass]
        w_x = tf.reshape(w_x, [-1, height, tf.cast((width+wavelet_pad)/2, tf.int32), 2])
        # [b c, h, w, pass]
        w_x = tf.transpose(w_x, [0, 2, 3, 1])
        # [b c, w, pass, height]
        w_x = tf.reshape(w_x, [-1, height, 1])
        # [b c w pass, height, 1]
        w_x = tf.pad(w_x, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "CONSTANT")
        w_x_y = tf.nn.conv1d(w_x, filters = filter, stride = 2, padding = 'VALID')
        # [b c w w_pass, height, h_pass]
        w_x_y = tf.reshape(w_x_y, [-1, channels, tf.cast((width+wavelet_pad)/2, tf.int32), 2, tf.cast((height+wavelet_pad)/2, tf.int32), 2])
        # [b, c, w, w_pass, height, h_pass]
    elif padding_mode is "reflect": # reflect
        inputs = tf.pad(inputs, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "REFLECT")
        w_x = tf.nn.conv1d(inputs, filters = filter, stride = 2, padding = 'VALID')
        # [b c h, w, pass]
        w_x = tf.reshape(w_x, [-1, height, tf.cast((width+wavelet_pad)/2, tf.int32), 2])
        # [b c, h, w, pass]
        w_x = tf.transpose(w_x, [0, 2, 3, 1])
        # [b c, w, pass, height]
        w_x = tf.reshape(w_x, [-1, height, 1])
        # [b c w pass, height, 1]
        w_x = tf.pad(w_x, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "REFLECT")
        w_x_y = tf.nn.conv1d(w_x, filters = filter, stride = 2, padding = 'VALID')
        # [b c w w_pass, height, h_pass]
        w_x_y = tf.reshape(w_x_y, [-1, channels, tf.cast((width+wavelet_pad)/2, tf.int32), 2, tf.cast((height+wavelet_pad)/2, tf.int32), 2])
        # [b, c, w, w_pass, height, h_pass]
    elif padding_mode is "symmetric": # symmetric
        inputs = tf.pad(inputs, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "SYMMETRIC")
        w_x = tf.nn.conv1d(inputs, filters = filter, stride = 2, padding = 'VALID')
        # [b c h, w, pass]
        w_x = tf.reshape(w_x, [-1, height, tf.cast((width+wavelet_pad)/2, tf.int32), 2])
        # [b c, h, w, pass]
        w_x = tf.transpose(w_x, [0, 2, 3, 1])
        # [b c, w, pass, height]
        w_x = tf.reshape(w_x, [-1, height, 1])
        # [b c w pass, height, 1]
        w_x = tf.pad(w_x, [[0, 0], [wavelet_pad, wavelet_pad], [0, 0]], mode = "SYMMETRIC")
        w_x_y = tf.nn.conv1d(w_x, filters = filter, stride = 2, padding = 'VALID')
        # [b c w w_pass, height, h_pass]
        w_x_y = tf.reshape(w_x_y, [-1, channels, tf.cast((width+wavelet_pad)/2, tf.int32), 2, tf.cast((height+wavelet_pad)/2, tf.int32), 2])
        # [b, c, w, w_pass, height, h_pass]

    w_x_y = tf.transpose(w_x_y, [3, 5, 0, 4, 2, 1])
    # [w_pass, h_pass, b, height, width, channel]

    return w_x_y

# ██     ██  █████  ██    ██ ███████ ██      ███████ ████████ ███████
# ██     ██ ██   ██ ██    ██ ██      ██      ██         ██    ██
# ██  █  ██ ███████ ██    ██ █████   ██      █████      ██    ███████
# ██ ███ ██ ██   ██  ██  ██  ██      ██      ██         ██         ██
#  ███ ███  ██   ██   ████   ███████ ███████ ███████    ██    ███████




# ██   ██  █████   █████  ██████
# ██   ██ ██   ██ ██   ██ ██   ██
# ███████ ███████ ███████ ██████
# ██   ██ ██   ██ ██   ██ ██   ██
# ██   ██ ██   ██ ██   ██ ██   ██


haar = [-0.7071067811865476, 0.7071067811865476]


# ██████   █████  ██    ██ ██████  ███████  ██████ ██   ██ ██ ███████ ███████
# ██   ██ ██   ██ ██    ██ ██   ██ ██      ██      ██   ██ ██ ██      ██
# ██   ██ ███████ ██    ██ ██████  █████   ██      ███████ ██ █████   ███████
# ██   ██ ██   ██ ██    ██ ██   ██ ██      ██      ██   ██ ██ ██           ██
# ██████  ██   ██  ██████  ██████  ███████  ██████ ██   ██ ██ ███████ ███████


db1 = [-0.7071067811865476, 0.7071067811865476]
db2 = [-0.48296291314469025, 0.836516303737469, -0.22414386804185735, -0.12940952255092145]
db3 = [-0.3326705529509569,
0.8068915093133388,
-0.4598775021193313,
-0.13501102001039084,
0.08544127388224149,
0.035226291882100656]
db4 = [-0.23037781330885523,
0.7148465705525415,
-0.6308807679295904,
-0.02798376941698385,
0.18703481171888114,
0.030841381835986965,
-0.032883011666982945,
-0.010597401784997278]
db5 = [-0.160102397974125,
0.6038292697974729,
-0.7243085284385744,
0.13842814590110342,
0.24229488706619015,
-0.03224486958502952,
-0.07757149384006515,
-0.006241490213011705,
0.012580751999015526,
0.003335725285001549]
db6 = [-0.11154074335008017,
0.4946238903983854,
-0.7511339080215775,
0.3152503517092432,
0.22626469396516913,
-0.12976686756709563,
-0.09750160558707936,
0.02752286553001629,
0.031582039318031156,
0.0005538422009938016,
-0.004777257511010651,
-0.00107730108499558]
db7 = [-0.07785205408506236,
0.39653931948230575,
-0.7291320908465551,
0.4697822874053586,
0.14390600392910627,
-0.22403618499416572,
-0.07130921926705004,
0.0806126091510659,
0.03802993693503463,
-0.01657454163101562,
-0.012550998556013784,
0.00042957797300470274,
0.0018016407039998328,
0.0003537138000010399]
db8 = [-0.05441584224308161,
0.3128715909144659,
-0.6756307362980128,
0.5853546836548691,
0.015829105256023893,
-0.2840155429624281,
-0.00047248457399797254,
0.128747426620186,
0.01736930100202211,
-0.04408825393106472,
-0.013981027917015516,
0.008746094047015655,
0.00487035299301066,
-0.0003917403729959771,
-0.0006754494059985568,
-0.00011747678400228192]
db9 = [-0.03807794736316728,
0.24383467463766728,
-0.6048231236767786,
0.6572880780366389,
-0.13319738582208895,
-0.29327378327258685,
0.09684078322087904,
0.14854074933476008,
-0.030725681478322865,
-0.06763282905952399,
-0.00025094711499193845,
0.022361662123515244,
0.004723204757894831,
-0.004281503681904723,
-0.0018476468829611268,
0.00023038576399541288,
0.0002519631889981789,
3.9347319995026124e-05]
db10 = [-0.026670057900950818,
0.18817680007762133,
-0.5272011889309198,
0.6884590394525921,
-0.2811723436604265,
-0.24984642432648865,
0.19594627437659665,
0.12736934033574265,
-0.09305736460380659,
-0.07139414716586077,
0.02945753682194567,
0.03321267405893324,
-0.0036065535669883944,
-0.010733175482979604,
-0.0013953517469940798,
0.00199240529499085,
0.0006858566950046825,
-0.0001164668549943862,
-9.358867000108985e-05,
-1.326420300235487e-05]
db11 = [-0.01869429776147044,
0.1440670211506196,
-0.44989976435603013,
0.6856867749161785,
-0.41196436894789695,
-0.16227524502747828,
0.27423084681792875,
0.06604358819669089,
-0.14981201246638268,
-0.04647995511667613,
0.06643878569502022,
0.03133509021904531,
-0.02084090436018004,
-0.015364820906201324,
0.0033408588730145018,
0.004928417656058778,
0.00030859285881515924,
-0.0008930232506662366,
-0.00024915252355281426,
5.443907469936638e-05,
3.463498418698379e-05,
4.494274277236352e-06]
db12 = [-0.013112257957229239,
0.10956627282118277,
-0.3773551352142041,
0.6571987225792911,
-0.5158864784278007,
-0.04476388565377762,
0.31617845375277914,
-0.023779257256064865,
-0.18247860592758275,
0.0053595696743599965,
0.09643212009649671,
0.010849130255828966,
-0.04154627749508764,
-0.01221864906974642,
0.012840825198299882,
0.006711499008795549,
-0.0022486072409952287,
-0.0021795036186277044,
-6.5451282125215034e-06,
0.0003886530628209267,
8.850410920820318e-05,
-2.4241545757030318e-05,
-1.2776952219379579e-05,
-1.5290717580684923e-06]
db13 = [-0.009202133538962279,
0.08286124387290195,
-0.3119963221604349,
0.6110558511587811,
-0.5888895704312119,
0.086985726179645,
0.31497290771138414,
-0.12457673075080665,
-0.17947607942935084,
0.07294893365678874,
0.10580761818792761,
-0.026488406475345658,
-0.056139477100276156,
0.002379972254052227,
0.02383142071032781,
0.003923941448795577,
-0.007255589401617119,
-0.002761911234656831,
0.0013156739118922766,
0.000932326130867249,
-4.9251525126285676e-05,
-0.0001651289885565057,
-3.067853757932436e-05,
1.0441930571407941e-05,
4.700416479360808e-06,
5.2200350984548e-07]
db14 = [-0.0064611534600864905,
0.062364758849384874,
-0.25485026779256437,
0.5543056179407709,
-0.6311878491047198,
0.21867068775886594,
0.27168855227867705,
-0.2180335299932165,
-0.13839521386479153,
0.13998901658445695,
0.0867484115681106,
-0.0715489555039835,
-0.05523712625925082,
0.02698140830794797,
0.030185351540353976,
-0.0056150495303375755,
-0.01278949326634007,
-0.0007462189892638753,
0.003849638868019787,
0.001061691085606874,
-0.0007080211542354048,
-0.00038683194731287514,
4.177724577037067e-05,
6.875504252695734e-05,
1.0337209184568496e-05,
-4.389704901780418e-06,
-1.7249946753674012e-06,
-1.7871399683109222e-07]
db15 = [-0.004538537361577376,
0.04674339489275062,
-0.20602386398692688,
0.4926317717079753,
-0.6458131403572103,
0.33900253545462167,
0.19320413960907623,
-0.28888259656686216,
-0.06528295284876569,
0.19014671400708816,
0.0396661765557336,
-0.11112093603713753,
-0.033877143923563204,
0.054780550584559995,
0.02576700732836694,
-0.020810050169636805,
-0.015083918027862582,
0.005101000360422873,
0.0064877345603061454,
-0.00024175649075894543,
-0.0019433239803823459,
-0.0003734823541372647,
0.00035956524436229364,
0.00015589648992055726,
-2.579269915531323e-05,
-2.8133296266037558e-05,
-3.3629871817363823e-06,
1.8112704079399406e-06,
6.316882325879451e-07,
6.133359913303714e-08]
db16 = [-2.1093396300980412e-08,
-2.3087840868545578e-07,
-7.363656785441815e-07,
1.0435713423102517e-06,
1.133660866126152e-05,
1.394566898819319e-05,
-6.103596621404321e-05,
-0.00017478724522506327,
0.00011424152003843815,
0.0009410217493585433,
0.00040789698084934395,
-0.00312802338120381,
-0.0036442796214883506,
0.006990014563390751,
0.013993768859843242,
-0.010297659641009963,
-0.036888397691556774,
0.007588974368642594,
0.07592423604445779,
0.006239722752156254,
-0.13238830556335474,
-0.027340263752899923,
0.21119069394696974,
0.02791820813292813,
-0.3270633105274758,
0.08975108940236352,
0.44029025688580486,
-0.6373563320829833,
0.43031272284545874,
-0.1650642834886438,
0.03490771432362905,
-0.0031892209253436892]
db17 = [-0.00224180700103879,
0.025985393703623173,
-0.13121490330791097,
0.3703507241528858,
-0.6109966156850273,
0.5183157640572823,
-0.027314970403312946,
-0.32832074836418546,
0.12659975221599248,
0.19731058956508457,
-0.10113548917744287,
-0.12681569177849797,
0.05709141963185808,
0.08110598665408082,
-0.022312336178011833,
-0.04692243838937891,
0.0032709555358783646,
0.022733676583919053,
0.0030429899813869555,
-0.008602921520347815,
-0.002967996691518064,
0.0023012052421511474,
0.001436845304805,
-0.00032813251941022427,
-0.0004394654277689454,
-2.5610109566546042e-05,
8.204803202458212e-05,
2.318681379876164e-05,
-6.990600985081294e-06,
-4.505942477225963e-06,
-3.0165496099963414e-07,
2.9577009333187617e-07,
8.423948446008154e-08,
7.26749296856637e-09]
db18 = [-0.0015763102184365595,
0.01928853172409497,
-0.10358846582214751,
0.31467894133619284,
-0.5718268077650818,
0.571801654887122,
-0.14722311196952223,
-0.2936540407357981,
0.21648093400458224,
0.14953397556500755,
-0.16708131276294505,
-0.09233188415030412,
0.10675224665906288,
0.0648872162123582,
-0.05705124773905827,
-0.04452614190225633,
0.023733210395336858,
0.026670705926689853,
-0.006262167954438661,
-0.013051480946517112,
-0.00011863003387493042,
0.004943343605456594,
0.0011187326669886426,
-0.0013405962983313922,
-0.0006284656829644715,
0.0002135815619103188,
0.00019864855231101547,
-1.535917123021341e-07,
-3.741237880730847e-05,
-8.520602537423464e-06,
3.3326344788769603e-06,
1.768712983622886e-06,
7.691632689865049e-08,
-1.1760987670250871e-07,
-3.06883586303703e-08,
-2.507934454941929e-09]
db19 = [-0.0011086697631864314,
0.01428109845082521,
-0.08127811326580564,
0.26438843174202237,
-0.5244363774668862,
0.6017045491300916,
-0.2608949526521201,
-0.22809139421653665,
0.28583863175723145,
0.07465226970806647,
-0.21234974330662043,
-0.03351854190320226,
0.14278569504021468,
0.02758435062488713,
-0.0869067555554507,
-0.026501236250778635,
0.04567422627778492,
0.021623767409452484,
-0.019375549889114482,
-0.013988388678695632,
0.005866922281112195,
0.007040747367080495,
-0.0007689543592242488,
-0.002687551800734441,
-0.00034180865344939543,
0.0007358025205041731,
0.0002606761356811995,
-0.00012460079173506306,
-8.711270467250443e-05,
5.105950487090694e-06,
1.664017629722462e-05,
3.0109643163099385e-06,
-1.531931476697877e-06,
-6.86275565779811e-07,
-1.447088298804088e-08,
4.636937775802368e-08,
1.1164020670405678e-08,
8.666848839034483e-10]
db20 = [-0.0007799536136659112,
0.010549394624937735,
-0.06342378045900529,
0.21994211355113222,
-0.4726961853103315,
0.6104932389378558,
-0.36150229873889705,
-0.13921208801128787,
0.3267868004335376,
-0.016727088308801888,
-0.22829105082013823,
0.039850246458519104,
0.1554587507060453,
-0.024716827337521424,
-0.10229171917513397,
0.005632246857685454,
0.061722899624668884,
0.0058746818113949465,
-0.03229429953011916,
-0.008789324924555765,
0.013810526137727442,
0.0067216273018096935,
-0.00442054238676635,
-0.003581494259744107,
0.0008315621728772474,
0.0013925596193045254,
5.349759844340453e-05,
-0.0003851047486990061,
-0.00010153288973669777,
6.774280828373048e-05,
3.710586183390615e-05,
-4.376143862182197e-06,
-7.241248287663791e-06,
-1.0119940100181473e-06,
6.847079596993149e-07,
2.633924226266962e-07,
-2.0143220235374613e-10,
-1.814843248297622e-08,
-4.05612705554717e-09,
-2.998836489615753e-10]
