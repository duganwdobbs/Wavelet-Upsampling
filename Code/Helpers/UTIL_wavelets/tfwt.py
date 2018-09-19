# import wavelets
import tensorflow as tf

class TFWAV:
  def __init__(self,trainable = False,wavelet = None):
    if trainable is None or wavelet is None:
      raise KeyError("Invalid values pass to TFWAV.")

    if type(wavelet) is str:
      try:
        import pywt
        wavelet = pywt.Wavelet(wavelet)
        wavelet = wavelet.filter_bank[1]
      except:
        try:
          wavelet = eval("wavelets." + wavelet)
        except:
          raise KeyError("Incomputable Wavelet type.")
      self.filter  = self.make_decomposition_filter(wavelet,trainable)
    if type(wavelet) is float:
      self.filter  = self.make_decomposition_filter(wavelet,trainable)
    # print("\rWAVELET: ", wavelet,"                           ")
    self.wavelet = wavelet
  # END __init__

  def make_decomposition_filter(self, wavelet, trainable = False):
    print("\rEvaluating Wavelet Filter...",end='')
    session = tf.Session()
    wavelet_length = len(wavelet)
    decomposition_high_pass_filter = tf.reverse(tf.constant(wavelet, dtype = tf.float32), [0])
    decomposition_low_pass_filter = tf.reshape(
                                    tf.multiply(
                                    tf.reshape(wavelet,[-1, 2]),
                                                                [-1, 1]),
                                                                         [-1])
    filter = tf.concat([tf.reshape(decomposition_low_pass_filter, [wavelet_length, 1, 1]),
                        tf.reshape(decomposition_high_pass_filter, [wavelet_length, 1, 1])],
                        axis = 2)
    filter = tf.Variable(session.run(filter),dtype = tf.float32, trainable = trainable, name = 'WaveletFilter')
    print("\rEvalutated, Closing session...",end='')
    session.close()
    print("\rTemp session closed...",end='')

    return filter
  # END make_decomposition_filter

  def dwt(self, inputs, padding_mode = "periodization"):
    with tf.variable_scope("DWT") as scope:
      wavelet = self.wavelet
      wavelet_length = len(wavelet)
      wavelet_pad = wavelet_length-2

      examples,height,width,channels = inputs.get_shape().as_list()

      filter = self.filter

      inputs = tf.transpose(inputs, [0, 3, 1, 2])
      # now [batch, channel, height, width]

      inputs = tf.reshape(inputs, [-1, width, 1])
      # now [batch channel height, width, 1]

      if wavelet_pad == 0:
        w_x = tf.nn.conv1d(inputs, filters = filter, stride = 2, padding = 'VALID')
        # [b c h, w, pass]
        w_x = tf.reshape(w_x, [-1, height, tf.cast(width/2, tf.int32), 2])
        # [b c, h, w, pass]
        w_x = tf.transpose(w_x, [0, 2, 3, 1])
        # [b c, w, pass, height]
        w_x = tf.reshape(w_x, [-1, height, 1])
        # [b c w pass, height, 1]
        w_x_y = tf.nn.conv1d(w_x, filters = filter, stride = 2, padding = 'VALID')
        # [b c w w_pass, height, h_pass]
        w_x_y = tf.reshape(w_x_y, [-1, channels, tf.cast(width/2, tf.int32), 2, tf.cast(height/2, tf.int32), 2])
        # [b, c, w, w_pass, height, h_pass]
      elif padding_mode is "periodization": # manually wrap
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

      w_x_y = tf.reshape(w_x_y,(2,2,examples,height//2, width//2, channels))


      return w_x_y
  # END dwt

  def idwt(self, inputs, padding_mode = "periodization"):
    with tf.variable_scope("IDWT") as scope:

      wavelet = self.wavelet
      wavelet_length = len(wavelet)
      wavelet_pad = wavelet_length-2

      _,_,examples,height,width,channels = inputs.get_shape().as_list()

      dest_width = (tf.shape(inputs)[4] * 2) - wavelet_pad
      dest_height = (tf.shape(inputs)[3] * 2) - wavelet_pad

      inputs = tf.transpose(inputs, [2, 5, 4, 0, 3, 1])
      # [b, channel, width, w_pass, height, h_pass]
      inputs = tf.reshape(inputs, [-1, tf.cast(height, tf.int32), 2])
      # [b channel width w_pass, height, h_pass]

      filter = self.filter

      if wavelet_pad == 0:
        w_x = tf.contrib.nn.conv1d_transpose(inputs,
                                             filter = filter,
                                             output_shape = [examples*channels*width*2, dest_height, 1],
                                             stride = 2,
                                             padding = "VALID")
        # [b channel width w_pass, height, 1]
        w_x = tf.reshape(w_x, [-1, tf.cast(width, tf.int32), 2, dest_height])
        # [b channel, width, w_pass, height]
        w_x = tf.transpose(w_x, [0, 3, 1, 2])
        # [b channel, height, width, width_pass]
        w_x = tf.reshape(w_x, [-1, tf.cast(width, tf.int32), 2])
        # [b channel height, width, width_pass]

        spatial = tf.contrib.nn.conv1d_transpose(w_x,
                                                 filter = filter,
                                                 output_shape = [examples*channels*dest_height, dest_width, 1],
                                                 stride = 2,
                                                 padding = "VALID")
        # [b channels height, width, 1]
        spatial = tf.reshape(spatial, [-1, channels, dest_height, dest_width])
        # [b, channels, height, width]
      elif padding_mode is "periodization":
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

      spatial = tf.reshape(spatial,(examples,height * 2, width * 2, channels))

      return w_x, spatial
  # END idwt

  def to_wav_format(self,ll,lh,hl,hh):
    # Format our wavelet features for IDWT
    with tf.variable_scope('Wavelet_Formatting') as scope:
      pred_dwt = tf.stack(
      [ tf.stack([ll , lh],-1),
        tf.stack([hl , hh],-1) ]
              ,-1)
      form_dwt = tf.transpose(pred_dwt, [4,5,0,1,2,3])
      return form_dwt

  def from_wav_format(self,wav):
    with tf.variable_scope("Wavelet_UnFormatting") as scope:
      ll = wav[0,0]
      lh = wav[0,1]
      hl = wav[1,0]
      hh = wav[1,1]
      return ll,lh,hl,hh

  def feat_norm(self,feat,kern,maxval = 1, minval = 0):
    max_k = tf.reduce_sum( tf.nn.relu( kern))
    min_k = tf.reduce_sum(-tf.nn.relu(-kern))
    max_v = maxval * max_k + minval * min_k
    min_v = minval * max_k + maxval * min_k
    feat  = (feat - min_v) / (max_v - min_v)
    return feat

  def feat_denorm(self,feat,kern,maxval = 1, minval = 0):
    max_k = tf.reduce_sum( tf.nn.relu( kern))
    min_k = tf.reduce_sum(-tf.nn.relu(-kern))
    max_v = maxval * max_k + minval * min_k
    min_v = minval * max_k + maxval * min_k
    feat  = feat * (max_v - min_v) + min_v
    return feat

  def wav_norm(self,ll,lh,hl,hh):
    filt_low    = self.filter[:,:,0]
    filt_high   = self.filter[:,:,1]
    ll = self.feat_norm(self.feat_norm(ll,filt_low),filt_low)
    lh = self.feat_norm(self.feat_norm(lh,filt_low),filt_high)
    hl = self.feat_norm(self.feat_norm(hl,filt_high),filt_low)
    hh = self.feat_norm(self.feat_norm(hh,filt_high),filt_high)
    return ll,lh,hl,hh

  def wav_denorm(self,ll,lh,hl,hh):
    filt_low    = self.filter[:,:,0]
    filt_high   = self.filter[:,:,1]
    ll = self.feat_denorm(self.feat_denorm(ll,filt_low) ,filt_low)
    lh = self.feat_denorm(self.feat_denorm(lh,filt_low) ,filt_high)
    hl = self.feat_denorm(self.feat_denorm(hl,filt_high),filt_low)
    hh = self.feat_denorm(self.feat_denorm(hh,filt_high),filt_high)
    return ll,lh,hl,hh
