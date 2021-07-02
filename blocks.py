from tensorflow.keras import layers, activations
import tensorflow.keras as keras
import tensorflow_addons as tfa


class SE(layers.Layer):
    def __init__(self,
                 in_channels,
                 reduction_ratio,
                 activation1=activations.relu,
                 activation2=activations.sigmoid):
        super(SE, self).__init__()
        self.reduced = in_channels // reduction_ratio
        self.activation1 = activation1
        self.activation2 = activation2

    def build(self, input_shape):
        self.pool = layers.GlobalAvgPool2D()
        self.dense1 = layers.Dense(self.reduced, activation=self.activation1)
        self.dense2 = layers.Dense(input_shape[-1], activation=self.activation2)
        self.mult = layers.Multiply()

    def call(self, inputs, **kwargs):
        out = self.pool(inputs)
        out = self.dense1(out)
        out = self.dense2(out)
        return self.mult([inputs, out])


class SkipConnect_d(layers.Layer):
    def __init__(self, out_channels):
        super(SkipConnect_d, self).__init__()
        self.conv = layers.Conv2D(out_channels, (1, 1), use_bias=False)
        self.norm = layers.BatchNormalization()
        self.pool = layers.AvgPool2D()

    def call(self, inputs, **kwargs):
        out = self.pool(inputs)
        out = self.conv(out)
        return self.norm(out) if self.norm is not None else out


class ECA(layers.Layer):
    def __init__(self, kernel_size, activation=activations.sigmoid):
        super(ECA, self).__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.mult = layers.Multiply()

    def build(self, input_shape):
        self.pool = tfa.layers.AdaptiveAveragePooling2D((1, 1))
        self.conv = layers.Conv1D(filters=1,
                                  kernel_size=self.kernel_size,
                                  strides=1,
                                  padding="same",
                                  activation=self.activation)

    def call(self, inputs, **kwargs):
        out = self.pool(inputs)
        out = self.conv(out)
        return self.mult([inputs, out])


class GhostConv(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 intermediate_norm=True,
                 activation=activations.relu):
        super(GhostConv, self).__init__()
        if padding != "valid" and padding != "same":
            raise ValueError("padding must be valid or same")
        if filters % 2 != 0:
            raise ValueError("number of filters must be even")
        self.conv1 = layers.Conv2D(filters=filters // 2,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding=padding,
                                   use_bias=not intermediate_norm)

        self.conv2 = keras.Sequential([])
        if intermediate_norm:
            self.conv2.add(intermediate_norm)
        self.conv2.add(layers.Activation(activation))
        self.conv2.add(layers.DepthwiseConv2D((1, 1)))
        self.concat = layers.Concatenate()

    def call(self, inputs, **kwargs):
        out1 = self.conv1(inputs)
        out2 = self.conv2(out1)
        return self.concat([out1, out2])


class ConvNormAct(layers.Layer):
    def __init__(self,
                 filters=1,
                 kernel_size=1,
                 strides=(1, 1),
                 padding="valid",
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 depth_multiplier=1,
                 activation=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 depthwise=False,
                 **kwargs):
        super(ConvNormAct, self).__init__()
        if depthwise:
            self.conv = layers.DepthwiseConv2D(kernel_size=kernel_size,
                                               strides=strides,
                                               padding=padding,
                                               depth_multiplier=depth_multiplier,
                                               data_format=data_format,
                                               dilation_rate=dilation_rate,
                                               use_bias=False,
                                               depthwise_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer,
                                               depthwise_regularizer=kernel_regularizer,
                                               bias_regularizer=bias_regularizer,
                                               activity_regularizer=activity_regularizer,
                                               depthwise_constraint=kernel_constraint,
                                               bias_constraint=bias_constraint)
        else:
            self.conv = layers.Conv2D(filters,
                                      kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      data_format=data_format,
                                      dilation_rate=dilation_rate,
                                      groups=groups,
                                      activation="linear",
                                      use_bias=False,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer,
                                      kernel_constraint=kernel_constraint,
                                      bias_constraint=bias_constraint,
                                      **kwargs)
        self.norm = layers.BatchNormalization()
        self.activation = layers.Activation(activation)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.activation(x)
        return x


class MBBlock(layers.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=(1, 1),
                 use_skipconnect_d=True,
                 expansion=6,
                 se_reduction=4,
                 se_prob_act=activations.sigmoid,
                 fused=False,
                 activation=activations.relu,
                 dropout_type="spatial",
                 dropout=.1,
                 drop_connect=.1):
        super(MBBlock, self).__init__()
        self.expand = ConvNormAct(filters=self._round(in_channels * expansion),
                                  kernel_size=1,
                                  activation=activation)
        self.depthwise = ConvNormAct(kernel_size=kernel_size,
                                     strides=strides,
                                     padding="same",
                                     depthwise=True) if not fused else keras.Sequential()
        self.se = keras.Sequential()
        if dropout != 0:
            if dropout_type is None or dropout_type == "dropout":
                self.se.add(layers.Dropout(dropout))
            elif dropout_type == "spatial":
                self.se.add(layers.SpatialDropout2D(dropout))
            else:
                raise ValueError("unsupported dropout type")
        if se_reduction is not None and se_reduction > 0:
            self.se.add(SE(in_channels=in_channels,
                           reduction_ratio=se_reduction,
                           activation1=activation,
                           activation2=se_prob_act))
        if drop_connect > 0 and strides == 1 and in_channels == out_channels:
            self.add = tfa.layers.StochasticDepth(survival_probability=1 - drop_connect)
        else:
            self.add = layers.Add()
        self.project = ConvNormAct(filters=out_channels,
                                   kernel_size=kernel_size if fused else 1,
                                   activation=activation,
                                   padding="same",
                                   strides=2 if fused and strides == 2 else 1)
        self.shortcut = SkipConnect_d(out_channels=out_channels) if ((strides == 2 or strides == (2, 2))
                                                                     and use_skipconnect_d) else keras.Sequential()

    def _round(self, num, divisor=8):
        if num % divisor != 0:
            return (num // divisor + 1) * divisor
        else:
            return num

    def call(self, inputs, *args, **kwargs):
        x = self.expand(inputs)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        inputs = self.shortcut(inputs)
        x = self.add([inputs, x])
        return x
