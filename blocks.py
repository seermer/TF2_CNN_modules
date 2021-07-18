from tensorflow.keras import layers, activations
import tensorflow.keras as keras
import tensorflow_addons as tfa


def roundup(num, divisor=8):
    if num % divisor != 0:
        return (num // divisor + 1) * divisor
    else:
        return num


def get_norm(norm):
    if norm == "bn":
        return layers.BatchNormalization()
    elif norm == "ln":
        return layers.LayerNormalization()
    elif norm == "in":
        return tfa.layers.InstanceNormalization()
    elif norm == "gn":
        return tfa.layers.GroupNormalization()
    elif norm == "frn":
        return tfa.layers.FilterResponseNormalization()
    elif norm == "wn":
        return tfa.layers.WeightNormalization()
    elif norm == "pn":
        return tfa.layers.PoincareNormalize()
    elif norm == "sn":
        return tfa.layers.SpectralNormalization()
    else:
        raise ValueError(str(norm) + " is not a supported normalization")


class SE(layers.Layer):
    """
    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
    """

    def __init__(self,
                 in_channels,
                 reduction_ratio,
                 activation1=activations.relu,
                 activation2=activations.sigmoid):
        super(SE, self).__init__()
        self.pool = layers.GlobalAvgPool2D()
        self.dense1 = layers.Dense(roundup(in_channels // reduction_ratio), activation=activation1)
        self.mult = layers.Multiply()
        self.activation2 = activation2

    def build(self, input_shape):
        self.dense2 = layers.Dense(input_shape[-1], activation=self.activation2)

    def call(self, inputs, **kwargs):
        out = self.pool(inputs)
        out = self.dense1(out)
        out = self.dense2(out)
        return self.mult([inputs, out])


class SkipConnect_d(layers.Layer):
    """
    Bag of Tricks for Image Classification with Convolutional Neural Networks
    https://arxiv.org/abs/1812.01187
    """

    def __init__(self, out_channels, norm="bn"):
        super(SkipConnect_d, self).__init__()
        self.conv = layers.Conv2D(out_channels, (1, 1), use_bias=False)
        self.norm = get_norm(norm)
        self.pool = layers.AvgPool2D(padding="same")

    def call(self, inputs, **kwargs):
        out = self.pool(inputs)
        out = self.conv(out)
        return self.norm(out)


class ECA(layers.Layer):
    """
    ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    https://arxiv.org/abs/1910.03151
    """

    def __init__(self, kernel_size, activation=activations.sigmoid):
        super(ECA, self).__init__()
        self.conv = layers.Conv1D(filters=1,
                                  kernel_size=kernel_size,
                                  strides=1,
                                  padding="same",
                                  use_bias=False,
                                  activation=activation)
        self.pool = tfa.layers.AdaptiveAveragePooling2D((1, 1))
        self.mult = layers.Multiply()

    def call(self, inputs, **kwargs):
        out = self.pool(inputs)
        out = self.conv(out)
        return self.mult([inputs, out])


class GhostConv(layers.Layer):
    """
    GhostNet: More Features from Cheap Operations
    https://arxiv.org/abs/1911.11907
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 intermediate_norm="bn",
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

        self.conv2 = keras.Sequential()
        if intermediate_norm:
            self.conv2.add(get_norm(intermediate_norm))
        self.conv2.add(layers.Activation(activation))
        self.conv2.add(layers.DepthwiseConv2D((1, 1)))
        self.concat = layers.Concatenate()

    def call(self, inputs, **kwargs):
        out1 = self.conv1(inputs)
        out2 = self.conv2(out1)
        return self.concat([out1, out2])


class ConvNormAct(layers.Layer):
    """
    a combination of conv2d + batchnorm + activation
    """

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
                 norm="bn",
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
        self.norm = get_norm(norm)
        self.activation = layers.Activation(activation)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.activation(x)
        return x


class MBBlock(layers.Layer):
    """
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
    """

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
        if (strides == 1 or strides == (1, 1)) and in_channels != out_channels:
            raise ValueError
        self.expand = ConvNormAct(filters=roundup(in_channels * expansion),
                                  kernel_size=1,
                                  activation=activation)
        self.depthwise = ConvNormAct(kernel_size=kernel_size,
                                     strides=strides,
                                     padding="same",
                                     depthwise=True) if not fused else lambda val: val
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
        self.add = tfa.layers.StochasticDepth(1 - drop_connect) if drop_connect > 0 else layers.Add()
        self.project = ConvNormAct(filters=out_channels,
                                   kernel_size=kernel_size if fused else 1,
                                   activation=activations.linear,
                                   padding="same",
                                   strides=2 if fused and strides == 2 else 1)
        self.shortcut = SkipConnect_d(out_channels=out_channels) if ((strides == 2 or strides == (2, 2))
                                                                     and use_skipconnect_d) else lambda val: val
        self.use_skipconnect_d = use_skipconnect_d
        self.strides = strides

    def call(self, inputs, *args, **kwargs):
        x = self.expand(inputs)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if (self.strides == 1 or self.strides == (1, 1)) or self.use_skipconnect_d:
            inputs = self.shortcut(inputs)
            x = self.add([inputs, x])
        return x


class Bottleneck(layers.Layer):
    """
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
    """

    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 reduction=4,
                 groups=1,
                 use_skipconnect_d=True,
                 se_reduction=4,
                 se_prob_act=activations.sigmoid,
                 activation=activations.relu,
                 drop_connect=.1,
                 norm="bn"):
        super(Bottleneck, self).__init__()
        self.add = tfa.layers.StochasticDepth(1 - drop_connect) if drop_connect > 0 else layers.Add()

        if strides == 2 or strides == (2, 2):
            if use_skipconnect_d:
                self.shortcut = SkipConnect_d(out_channels=filters, norm=norm)
            else:
                self.shortcut = ConvNormAct(filters=filters, kernel_size=1, strides=2, activation="linear", norm=norm)
        else:
            self.shortcut = lambda val: val
        self.conv1 = ConvNormAct(filters=filters // reduction, kernel_size=1, activation=activation, norm=norm)
        self.conv2 = ConvNormAct(filters=filters // reduction,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding="same",
                                 groups=groups,
                                 activation=activation,
                                 norm=norm)
        self.conv3 = keras.Sequential([
            layers.Conv2D(filters=filters, kernel_size=1),
            get_norm(norm)
        ])
        self.activation = layers.Activation(activation)
        self.se = SE(in_channels=filters,
                     reduction_ratio=se_reduction,
                     activation1=activation,
                     activation2=se_prob_act) if se_reduction > 0 else lambda val: val
        self.stride1 = (strides == 1 or strides == (1, 1))
        self.filters = filters
        self.norm = norm

    def build(self, input_shape):
        if self.stride1 and self.filters != input_shape[-1]:
            self.shortcut = ConvNormAct(filters=self.filters, kernel_size=1, strides=1,
                                        activation="linear", norm=self.norm)

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)
        shortcut = self.shortcut(inputs)
        x = self.add([shortcut, x])
        return self.activation(x)


class ResidualBlock(layers.Layer):
    """
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
    """

    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 use_skipconnect_d=True,
                 se_reduction=4,
                 se_prob_act=activations.sigmoid,
                 activation=activations.relu,
                 drop_connect=.1,
                 norm="bn"):
        super(ResidualBlock, self).__init__()
        self.add = tfa.layers.StochasticDepth(1 - drop_connect) if drop_connect > 0 else layers.Add()
        if strides == 2 or strides == (2, 2):
            self.shortcut = SkipConnect_d(out_channels=filters,
                                          norm=norm) if use_skipconnect_d else ConvNormAct(filters=filters,
                                                                                           kernel_size=1,
                                                                                           strides=2,
                                                                                           norm=norm)
        else:
            self.shortcut = None
        self.conv1 = ConvNormAct(filters=filters, kernel_size=kernel_size, strides=strides,
                                 padding="same", activation=activation, norm=norm)
        self.conv2 = keras.Sequential([
            layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same"),
            get_norm(norm)
        ])
        self.activation = layers.Activation(activation)
        self.se = SE(in_channels=filters,
                     reduction_ratio=se_reduction,
                     activation1=activation,
                     activation2=se_prob_act) if se_reduction > 0 else lambda val: val
        self.stride1 = (strides == 1 or strides == (1, 1))
        self.filters = filters
        self.norm = norm

    def build(self, input_shape):
        if self.stride1:
            if self.filters != input_shape[-1]:
                self.shortcut = ConvNormAct(filters=self.filters, kernel_size=1, strides=1,
                                            activation="linear", norm=self.norm)
            else:
                self.shortcut = lambda val: val

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.se(x)
        shortcut = self.shortcut(inputs)
        x = self.add([shortcut, x])
        return self.activation(x)
