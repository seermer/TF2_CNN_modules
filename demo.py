from tensorflow.keras import Sequential, layers
import blocks


def repeat(model, filters, n, downsample=True):
    if downsample:
        model.add(blocks.ResidualBlock(filters=filters, use_skipconnect_d=False, strides=2,
                                       se_reduction=0, drop_connect=0))
        n -= 1
    for _ in range(n):
        model.add(blocks.ResidualBlock(filters=filters, use_skipconnect_d=False, se_reduction=0, drop_connect=0))


def get_model():
    model = Sequential([
        layers.InputLayer((224, 224, 3)),
        blocks.ConvNormAct(filters=64, kernel_size=7, strides=2, padding="same"),
        layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
    ])
    repeat(model, 64, 3, downsample=False)
    repeat(model, 128, 4)
    repeat(model, 256, 6)
    repeat(model, 512, 3)
    model.add(layers.GlobalAvgPool2D())
    model.add(layers.Dense(1000, activation="softmax"))
    return model


if __name__ == '__main__':
    m = get_model()
    m.summary()
