from tensorflow.keras import Sequential, layers

import blocks

"""
a simple demo for how to use these blocks, this shows how easy it is to build a resnet-34 with this util
"""

def repeat(model, filters, n):
    for _ in range(n):
        model.add(blocks.ResidualBlock(filters=filters, use_skipconnect_d=False, se_reduction=0, drop_connect=0))

def get_model():
    model = Sequential([
        layers.InputLayer((224, 224, 3)),
        blocks.ConvNormAct(filters=64, kernel_size=7, strides=2, padding="same"),
        layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
    ])
    repeat(model, 64, 3)
    model.add(blocks.ResidualBlock(filters=128, use_skipconnect_d=False, strides=2, se_reduction=0, drop_connect=0))
    repeat(model, 128, 3)
    model.add(blocks.ResidualBlock(filters=256, use_skipconnect_d=False, strides=2, se_reduction=0, drop_connect=0))
    repeat(model, 256, 5)
    model.add(blocks.ResidualBlock(filters=512, use_skipconnect_d=False, strides=2, se_reduction=0, drop_connect=0))
    repeat(model, 512, 2)
    model.add(layers.GlobalAvgPool2D())
    model.add(layers.Dense(1000, activation="softmax"))
    return model

if __name__ == '__main__':
    m = get_model()
    m.summary()
