import tensorflow as tf
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

# -------------------------------Unet Model( image segmentation )---------------------------------------------
class UnetModelGenerator(object):

    @staticmethod
    def dice_coeff(y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        loss = losses.binary_crossentropy(y_true, y_pred) + UnetModelGenerator.dice_loss(y_true, y_pred)
        return loss

    @staticmethod
    def dice_loss(y_true, y_pred):
        loss = 1 - UnetModelGenerator.dice_coeff(y_true, y_pred)
        return loss

    @staticmethod
    def conv_block(input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder

    @staticmethod
    def encoder_block(input_tensor, num_filters):
        encoder = UnetModelGenerator.conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

        return encoder_pool, encoder

    @staticmethod
    def decoder_block(input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    @staticmethod
    def generate_outputs(inputs):
        encoder0_pool, encoder0 = UnetModelGenerator.encoder_block(inputs, 32)  # 128
        encoder1_pool, encoder1 = UnetModelGenerator.encoder_block(encoder0_pool, 64)  # 64
        encoder2_pool, encoder2 = UnetModelGenerator.encoder_block(encoder1_pool, 128)  # 32
        encoder3_pool, encoder3 = UnetModelGenerator.encoder_block(encoder2_pool, 256)  # 16
        encoder4_pool, encoder4 = UnetModelGenerator.encoder_block(encoder3_pool, 512)  # 8
        center = UnetModelGenerator.conv_block(encoder4_pool, 1024)  # center
        decoder4 = UnetModelGenerator.decoder_block(center, encoder4, 512)  # 16
        decoder3 = UnetModelGenerator.decoder_block(decoder4, encoder3, 256)  # 32
        decoder2 = UnetModelGenerator.decoder_block(decoder3, encoder2, 128)  # 64
        decoder1 = UnetModelGenerator.decoder_block(decoder2, encoder1, 64)  # 128
        decoder0 = UnetModelGenerator.decoder_block(decoder1, encoder0, 32)  # 256
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
        return outputs

    def __init__(self, input_img_shape):
        self.inputs = layers.Input(shape=input_img_shape)
        self.outputs = UnetModelGenerator.generate_outputs(self.inputs)
        self.keras_model = models.Model(inputs=[self.inputs], outputs=[self.outputs])

    def summary(self):
        self.keras_model.summary()

    def get_model(self):
        return self.keras_model
# -------------------------------------------------------------------------------------------------------------
# --------------------- self-designed sequential nn model( image classification ）-----------------------------
class SeqNnModelGenerator(object):

    @staticmethod
    def loss(y_true, y_pred):
        return losses.categorical_crossentropy(y_true, y_pred)

    def __init__(self, input_img_shape, num_classes):
        self.num_classes = num_classes
        self.inputs = layers.Input(shape=input_img_shape)
        self.outputs = self.generate_outputs()
        self.keras_model = models.Model(inputs=[self.inputs], outputs=[self.outputs])

    def generate_outputs(self):
        flatterned = layers.Flatten()(self.inputs)
        d1 = layers.Dense(64, activation='relu')(flatterned)
        d1_dropout = layers.Dropout(rate=0.25)(d1)
        d2 = layers.Dense(128, activation='relu')(d1_dropout)
        d2_dropout = layers.Dropout(rate=0.5)(d2)
        outputs = layers.Dense(self.num_classes, activation='softmax')(d2_dropout)
        return outputs

    def summary(self):
        self.keras_model.summary()

    def get_model(self):
        return self.keras_model
# -------------------------------------------------------------------------------------------------------
# ---------------------------------- CNN model( image classification ) ----------------------------------
class CnnModel(object):

    def __init__(self, input_img_shape, num_classes):
        self.num_classes = num_classes
        self.inputs = layers.Input(shape=input_img_shape)
        self.outputs = self.generate_outputs()
        self.keras_model = models.Model(inputs=[self.inputs], outputs=[self.outputs])

    def generate_outputs(self):
        conv_l1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(self.inputs)
        conv_l1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv_l1)
        conv_l2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv_l1)
        conv_l2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv_l2)
        conv_l3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv_l2)
        conv_l3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv_l3)
        l3_dropout = layers.Dropout(rate=0.5)(conv_l3)
        flatterned = layers.Flatten()(l3_dropout)
        l4 = layers.Dense(1024, activation='relu')(flatterned)
        l4_droupout = layers.Dropout(rate=0.5)(l4)
        outputs = layers.Dense(self.num_classes, activation='softmax')(l4_droupout)
        return outputs

    def summary(self):
        self.keras_model.summary()

    def get_model(self):
        return self.keras_model
# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------Inception V3 Model-------------------------------------------------
# class InceptionModel(object):
#
#     def __init__(self, input_img_shape, num_classes):
#         self.num_classes = num_classes
#         self.inputs = layers.Input(shape=input_img_shape)
#         self.outputs = self.generate_outputs()
#         self.keras_model = models.
#
#     def generate_outputs(self):
#         conv_l1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(self.inputs)
#         conv_l1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv_l1)
#         conv_l2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv_l1)
#         conv_l2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv_l2)
#         conv_l3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv_l2)
#         conv_l3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv_l3)
#         l3_dropout = layers.Dropout(rate=0.5)(conv_l3)
#         flatterned = layers.Flatten()(l3_dropout)
#         l4 = layers.Dense(1024, activation='relu')(flatterned)
#         l4_droupout = layers.Dropout(rate=0.5)(l4)
#         outputs = layers.Dense(self.num_classes, activation='softmax')(l4_droupout)
#         return outputs
#
#     def summary(self):
#         self.keras_model.summary()
#
#     def get_model(self):
#         return self.keras_model