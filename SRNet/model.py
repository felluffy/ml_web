from distutils.command.build import build
import os 
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import numba as nb
import cv2

K = keras.backend

def build_discriminator_loss(x, name='discriminator_loss'):
    return 0.0

def build_generator_loss(a, b, c, d, e):
    return 0.0

class SRNet():
    def __init__(self, shape=[224, 224], name='', lr=0.05) -> None:
        self.shape = shape 
        self.lr = lr 
        self.name = name
        self.features = 32
        self.graph = tf.Graph()
        self.i_t = None
        self.i_s = None
        self.t_sk = None
        self.t_t = None
        self.t_b = None
        self.t_f = None
        self.mask_t = None
        self.gloabl_step = tf.Variable(tf.constant(0))
        
        
        # with self.graph.as_default():
        #     self.i_t = tf.place
    def res_block(self, x, activation = keras.activations.selu, padding='SAME', name='res_block'):
        features = x.shape[-1]
        xin = x
        x = keras.layers.Conv2D(features // 4, 1, 1, padding, activation=activation, name = name+"_conv1")(x)
        x = keras.layers.Conv2D(features // 4, 3, 1, padding, activation=activation, name = name+"_conv2")(x)
        x = keras.layers.Conv2D(features, 1, 1, padding, activation=None, name = name+"_conv3")(x)
        x = keras.layers.Add()([x, xin])
        x = keras.layers.BatchNormalization(name=name+"_batch_norm")(x)
        x = keras.layers.Activation(activation, name=name+'act_out')(x)
        return x
    
    def conv_bn_relu(self, x, features=None, activation=keras.activations.selu, padding='SAME', name='conv_bn_relu'):
        if features == None:
            features = x.shape[-1]
        x = keras.layers.Conv2D(features, 3, 1, padding=padding, activation=None, name=name+'_conv1')(x)
        x = keras.layers.BatchNormalization(name=name+'_batch_norm')(x)
        x = keras.layers.Activation(activation)(x)
        
    def build_res_net(self, x, activation=keras.activations.selu, padding='SAME', name='res_net'):
        x = self.res_block(x, activation=activation, padding=padding, name=name+'_block1')
        x = self.res_block(x, activation=activation, padding=padding, name=name+'_block2')
        x = self.res_block(x, activation=activation, padding=padding, name=name+'_block3')
        x = self.res_block(x, activation=activation, padding=padding, name=name+'_block4')
        return x

    def build_encoder_net(self, x, activation=keras.activations.selu, padding='SAME', name='encoder_net', get_feature_map=False):
        features = self.features
        x = self.conv_bn_relu(x, features, name=name+'_conv1_1')
        x = self.conv_bn_relu(x, features, name=name+'_conv1_2')
        
        features *= 2
        x = keras.layers.Conv2D(features, 3, 2, activation=activation, padding=padding, name=name+"_pool1")(x)
        x = self.conv_bn_relu(x, features, name=name+'_conv2_1')
        x = self.conv_bn_relu(x, features, name=name+'_conv2_2')
        
        f1=x
        
        features *= 2
        x = keras.layers.Conv2D(features, 3, 2, activation=activation, padding=padding, name=name+"_pool2")(x)
        x = self.conv_bn_relu(x, features, name=name+'_conv3_1')
        x = self.conv_bn_relu(x, features, name=name+'_conv3_2')
        
        f2=x
        
        features *= 2
        x = keras.layers.Conv2D(features, 3, 2, activation=activation, padding=padding, name=name+"_pool3")(x)
        x = self.conv_bn_relu(x, features, name=name+'_conv4_1')
        x = self.conv_bn_relu(x, features, name=name+'_conv4_2')
        
        if get_feature_map:
            return x, [f1,f2]
        else:
            return x
        
    def build_decoder_net(self, x, concat=False, activation=keras.activations.selu, padding='SAME', name='decoder_net', get_feature_map = False):
        f1, f2, f3 = None, None, None
        if concat and concat[0] is not None:
            x = keras.layers.Concatenate([x, concat[0]], axis=-1, name=name+'concat_1')
        x = self.conv_bn_relu(x, 8 * self.features, name=name+'conv1_1')
        x = self.conv_bn_relu(x, 8 * self.features, name=name+'conv1_2')
        if get_feature_map:
            f1 = x 
        x = keras.layers.Conv2DTranspose(4 * self.features, 3, 2, activation=activation, padding=padding, name=name+'conv_trans_1')(x)
        
        if concat and concat[1] is not None:
            x = keras.layers.Concatenate([x, concat[1]], axis=-1, name=name+'concat_2')
        x = self.conv_bn_relu(x, 4 * self.features, name=name+'conv2_1')
        x = self.conv_bn_relu(x, 4 * self.features, name=name+'conv2_2')
        if get_feature_map:
            f2 = x 
        x = keras.layers.Conv2DTranspose(2 * self.features, 3, 2, activation=activation, padding=padding, name=name+'conv_trans_2')(x)
        
        if concat and concat[2] is not None:
            x = keras.layers.Concatenate([x, concat[1]], axis=-1, name=name+'concat_3')
        x = self.conv_bn_relu(x, 2 * self.features, name=name+'conv3_1')
        x = self.conv_bn_relu(x, 2 * self.features, name=name+'conv3_2')
        if get_feature_map:
            f3 = x 
        
        x = keras.layers.Conv2DTranspose(self.features, 3, 2, activation=activation, padding=padding, name=name+"conv_trans_3")(x)
        x = self.conv_bn_relu(x, self.features, name=name+'conv4_1')
        x = self.conv_bn_relu(x, self.features, name=name+'conv4_2')
        if get_feature_map: 
            return x, [f1, f2, f3]
        else:
            return x
        
    def build_text_conversion_net(self, x_t, x_s, padding='SAME', name='tcn_'):
        x_t = self.build_encoder_net(x_t, name = name + 't_encoder')
        x_t = self.build_res_net(x_t, name=name+'t_resnet`')
        
        x_s = self.build_encoder_net(x_s, name=name+'s_encoder')
        x_s = self.build_res_net(x_s, name=name+'s_resnet')
        
        x = keras.layers.Concatenate([x_t, x_s], axis=-1, name=name+'concat')
        
        y_sk = self.build_decoder_net(x, name=name+'t_decoder')
        y_sk_out = keras.layers.Conv2D(1, 3, 1, activation='sigmoid', padding=padding, name=name+'sk_out')(y_sk)
        
        y_t = self.build_decoder_net(x, name=name+'t_decoder')
        y_t = keras.layers.Concatenate([y_sk, y_t], axis=-1, name=name+'concat2')
        y_t = self.conv_bn_relu(y_t, name=name+'t_conv_bn')

        y_t_out = keras.layers.Conv2D(3, 3, 1, activation='tanh', padding=padding, name=name+'t_out')(y_t)
        return y_sk_out, y_t_out
    
    
    def build_background_inpainting_net(self, x, padding='same', name='background_inpainter_'):
        x, f_encoder= self.build_encoder_net(x, name=name+'encoder', get_feature_map=True)
        x = self.build_res_net(x, name=name+'_res')
        x, concated = self.build_decoder_net(x, concat = [None] + f_encoder, name=name+'_decoder', get_feature_map=True)
        x = keras.layers.Conv2D(3, 3, 1, activation='tanh', padding=padding, name=name+'_out')(x)
        return x, concated
    
    def build_fusion_net(self, x, concated, padding='SAME', name='fusion_net_'):
        x = self.build_encoder_net(x, name=name+'encoder')
        x = self.build_res_net(x, name=name+'res')
        x = self.build_decoder_net(x, concated, name=name+'decoder')
        x = keras.layers.Conv2D(3, 3, 1, activation='tanh', padding=padding, name=name+'out')
        return x
    
    def build_discriminator(self, x, activation=keras.activations.selu, padding='SAME', name='discriminator_'):
        x = keras.layers.Conv2D(64, 3, 2, activation=activation, padding=padding, name=name+'conv1')(x)
        x = keras.layers.Conv2D(128, 3, 2, activation=None, padding=padding, name=name+'conv2')(x)
        x = keras.layers.BatchNormalization(name=name+'conv2_bn')(x)        
        x = keras.layers.Activation(activation=activation, name=name+'conv2_activation')(x)
        
        x = keras.layers.Conv2D(256, 3, 2, activation=None, padding=padding, name=name+'conv3')(x)
        x = keras.layers.BatchNormalization(name=name+'conv3_bn')(x)
        x = keras.layers.Activation(activation=activation, name='conv3_activation')(x)
        
        x = keras.layers.Conv2D(512, 3, 2, activation=None, padding=padding, name=name+'conv4')(x)
        x = keras.layers.BatchNormalization(name=name+'conv4_bn')(x)
        x = keras.layers.Activation(activation=activation, name='conv4_activation')(x)
        
        x = keras.layers.Conv2D(1, 3, 1, activation=None, padding=padding, name=name+'conv5')
        x = keras.layers.BatchNormalization(name=name+'conv5_bn')
        x = keras.layers.Activation(keras.activations.sigmoid, name=name+'sig_out')(x)
        return x
    
    def build_generator(self, inputs, name='generator_'):
        i_t, i_s = inputs
        o_sk, o_t = self.build_text_conversion_net(i_t, i_s, name=name+'tcn')
        o_b, concated = self.build_background_inpainting_net(i_s, name= name+'_background_inpainter')
        o_f = self.build_fusion_net(o_t, concated=concated, name=name+'fusion_net')
        return o_sk, o_t, o_b, o_f
    
    def build_network(self):
        i_t, i_s = self.i_t, self.i_s 
        t_sk, t_t, t_b, t_f, mask_t = self.t_sk, self.t_t, self.t_b, self.t_f, self.mask_t
        inputs = [i_t, i_s]
        labels = [t_sk, t_t, t_b, t_f]
        
        o_sk, o_t, o_b, o_f = self.build_generator(inputs)
        
        self.o_sk = tf.identity(o_sk, name='o_sk')
        self.o_t = tf.identity(o_t, name='o_t')
        self.o_b = tf.identity(o_b, name='o_b')
        self.o_f = tf.identity(o_f, name='o_f')
        
        i_db_true = keras.layers.Concatenate(axis=-1, name='db_true_concat')([t_b, i_s])
        i_db_pred = keras.layers.Concatenate(axis=-1, name='db_pred_concat')([o_b, i_s])
        i_db = keras.layers.Concatenate(axis=0, name='db_concat')([i_db_true, i_db_pred])
        
        i_df_true = keras.layers.Concatenate(axis=-1, name='df_true_concat')([t_f, i_t])
        i_df_pred = keras.layers.Concatenate(axis=-1, name='df_true_concat')([o_f, i_t])
        i_df = keras.layers.Concatenate(axis=0, name='df_concat')([i_df_true, i_df_pred])
        
        o_db = self.build_discriminator(i_db, name='db')
        o_df = self.build_discriminator(i_df, name='df')
        
        i_vgg = keras.layers.Concatenate(axis=0, name='vgg_concat')([t_f, o_f])
        
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
        for layer in vgg.layers:
            layer.trainable = False
            
        out_vgg = keras.models.Model(inputs=[vgg.input], outputs=[vgg.layers[1].output,
                                                             vgg.layers[4].output,
                                                             vgg.layers[7].output,
                                                             vgg.layers[12].output,
                                                             vgg.layers[17].output])
        
        out_d = [o_db, o_df]
        out_g = [o_sk, o_t, o_b, o_f, mask_t]
        
        db_loss = build_discriminator_loss(o_db, name='db_loss')
        df_loss = build_discriminator_loss(o_df, name='df_loss')
        
        self.d_loss_detail = [db_loss, df_loss]
        self.d_loss = tf.add(db_loss, df_loss, name='d_loss')
        self.g_loss, self.g_loss_detail = build_generator_loss(out_g, out_d, out_vgg, labels, name='g_loss')
        
        # path = 'path/to/vgg'
        # vgg = keras.models.load_model(path)
        # with tf.GradientTape() as tape:
            # o_vgg
        
        
    def build_optim(self, lr, decay_steps, decay_rate, staircase):
        self.learning_rate = keras.optimizers.schedules.ExponentialDecay(lr, decay_steps, decay_rate, staircase)