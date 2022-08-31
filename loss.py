import numpy as np
import tensorflow as tf 

eps = 0.001


def discriminator_loss(x, name='d_loss', epsilon=eps):
    x_true, x_pred = tf.split(x, 2, name=name+'_split')
    d_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(x_true, epsilon, 1.0)) + tf.log(tf.clip_by_value(1.0 - x_pred, epsilon, 1.0)))
    return d_loss


def dice_loss(x_t, x_o, name = 'dice_loss', epsilon=eps):
    intersection = tf.reduce_sum(x_t * x_o, axis=[1,2,3])
    union = tf.reduce_sum(x_t, axis=[1,2,3]) + tf.reduce_sum(x_o, axis=[1,2,3])
    return 1. - tf.reduce_mean((2. * intersection + epsilon) / (union + epsilon), axis=0)

def l1_loss(x_t, x_o, name='l1_loss'):
    return tf.reduce_mean(tf.abs(x_t - x_o))

def l1_mask_loss(x_t, x_o, mask, name='l1_mask_loss'):
    mask_ratio = 1. -tf.reduce_mean(mask) / tf.cast(tf.size(mask), tf.float32)
    l1 = tf.abs(x_t, x_o)
    return mask_ratio * tf.reduce_mean(l1 * mask) + (1. - mask_ratio) * tf.reduce_mean(l1 * (1. - mask))

def perceptual_loss(x, name='perceptual_loss'):
    losses = []
    for i, f in enumerate(x):
        losses.append(l1_loss(f[0], f[1], name=name+'_l1_'+str(i)))
    losses = tf.stack(losses, axis=0, name=name+'_stack')
    return tf.reduce_mean(losses)

def gram_matrix(x, name='gram_matrix'):
    shp = tf.shape(x)
    matrix = tf.reshape(x, [-1, shp.shape[1] * shp.shape[2], shp.shape[3]])
    return tf.matmul(matrix, matrix, True) / tf.cast(shp.shape[1] * shp.shape[2], shp.shape[3], tf.float32)

    