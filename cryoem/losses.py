
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MAE, MSE
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.util import safe_ops, asserts, shape
from tensorflow_graphics.math import vector
from tensorflow_graphics.geometry.transformation.quaternion import relative_angle
import tensorflow_graphics as tfg


def cosine_distance(vests):
    """Cosine distance between two feature vectors from every projection"""
    x, y = vests
    xy_sum_square = K.sum(x * y, axis=1, keepdims=True) 
    xx_sum_square = K.sum(x * x, axis=1, keepdims=True)
    xx_sum_square = K.maximum(xx_sum_square,  1e-08) 
    yy_sum_square = K.sum(y * y, axis=1, keepdims=True)
    yy_sum_square = K.maximum(yy_sum_square, 1e-08) 
    
    cos_theta = tf.divide(xy_sum_square, K.sqrt(xx_sum_square)*K.sqrt(yy_sum_square))
    eps = K.epsilon()
    return 2*tf.acos(tf.clip_by_value(cos_theta, 0.0+eps, 1.0-eps)) 

def cos_dist_output_shape(shapes):
    """The output shape of cosine distance"""
    shape1, shape2 = shapes
    return (shape1[0], 1)

def cus_mae6(y_true, y_pred):
    """Mean absolute error"""
    #u1 = y_pred[...,0:3]
    #u2 = y_pred[...,3:]
    #e1 = u1 / tf.clip_by_value(tf.norm(u1, ord=2, axis=-1, keepdims=True), clip_value_min=1E-5, clip_value_max=np.inf)
    #e2 = u2 / tf.clip_by_value(tf.norm(u2, ord=2, axis=-1, keepdims=True), clip_value_min=1E-5, clip_value_max=np.inf)
    #y_pred = tf.concat([e1,e2], axis=1)
    return MAE(y_true, y_pred) 

def cus_mae66(loss_w):
    def cus_mae_function(y_true,y_pred):
        return loss_w*cus_mae6(y_true,y_pred)
    return cus_mae_function

def mae(y_true, y_pred):
    """Mean absolute error"""
    return MAE(y_true, y_pred)

def mse(y_true, y_pred):
    """Mean square error"""
    return MSE(y_true, y_pred) 


def quat_norm_diff(q_a, q_b):
    assert(q_a.shape == q_b.shape)
    #assert(q_a.shape[-1] == 4)
    #if len(q_a.shape) < 2:
        #q_a = tf.expand_dims(q_a, axis=0)
        #q_b = tf.expand_dims(q_a, axis=0)
    return tf.squeeze(tf.math.minimum(tf.norm(q_a-q_b, axis=1), tf.norm(q_a+q_b, axis=1)))

def quat_chordal_squared_loss(q, q_target, reduce=True):
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses =  2*d*d*(4. - d*d) 
    loss = tf.math.reduce_mean(losses) if reduce else losses
    return loss

def d_q2(q1, q2):
    """Distance between two quaternions"""
    q1 = tf.cast(tf.convert_to_tensor(value=q1), dtype=tf.float32)
    q2 = tf.cast(tf.convert_to_tensor(value=q2), dtype=tf.float32)
    
    #shape.check_static(tensor=q1, tensor_name="quaternion1", has_dim_equals=(-1, 4)) #other wise there are problems when compiling
    #shape.check_static(tensor=q2, tensor_name="quaternion2", has_dim_equals=(-1, 4))
    q1 = tf.reshape(q1,[-1,4])
    q2 = tf.reshape(q2,[-1,4])
    q1 = quaternion.normalize(q1)
    q2 = quaternion.normalize(q2)
    
    dot_product = vector.dot(q1, q2, keepdims=False)
    
    # Ensure dot product is in range [-1. 1].
    eps_dot_prod = 1.8 * asserts.select_eps_for_addition(dot_product.dtype)
    dot_product = safe_ops.safe_shrink(dot_product, -1, 1, open_bounds=False, eps=eps_dot_prod)

    return 2.0 * tf.acos(tf.abs(dot_product)) 


def d_q3(qs):
    """Distance between two quaternions"""
    q1, q2 = qs
    q1 = tf.cast(tf.convert_to_tensor(value=q1), dtype=tf.float32)
    q2 = tf.cast(tf.convert_to_tensor(value=q2), dtype=tf.float32)
    
    #shape.check_static(tensor=q1, tensor_name="quaternion1", has_dim_equals=(-1, 4)) #other wise there are problems when compiling
    #shape.check_static(tensor=q2, tensor_name="quaternion2", has_dim_equals=(-1, 4))
    q1 = tf.reshape(q1,[-1,4])
    q2 = tf.reshape(q2,[-1,4])
    q1 = quaternion.normalize(q1)
    q2 = quaternion.normalize(q2)
    
    dot_product = vector.dot(q1, q2, keepdims=False)
    
    # Ensure dot product is in range [-1. 1].
    eps_dot_prod = 1.8 * asserts.select_eps_for_addition(dot_product.dtype)
    dot_product = safe_ops.safe_shrink(dot_product, -1, 1, open_bounds=False, eps=eps_dot_prod)

    return 2.0 * tf.acos(tf.abs(dot_product)) 

def cus_loss(loss_w):
    def cus_loss_function(y_true,y_pred):
        return loss_w*tf.reduce_mean(d_q2(y_true,y_pred))
    return cus_loss_function

def cus_loss2(loss_w):
    def cus_loss_function(y_true,y_pred):
        return loss_w*quat_chordal_squared_loss(y_true, y_pred)
    return cus_loss_function

def cus_mae(loss_w):
    def cus_mae_function(y_true,y_pred):
        return loss_w*mae(y_true,y_pred)
    return cus_mae_function


def cus_neg(loss_w):
    def cus_neg_function(y_true,y_pred):
        return loss_w*(-tf.norm(y_pred))
    return cus_neg_function

def cus_loss_function(y_true,y_pred):
    return tf.reduce_mean(d_q2(y_true,y_pred))