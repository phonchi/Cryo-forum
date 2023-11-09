
import numpy as np
import os
from time import time, strftime

from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras import callbacks
#import tensorflow_similarity as tfsim
from tensorflow.keras.utils import Sequence
from cryoem.conversions import euler2quaternion, d_q, SO3_to_s2s2, s2s2_to_SO3, matrix2quaternion, euler2matrix, matrix2euler, euler2matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
import itertools
from skimage import transform




   

num_dec = 1
num_bins = 32

class DataGenerator_SO3(Sequence):
    """Custom datsaet generator used to create data for training of distance learning algorithm"""
    def __init__(self, X, y, list_ids, loss_fn, limit_num_pairs=None, limit_style="random", batch_size=256, shuffle=True):
        start_time = time()
        if batch_size > limit_num_pairs:
            raise Exception("Please specify limit_num_pairs that is much higher than batch_size")
        self.X = X  
        self.y = y  
        self.limit_num_pairs = limit_num_pairs
        self.list_ids = list_ids
        self.batch_size = batch_size  
        # all the possible combinations of 2 image id pairs
        self.pair_ids = np.array(list(zip(*list(map(lambda x: x.flatten(), np.meshgrid(list_ids, list_ids))))))  # e.g. train_idx
        self.loss_fn = loss_fn
        # Don't use all possible combination of pairs, limit them here
        if self.limit_num_pairs:
            limited_pair_indices = np.random.choice(np.arange(len(self.pair_ids)), size=self.limit_num_pairs)
            self.pair_ids = self.pair_ids[limited_pair_indices]   
        
        if limit_style=="random":
            b = np.array([list_ids,list_ids]).T      
            self.pair_ids = np.concatenate((b,self.pair_ids))
            
        
        if limit_style=="uniform":
            self.pair_ids = self._generate_uniform()
            #b = np.array([list_ids,list_ids]).T
            #self.pair_ids = np.concatenate((b,self.pair_ids))
      
        self.shuffle = shuffle 
        self._on_epoch_start()
        print(f"Data created in {time()-start_time} sec")
        
    def closest(self, lst, K):
      
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]      
    
    def _generate_uniform(self):
        if os.path.exists(f"{len(self.list_ids)}_{self.limit_num_pairs}_{self.batch_size}.npy"):
            return np.load(f"{len(self.list_ids)}_{self.limit_num_pairs}_{self.batch_size}.npy")
        else:
            bincount = 8
            bins = {}
            for i in range(bincount): # so we have 8 bins
              bins[i] = []


            aa = d_q(matrix2quaternion(self.y[self.pair_ids[:,0]]), matrix2quaternion(self.y[self.pair_ids[:,1]]))
            hist = np.histogram(aa, bins=bincount)
            cc = np.fmin(np.digitize(aa, hist[1]), bincount)
            #bins[np.around(label, num_dec)].append([idx1, idx2])

            for i in range(bincount):
              bins[i] = list(zip(self.pair_ids[cc==(i+1),0], self.pair_ids[cc==(i+1),1]))

            min_bin_size = len(bins[min(bins.keys(), key=lambda x: len(bins[x]))])
            print("min=", min_bin_size)
            if min_bin_size == 0:
                raise Exception("It haven't yet managed to fill all the bins, please increase limit_num_pairs")


            # cut the top of histogram to make it uniform
            for i in range(bincount): # so we have 8 bins
              bins[i] = np.take(bins[i], np.arange(min_bin_size), axis=0)  
            l = np.array(list(itertools.chain(*list(bins.values()))))

            np.save(f"{len(self.list_ids)}_{self.limit_num_pairs}_{self.batch_size}.npy", l)
            print("total number of data = ", bincount*min_bin_size)
            return l
    
    def __len__(self):
        # Denotes the number of batches per epoch
        if (len(self.pair_ids))%self.batch_size == 0:
            return (len(self.pair_ids))// self.batch_size 
        else:
            return (len(self.pair_ids))// self.batch_size + 1
        
    def __getitem__(self, index):
        # Generate one batch of data
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of ids
        list_ids_batch = np.take(self.pair_ids, indices, axis=0)

        # Generate data
        idx1, idx2 = list_ids_batch[:,0], list_ids_batch[:,1]
        pairs = np.stack((self.X[idx1], self.X[idx2]), axis=1)


        if self.loss_fn == "QCQP_dist" or self.loss_fn == "Q_dist":
          o_a = matrix2quaternion(self.y[idx1])
          o_b = matrix2quaternion(self.y[idx2])
          labels = d_q(o_a, o_b) 
          return (pairs[:, 0], pairs[:, 1]), (labels, o_a, o_b) 
        elif self.loss_fn == "SO3_dist":
          o_a = matrix2quaternion(self.y[idx1])
          o_b = matrix2quaternion(self.y[idx2])
          labels = d_q(o_a, o_b) 
          return (pairs[:, 0], pairs[:, 1]), (labels, self.y[idx1], self.y[idx2])   
        elif self.loss_fn == "S2S2_dist":
          o_1 = SO3_to_s2s2(self.y[idx1]) 
          o_2 = SO3_to_s2s2(self.y[idx2]) 
          o_a = matrix2quaternion(self.y[idx1])
          o_b = matrix2quaternion(self.y[idx2])
          labels = d_q(o_a, o_b) 
          return (pairs[:, 0], pairs[:, 1]), (labels, o_1, o_2) 
        elif self.loss_fn == "QCQP_direct":
          o_a = matrix2quaternion(self.y[idx1])
          o_b = matrix2quaternion(self.y[idx2])
          return (pairs[:, 0], pairs[:, 1]), (o_a, o_b) 
        elif self.loss_fn == "SO3_direct":
          return (pairs[:, 0], pairs[:, 1]), (self.y[idx1], self.y[idx2])
        elif self.loss_fn == "S2S2_direct":
          o_1 = SO3_to_s2s2(self.y[idx1]) 
          o_2 = SO3_to_s2s2(self.y[idx2]) 
          return (pairs[:, 0], pairs[:, 1]), (o_1, o_2)  
        else:
          raise ValueError("This loss not yet implemented")       
        

    def _on_epoch_start(self):
        # Updates indices after each epoch
        self.indices = np.arange(len(self.pair_ids))
        if self.shuffle:
            np.random.shuffle(self.indices)  
        #self.indices = np.concatenate((np.arange(len(X)), self.indices))

class DataGenerator_SO3_s(Sequence):
    """Custom datsaet generator used to create data for training of distance learning algorithm"""
    def __init__(self, X, y1, y2, list_ids, loss_fn, limit_num_pairs=None, limit_style="random", batch_size=256, shuffle=True):
        start_time = time()
        if batch_size > limit_num_pairs:
            raise Exception("Please specify limit_num_pairs that is much higher than batch_size")
        self.X = X  
        self.y1 = y1  
        self.y2 = y2 
        self.limit_num_pairs = limit_num_pairs
        self.list_ids = list_ids
        self.batch_size = batch_size  
        # all the possible combinations of 2 image id pairs
        self.pair_ids = np.array(list(zip(*list(map(lambda x: x.flatten(), np.meshgrid(list_ids, list_ids))))))  # e.g. train_idx
        self.loss_fn = loss_fn
        # Don't use all possible combination of pairs, limit them here
        if self.limit_num_pairs:
            limited_pair_indices = np.random.choice(np.arange(len(self.pair_ids)), size=self.limit_num_pairs)
            self.pair_ids = self.pair_ids[limited_pair_indices]   
        
        if limit_style=="random":
            b = np.array([list_ids,list_ids]).T      
            self.pair_ids = np.concatenate((b,self.pair_ids))
            
        
        if limit_style=="uniform":
            self.pair_ids = self._generate_uniform()
            #b = np.array([list_ids,list_ids]).T
            #self.pair_ids = np.concatenate((b,self.pair_ids))
      
        self.shuffle = shuffle 
        self._on_epoch_start()
        print(f"Data created in {time()-start_time} sec")
        
    def closest(self, lst, K):
      
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]      
    
    def _generate_uniform(self):
        if os.path.exists(f"70s/{len(self.list_ids)}_{self.limit_num_pairs}_{self.batch_size}.npy"):
            return np.load(f"70s/{len(self.list_ids)}_{self.limit_num_pairs}_{self.batch_size}.npy")
        else:
            bincount = 8
            bins = {}
            for i in range(bincount): # so we have 8 bins
              bins[i] = []


            aa = d_q(matrix2quaternion(self.y1[self.pair_ids[:,0]]), matrix2quaternion(self.y1[self.pair_ids[:,1]]))
            hist = np.histogram(aa, bins=bincount)
            cc = np.fmin(np.digitize(aa, hist[1]), bincount)
            #bins[np.around(label, num_dec)].append([idx1, idx2])

            for i in range(bincount):
              bins[i] = list(zip(self.pair_ids[cc==(i+1),0], self.pair_ids[cc==(i+1),1]))

            min_bin_size = len(bins[min(bins.keys(), key=lambda x: len(bins[x]))])
            print("min=", min_bin_size)
            if min_bin_size == 0:
                raise Exception("It haven't yet managed to fill all the bins, please increase limit_num_pairs")


            # cut the top of histogram to make it uniform
            for i in range(bincount): # so we have 8 bins
              bins[i] = np.take(bins[i], np.arange(min_bin_size), axis=0)  
            l = np.array(list(itertools.chain(*list(bins.values()))))

            np.save(f"70s/{len(self.list_ids)}_{self.limit_num_pairs}_{self.batch_size}.npy", l)
            print("total number of data = ", bincount*min_bin_size)
            return l
    
    def __len__(self):
        # Denotes the number of batches per epoch
        if (len(self.pair_ids))%self.batch_size == 0:
            return (len(self.pair_ids))// self.batch_size 
        else:
            return (len(self.pair_ids))// self.batch_size + 1
        
    def __getitem__(self, index):
        # Generate one batch of data
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of ids
        list_ids_batch = np.take(self.pair_ids, indices, axis=0)

        # Generate data
        idx1, idx2 = list_ids_batch[:,0], list_ids_batch[:,1]
        pairs = np.stack((self.X[idx1], self.X[idx2]), axis=1)

        rot1 = self.y1[idx1]
        rot2 = self.y1[idx2]
        shift1 = self.y2[idx1]
        shift2 = self.y2[idx2]
        
        if self.loss_fn == "QCQP_dist":
          o_a = matrix2quaternion(rot1)
          o_b = matrix2quaternion(rot2)
          labels = d_q(o_a, o_b) 
          return (pairs[:, 0], pairs[:, 1]), (labels, o_a, o_b, tf.zeros([self.X.shape[1], self.X.shape[2]]), tf.zeros([self.X.shape[1], self.X.shape[2]]), shift1, shift2) 
        elif self.loss_fn == "SO3_dist":
          o_a = matrix2quaternion(rot1)
          o_b = matrix2quaternion(rot2)
          labels = d_q(o_a, o_b) 
          return (pairs[:, 0], pairs[:, 1]), (labels, rot1, rot2, shift1, shift2)   
        elif self.loss_fn == "S2S2_dist":
          o_1 = SO3_to_s2s2(rot1) 
          o_2 = SO3_to_s2s2(rot2) 
          o_a = matrix2quaternion(rot1)
          o_b = matrix2quaternion(rot2)
          labels = d_q(o_a, o_b) 
          return (pairs[:, 0], pairs[:, 1]), (labels, o_1, o_2, shift1, shift2) 
        elif self.loss_fn == "QCQP_direct":
          o_a = matrix2quaternion(rot1)
          o_b = matrix2quaternion(rot2)
          return (pairs[:, 0], pairs[:, 1]), (o_a, o_b, shift1, shift2) 
        elif self.loss_fn == "SO3_direct":
          return (pairs[:, 0], pairs[:, 1]), (rot1, rot2, shift1, shift2)
        elif self.loss_fn == "S2S2_direct":
          o_1 = SO3_to_s2s2(rot1) 
          o_2 = SO3_to_s2s2(rot2) 
          return (pairs[:, 0], pairs[:, 1]), (o_1, o_2, shift1, shift2)  
        else:
          raise ValueError("This loss not yet implemented")       
        

    def _on_epoch_start(self):
        # Updates indices after each epoch
        self.indices = np.arange(len(self.pair_ids))
        if self.shuffle:
            np.random.shuffle(self.indices)  
        #self.indices = np.concatenate((np.arange(len(X)), self.indices))
    

def convert_Avec_to_A(A_vec):
    """ Convert BxM tensor to BxNxN symmetric matrices """
    """ M = N*(N+1)/2"""
    if len(A_vec.shape) < 2:
        A_vec = tf.expand_dims(A_vec, axis=0)
    
    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 55:
        A_dim = 10
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")
  
    b = tfp.math.fill_triangular(A_vec, upper=True) 
    b = b + tf.transpose(b, perm=[0,2,1]) - tf.linalg.band_part(tf.transpose(b, perm=[0,2,1]), 0, 0)
    return b
    
def convert_Avec_to_A_psd(A_vec):
    """ Convert BxM tensor to BxNxN symmetric matrices """
    """ M = N*(N+1)/2"""
    if len(A_vec.shape) < 2:
        A_vec = tf.expand_dims(A_vec, axis=0)
    
    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 55:
        A_dim = 10
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")
  
    b = tfp.math.fill_triangular(A_vec) 
    b = tf.linalg.matmul(b, b, transpose_b=True)
    return b

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def tfdot(a, b):
  return tf.reduce_sum(a*b, axis=-1, keepdims=True)

def visualise_images(X, n_images, n_columns, randomise=True):
    indices = np.arange(X.shape[0])
    if randomise:
        np.random.shuffle(indices)
    indices = indices[:n_images]
    cmap = plt.cm.Greys_r
    n_rows = np.ceil(n_images / n_columns)
    fig = plt.figure(figsize=(2*n_columns, 2*n_rows))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is 8x8 pixels
    for i, e in enumerate(indices):
        ax = fig.add_subplot(n_rows, n_columns, i + 1, xticks=[], yticks=[])
        ax.imshow(X[e], cmap=cmap, interpolation='nearest')


#### Scheduler

# Define the callback that will update the weight values
class WeightAdjuster(callbacks.Callback):
    def __init__(self, weights: list, epochs: int = None, exp_const: float = 0.3):
        """
        Args:
        weights (list): list of loss weights
        """
        self.alpha = weights[0]
        self.beta1 = weights[1]
        self.beta2 = weights[2]
        self.n_epochs = epochs
        self.exp_const = exp_const

    def on_epoch_end(self, epoch, logs=None):
        # Updated loss weights
        #if epoch <50:
        alpha = (epoch/self.n_epochs)**(self.exp_const)
        K.set_value(self.alpha, 1-alpha)
        K.set_value(self.beta1, (alpha)/2)
        K.set_value(self.beta2, (alpha)/2)


class WeightAdjuster_s(callbacks.Callback):
    def __init__(self, weights: list, epochs: int = None, exp_const: float = 0.3):
        """
        Args:
        weights (list): list of loss weights
        """
        self.alpha = weights[0]
        self.beta1 = weights[1]
        self.beta2 = weights[2]
        self.n_epochs = epochs
        self.exp_const = exp_const

    def on_epoch_end(self, epoch, logs=None):
        # Updated loss weights
        #if epoch <50:
        first = self.n_epochs/2
        if epoch < first:
            alpha = (epoch/first)**(self.exp_const)
            K.set_value(self.alpha, 1-alpha)
            K.set_value(self.beta1, alpha)
            K.set_value(self.beta2, 0)
        else:
            alpha = 1
            K.set_value(self.alpha, 0)
            K.set_value(self.beta1, alpha)
            K.set_value(self.beta2, alpha)

class WeightAdjuster_s2(callbacks.Callback):
    def __init__(self, weights: list, epochs: int = None, exp_const: float = 0.3):
        """
        Args:
        weights (list): list of loss weights
        """
        self.alpha = weights[0]
        self.beta1 = weights[1]
        self.beta2 = weights[2]
        self.beta3 = weights[3]
        self.n_epochs = epochs
        self.exp_const = exp_const

    def on_epoch_end(self, epoch, logs=None):
        # Updated loss weights
        #if epoch <50:
        first = self.n_epochs/2
        if epoch < first:
            alpha = (epoch/first)**(self.exp_const)
            K.set_value(self.alpha, 1-alpha)
            K.set_value(self.beta1, alpha)
            K.set_value(self.beta2, 0)
            K.set_value(self.beta3, 0)
        else:
            alpha = ((epoch-first)/self.n_epochs)
            beta = ((epoch-first)/first)
            K.set_value(self.alpha, 0)
            K.set_value(self.beta1, 1-alpha)
            K.set_value(self.beta2, alpha)
            K.set_value(self.beta3, beta)


class WeightAdjuster_s3(callbacks.Callback):
    def __init__(self, weights: list, epochs: int = None, exp_const: float = 0.3):
        """
        Args:
        weights (list): list of loss weights
        """
        self.alpha = weights[0]
        self.beta1 = weights[1]
        self.beta2 = weights[2]
        self.beta3 = weights[3]
        self.n_epochs = epochs
        self.exp_const = exp_const

    def on_epoch_end(self, epoch, logs=None):
        # Updated loss weights
        #if epoch <50:
        first = self.n_epochs/2
        if epoch < first:
            alpha = (epoch/first)**(self.exp_const)
            K.set_value(self.alpha, 1-alpha)
            K.set_value(self.beta1, alpha)
            K.set_value(self.beta2, 0)
            K.set_value(self.beta3, 0)
        else:
            alpha = ((epoch-first)/self.n_epochs)
            beta = ((epoch-first)/first)
            K.set_value(self.alpha, 0)
            K.set_value(self.beta1, 1-alpha)
            K.set_value(self.beta2, alpha)
            K.set_value(self.beta3, 1-beta)
            
class WeightAdjuster_s4(callbacks.Callback):
    def __init__(self, weights: list, epochs: int = None, exp_const: float = 0.3):
        """
        Args:
        weights (list): list of loss weights
        """
        self.alpha = weights[0]
        self.beta1 = weights[1]
        self.beta2 = weights[2]
        self.beta3 = weights[3]
        self.n_epochs = epochs
        self.exp_const = exp_const

    def on_epoch_end(self, epoch, logs=None):
        # Updated loss weights
        #if epoch <50:
        alpha = (epoch/self.n_epochs)**(self.exp_const)
        beta = (epoch/self.n_epochs)**(self.exp_const)
        K.set_value(self.alpha, 1-alpha)
        K.set_value(self.beta1, alpha)
        K.set_value(self.beta2, beta)
        K.set_value(self.beta3, 1-beta)

class WeightAdjuster_s5(callbacks.Callback):
    def __init__(self, weights: list, epochs: int = None, exp_const: float = 0.3):
        """
        Args:
        weights (list): list of loss weights
        """
        self.alpha = weights[0]
        self.beta1 = weights[1]
        self.beta2 = weights[2]
        self.beta3 = weights[3]
        self.n_epochs = epochs
        self.exp_const = exp_const

    def on_epoch_end(self, epoch, logs=None):
        # Updated loss weights
        #if epoch <50:
        alpha = (epoch/self.n_epochs)**(self.exp_const)
        #beta = (epoch/self.n_epochs)**(self.exp_const)
        K.set_value(self.alpha, 1-alpha)
        K.set_value(self.beta1, alpha)
        K.set_value(self.beta2, 1)
        K.set_value(self.beta3, 0)   

class ExponentialLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_epoch_begin(self, epoch, logs=None):
        self.sum_of_epoch_losses = 0

    def on_batch_end(self, batch, logs=None):
        mean_epoch_loss = logs["loss"]  # the epoch's mean loss so far 
        new_sum_of_epoch_losses = mean_epoch_loss * (batch + 1)
        batch_loss = new_sum_of_epoch_losses - self.sum_of_epoch_losses
        self.sum_of_epoch_losses = new_sum_of_epoch_losses
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(batch_loss)
        K.set_value(self.model.optimizer.learning_rate,
                    self.model.optimizer.learning_rate * self.factor)
        
def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=1e-4,
                       max_rate=1, training_steps=1):
    init_weights = model.get_weights()
    iterations = math.ceil(training_steps / batch_size) * epochs
    print(iterations)
    factor = (max_rate / min_rate) ** (1 / iterations)
    init_lr = K.get_value(model.optimizer.learning_rate)
    K.set_value(model.optimizer.learning_rate, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, validation_data=y, epochs=epochs,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.learning_rate, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses


def plot_lr_vs_loss(rates, losses):
    fig, ax=plt.subplots()
    ax.plot(rates, losses, "b")
    #plt.gca().set_xscale('log')
    max_loss = losses[0] + min(losses)
    #plt.hlines(min(losses), min(rates), max(rates), color="k")
    #plt.axis([min(rates), max(rates), 0, max_loss])
    #plt.xlabel("Learning rate")
    #plt.ylabel("Loss")
    #plt.grid()
    ax.set_xscale("log")
    ax.grid(True, which="both", axis='x', ls="--")
    locmin = mticker.LogLocator(base=10, subs=np.arange(0.1,1,0.1), numticks=10) 
    ax.xaxis.set_minor_locator(locmin)

class OneCycleLr(tf.keras.callbacks.Callback):
    """
    Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.
    [Implementation taken from PyTorch:
    (https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR)]
    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):
    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch
    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.
    Args:
    max_lr (float): Upper learning rate boundaries in the cycle.
    total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
    epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
    steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
    pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
    anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: 'cos'
    cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
    base_momentum (float): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
    max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.95
    div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
    final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
    """

    def __init__(self,
                 max_lr: float,
                 total_steps: int = None,
                 epochs: int = None,
                 steps_per_epoch: int = None,
                 pct_start: float = 0.3,
                 anneal_strategy: str = "cos",
                 cycle_momentum: bool = True,
                 base_momentum: float = 0.85,
                 max_momentum: float = 0.95,
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 ) -> None:

        super().__init__()

        # validate total steps:
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError(
                "You must define either total_steps OR (epochs AND steps_per_epoch)"
            )
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError(
                    "Expected non-negative integer total_steps, but got {}".format(
                        total_steps
                    )
                )
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError(
                    "Expected non-negative integer epochs, but got {}".format(
                        epochs)
                )
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError(
                    "Expected non-negative integer steps_per_epoch, but got {}".format(
                        steps_per_epoch
                    )
                )
            # Compute total steps
            self.total_steps = epochs * steps_per_epoch

        self.step_num = 0
        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(
                "Expected float between 0 and 1 pct_start, but got {}".format(
                    pct_start)
            )

        # Validate anneal_strategy
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                "anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(
                    anneal_strategy
                )
            )
        elif anneal_strategy == "cos":
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == "linear":
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        self.initial_lr = max_lr / div_factor
        self.max_lr = max_lr
        self.min_lr = self.initial_lr / final_div_factor

        # Initial momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            self.m_momentum = max_momentum
            self.momentum = max_momentum
            self.b_momentum = base_momentum

        # Initialize variable to learning_rate & momentum
        self.track_lr = []
        self.track_mom = []

    def _annealing_cos(self, start, end, pct) -> float:
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct) -> float:
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def set_lr_mom(self) -> None:
        """Update the learning rate and momentum"""
        if self.step_num <= self.step_size_up:
            # update learining rate
            computed_lr = self.anneal_func(
                self.initial_lr, self.max_lr, self.step_num / self.step_size_up
            )
            K.set_value(self.model.optimizer.lr, computed_lr)
            # update momentum if cycle_momentum
            if self.cycle_momentum:
                computed_momentum = self.anneal_func(
                    self.m_momentum, self.b_momentum, self.step_num / self.step_size_up
                )
                try:
                    K.set_value(self.model.optimizer.momentum,
                                computed_momentum)
                    #self.model.optimizer.momentum = computed_momentum
                except:
                    #K.set_value(self.model.optimizer.beta_1, computed_momentum)
                    self.model.optimizer.beta_1 = computed_momentum
        else:
            down_step_num = self.step_num - self.step_size_up
            # update learning rate
            computed_lr = self.anneal_func(
                self.max_lr, self.min_lr, down_step_num / self.step_size_down
            )
            K.set_value(self.model.optimizer.lr, computed_lr)
            # update momentum if cycle_momentum
            if self.cycle_momentum:
                computed_momentum = self.anneal_func(
                    self.b_momentum,
                    self.m_momentum,
                    down_step_num / self.step_size_down,
                )
                try:
                    K.set_value(self.model.optimizer.momentum,
                                computed_momentum)
                    #self.model.optimizer.momentum = computed_momentum
                except:
                    #K.set_value(self.model.optimizer.beta_1, computed_momentum)
                    self.model.optimizer.beta_1 = computed_momentum

    def on_train_begin(self, logs=None) -> None:
        # Set initial learning rate & momentum values
        K.set_value(self.model.optimizer.lr, self.initial_lr)
        if self.cycle_momentum:
            try:
                #print(self.model.optimizer.momentum)
                K.set_value(self.model.optimizer.momentum, self.momentum)
                #self.model.optimizer.momentum = self.momentum
            except:
                #print(self.model.optimizer.beta_1)
                #print(self.momentum)
                #K.set_value(self.model.optimizer.beta_1, self.momentum)
                self.model.optimizer.beta_1 = self.momentum

    def on_train_batch_end(self, batch, logs=None) -> None:
        # Grab the current learning rate & momentum
        lr = float(K.get_value(self.model.optimizer.lr))
        try:
            mom = float(K.get_value(self.model.optimizer.momentum))
        except:
            mom = float(K.get_value(self.model.optimizer.beta_1))
        # Append to the list
        self.track_lr.append(lr)
        self.track_mom.append(mom)
        # Update learning rate & momentum
        self.set_lr_mom()
        # increment step_num
        self.step_num += 1
        
    def on_epoch_end(self, epoch, logs=None):     
      print(" lr:%f  mom:%f"%(self.track_lr[-1], self.track_mom[-1]))



    def plot_lrs_moms(self, axes=None) -> None:
        if axes == None:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        else:
            try:
                ax1, ax2 = axes
            except:
                ax1, ax2 = axes[0], axes[1]
        ax1.plot(self.track_lr)
        ax1.set_title("Learning Rate vs Steps")
        ax2.plot(self.track_mom)
        ax2.set_title("Momentum (or beta_1) vs Steps")


def window_mask(D, in_rad, out_rad):
    assert D % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False, dtype=np.float32), 
                         np.linspace(-1, 1, D, endpoint=False, dtype=np.float32))
    r = (x0**2 + x1**2)**.5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r-in_rad)/(out_rad-in_rad)))
    return mask

##### Original

def train_val_test_split(indices, file_name):
    """Train-validation-test split of indices"""
    if not os.path.exists(file_name):
        # the data, split between train and test sets
        train_idx, test_idx = train_test_split(indices, 
                                               test_size=0.33, 
                                               random_state=42)
        train_idx, val_idx= train_test_split(train_idx, 
                                             test_size=0.25, 
                                             random_state=1)

        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        test_idx = sorted(test_idx)

        np.savez(file_name, train_idx, val_idx, test_idx)
    else:
        data = np.load(file_name)
        train_idx, val_idx, test_idx = data["arr_0"], data["arr_1"], data["arr_2"]
        
    return train_idx, val_idx, test_idx

def global_standardization(X):
    """Does not have all the positive piels
    Ref: https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/""" 
    print(f'Image shape: {X[0].shape}')
    print(f'Data Type: {X[0].dtype}')
    X = X.astype('float32')

    print("***")
    ## GLOBAL STANDARDIZATION
    # calculate global mean and standard deviation
    mean, std = X.mean(), X.std()
    print(f'Mean: {mean:.3f} | Std: {std:.3f}')
    print(f'Min:  {X.min():.3f} | Max: {X.max():.3f}')
    # global standardization of pixels
    X = (X - mean) / std
    # confirm it had the desired effect
    mean, std = X.mean(), X.std()
    print(f'Mean: {mean:.3f} | Std: {std:.3f}')
    print(f'Min:  {X.min():.3f} | Max: {X.max():.3f}')
    
    return X

def positive_global_standardization(X):
    """Has all positive pixels
    Ref: https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/"""
    mean, std = X.mean(), X.std()
    print(f"Mean: {mean:.3f} | Std: {std:.3f}")

    # global standardization of pixels
    X = (X - mean) / std

    # clip pixel values to [-1,1]
    X = np.clip(X, -1.0, 1.0)

    # shift from [-1,1] to [0,1] with 0.5 mean
    X = (X + 1.0) / 2.0

    # confirm it had the desired effect
    mean, std = X.mean(), X.std()
    print(f'Mean: {mean:.3f} | Std: {std:.3f}')
    print(f'Min:  {X.min():.3f} | Max: {X.max():.3f}')
    
    return X

def rescale_images(original_images):
    """Rescale the protein images"""
    mobile_net_possible_dims = [128, 160, 192, 224]
    dim_goal = 128
    
    for dim in mobile_net_possible_dims:
        if original_images.shape[1] <= dim:
            dim_goal = dim
            break;
    print(f"Image rescaled from dimension {original_images.shape[1]} to {dim_goal} for MobileNet")
    scale = dim_goal/original_images.shape[1]
    images = np.empty((original_images.shape[0], dim_goal, dim_goal))
    for i, original_image in enumerate(original_images):
        images[i] = rescale(original_image, (scale, scale), multichannel=False)
    return images


def add_gaussian_noise(projections, noise_var):
    """Add Gaussian noise to the protein projection image"""
    noise_sigma   = noise_var**0.5
    nproj,row,col = projections.shape
    gauss_noise   = np.random.normal(0, noise_sigma, (nproj, row, col))
    gauss_noise   = gauss_noise.reshape(nproj, row, col) 
    projections   = projections + gauss_noise
    return projections

def add_triangle_translation(projections, mu, sigma):
    """Add triangular distribution shift to protein center"""
    horizontal_shift = np.random.normal(mu, sigma, len(projections))
    vertical_shift   = np.random.normal(mu, sigma, len(projections))

    for i, (hs, vs) in enumerate(zip(horizontal_shift, vertical_shift)):
        #print(hs, vs)
        tform = transform.EuclideanTransform(
           translation = (hs, vs)
        )
        projections[i] = transform.warp(projections[i], tform.inverse, preserve_range=True)
        #projections[i] = np.roll(projections[i], int(hs), axis=0) # shift 1 place in horizontal axis
        #projections[i] = np.roll(projections[i], int(vs), axis=1) # shift 1 place in vertical axis
    return projections

def projections_preprocessing(projections, angles_true, settings=None):
    """Collection of projection's preprocessing"""
    
    settings_default = dict(
        noise={"variance":0.0},
        shift={"mu": 0,
               "sigma":0},
        channels="gray")
    if settings is None:
        settings = {}
    settings_final = {**settings_default, **settings}
    print(settings_final)
    
    projections = add_gaussian_noise(projections, settings_final["noise"]["variance"])
    if settings_final["shift"]["sigma"] != 0:
        projections = add_triangle_translation(projections, mu=settings_final["shift"]["mu"], sigma=settings_final["shift"]["sigma"])
    
    X, y = np.array(projections, dtype=np.float32), np.array(angles_true, dtype=np.float32)
    X = global_standardization(X)
    
    if settings_final["channels"] == "rgb":
        X = np.stack((X,)*3, axis=-1)
    elif settings_final["channels"] == "gray":
        X = X[:,:,:,np.newaxis]
        
    return X, y