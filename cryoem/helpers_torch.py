
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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

def cosine_distance(vests):
    """Cosine distance between two feature vectors from every projection"""
    x, y = vests
    xy_sum_square = torch.sum(x * y, dim=1, keepdims=True) 
    xx_sum_square = torch.sum(x * x, dim=1, keepdims=True)
    xx_sum_square = torch.maximum(xx_sum_square,  torch.full(xx_sum_square.shape, 1e-08, device=device)) 
    yy_sum_square = torch.sum(y * y, dim=1, keepdims=True)
    yy_sum_square = torch.maximum(yy_sum_square, torch.full(yy_sum_square.shape, 1e-08, device=device)) 
    
    cos_theta = torch.divide(xy_sum_square, torch.sqrt(xx_sum_square)*torch.sqrt(yy_sum_square))
    epsilon=1e-7 
    cos_theta = torch.clamp(cos_theta, 0.0 + epsilon, 1.0 - epsilon) #why zero
    return 2*torch.acos(cos_theta) 

def euler_to_SO3(euler):
    euler = torch.from_numpy(euler).double()
    batch=euler.shape[0]
        
    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1 
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1 
    c2=torch.cos(euler[:,1]).view(batch,1)#batch*1 
    s2=torch.sin(euler[:,1]).view(batch,1)#batch*1 
    c3=torch.cos(euler[:,2]).view(batch,1)#batch*1 
    s3=torch.sin(euler[:,2]).view(batch,1)#batch*1 


    r00 = c1*c2*c3-s1*s3
    r10 = c1*s3+c2*c3*s1
    r20 = -c3*s2
    r01 = -c3*s1-c1*c2*s3
    r11 = c1*c3-c2*s1*s3
    r21 = s2*s3
    r02 = c1*s2
    r12 = s1*s2 
    r22 = c2

    row1=torch.cat((r00, r10, r20), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((r01, r11, r21), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((r02, r12, r22), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix

def R_from_relion(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1
    return R

def R_from_relion_scipy(euler_, degrees=True):
    '''Nx3 array of RELION euler angles to rotation matrix'''
    from scipy.spatial.transform import Rotation as RR
    euler = euler_.copy()
    if euler.shape == (3,):
        euler = euler.reshape(1,3)
    euler[:,0] += 90
    euler[:,2] -= 90
    f = np.ones((3,3))
    f[0,1] = -1
    f[1,0] = -1
    f[1,2] = -1
    f[2,1] = -1
    rot = RR.from_euler('zxz', euler, degrees=degrees).as_matrix()*f
    return rot

def R_to_relion_scipy(rot, degrees=True):
    '''Nx3x3 rotation matrices to RELION euler angles'''
    from scipy.spatial.transform import Rotation as RR
    if rot.shape == (3,3):
        rot = rot.reshape(1,3,3)
    assert len(rot.shape) == 3, "Input must have dim Nx3x3"
    f = np.ones((3,3))
    f[0,1] = -1
    f[1,0] = -1
    f[1,2] = -1
    f[2,1] = -1
    euler = RR.from_matrix(rot*f).as_euler('zxz', degrees=True)
    euler[:,0] -= 90
    euler[:,2] += 90
    euler += 180
    euler %= 360
    euler -= 180
    if not degrees:
        euler *= np.pi/180
    return euler

def s2s2_to_SO3(v1, v2=None):
    '''Normalize 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix.'''
    if v2 is None:
        assert v1.shape[-1] == 6
        v2 = v1[...,3:]
        v1 = v1[...,0:3]
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    e3 = torch.cross(e1, e2)
    return torch.stack([e1, e2, e3], 1)

def SO3_to_s2s2(r):
    '''Map batch of SO(3) matrices to s2s2 representation as first two
    basis vectors, concatenated as Bx6'''
    return r.reshape(*r.shape[:-2],9)[...,:6].contiguous()

def SO3_to_quaternions(r):
    """Map batch of SO(3) matrices to quaternions."""
    batch_dims = r.shape[:-2]
    assert list(r.shape[-2:]) == [3, 3], 'Input must be 3x3 matrices'
    r = r.view(-1, 3, 3)
    n = r.shape[0]

    diags = [r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]]
    denom_pre = torch.stack([
        1 + diags[0] - diags[1] - diags[2],
        1 - diags[0] + diags[1] - diags[2],
        1 - diags[0] - diags[1] + diags[2],
        1 + diags[0] + diags[1] + diags[2]
    ], 1)
    denom = 0.5 * torch.sqrt(1E-6 + torch.abs(denom_pre))

    case0 = torch.stack([
        denom[:, 0],
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 0]),
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 0]),
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 0])
    ], 1)
    case1 = torch.stack([
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 1]),
        denom[:, 1],
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 1]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 1])
    ], 1)
    case2 = torch.stack([
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 2]),
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 2]),
        denom[:, 2],
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 2])
    ], 1)
    case3 = torch.stack([
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 3]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 3]),
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 3]),
        denom[:, 3]
    ], 1)

    cases = torch.stack([case0, case1, case2, case3], 1)

    quaternions = cases[torch.arange(n, dtype=torch.long),
                        torch.argmax(denom.detach(), 1)]
    return quaternions.view(*batch_dims, 4)


def quaternions_to_SO3(q):
    '''Normalizes q and maps to group matrix.'''
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack([
        r*r - i*i - j*j + k*k, 2*(r*i + j*k), 2*(r*j - i*k),
        2*(r*i - j*k), -r*r + i*i - j*j + k*k, 2*(i*j + r*k),
        2*(r*j + i*k), 2*(i*j - r*k), -r*r - i*i + j*j + k*k
        ], -1).view(*q.shape[:-1], 3, 3)

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

def add_triangle_translation(projections, left_limit, peak_limit, right_limit):
    """Add triangular distribution shift to protein center"""
    horizontal_shift = np.random.triangular(left_limit, peak_limit, right_limit, len(projections))
    vertical_shift   = np.random.triangular(left_limit, peak_limit, right_limit, len(projections))
    for i, (hs, vs) in enumerate(zip(horizontal_shift, vertical_shift)):
        projections[i] = np.roll(projections[i], int(hs), axis=0) # shift 1 place in horizontal axis
        projections[i] = np.roll(projections[i], int(vs), axis=1) # shift 1 place in vertical axis
    return projections

def projections_preprocessing(projections, angles_true, settings=None):
    """Collection of projection's preprocessing"""
    
    settings_default = dict(
        noise={"variance":0.0},
        shift={"left_limit":-0.01,
               "peak_limit":0,
               "right_limit":0.01},
        channels="gray")
    if settings is None:
        settings = {}
    settings_final = {**settings_default, **settings}
    
    projections = add_gaussian_noise(projections, settings_final["noise"]["variance"])
    projections = add_triangle_translation(projections, left_limit=settings_final["shift"]["left_limit"], peak_limit=settings_final["shift"]["peak_limit"], right_limit=settings_final["shift"]["right_limit"])
    
    X, y = np.array(projections, dtype=np.float32), np.array(angles_true, dtype=np.float32)
    X = global_standardization(X)
    
    if settings_final["channels"] == "rgb":
        X = np.stack((X,)*3, axis=-1)
    elif settings_final["channels"] == "gray":
        X = X[:,:,:,np.newaxis]
        
    return X, y