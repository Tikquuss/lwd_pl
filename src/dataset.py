import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

import numpy as np
import random

from .utils import AttrDict

def get_noise_function(noise_params) :
    """https://numpy.org/doc/1.16/reference/routines.random.html#distributions"""
    if noise_params is None : noise_params = {}
    distribution = noise_params.pop("distribution", None)
    if distribution is None :
        return lambda shape : np.zeros(shape)
    else :
        return lambda shape : getattr(np.random, distribution)(**noise_params, size=shape)

epsilon = 1.0e-08
def normalize_data(x_raw, y_raw, dydx_raw=None, crop=None):
    """Data normalization"""
    # crop dataset
    m = crop if crop is not None else x_raw.shape[0]
    x_cropped = x_raw[:m]
    y_cropped = y_raw[:m]
    dydx_cropped = dydx_raw[:m] if dydx_raw is not None else None
    
    # normalize dataset
    x_mean = x_cropped.mean(axis=0)
    x_std = x_cropped.std(axis=0) + epsilon
    x = (x_cropped - x_mean) / x_std
    y_mean = y_cropped.mean(axis=0)
    y_std = y_cropped.std(axis=0) + epsilon
    y = (y_cropped-y_mean) / y_std
    
    # normalize derivatives too
    if dydx_cropped is not None:
        dydx = (x_std / y_std) * dydx_cropped  
        # weights of derivatives in cost function = (quad) mean size
        lambda_j = 1.0 / np.sqrt((dydx ** 2).mean(axis=0)).reshape(1, -1)
    else:
        dydx = None
        lambda_j = 1.0 
    
    return (x_mean, x_std, x), (y_mean, y_std, y), (dydx, lambda_j)


def genData(
    function, ndim, min_x, max_x, num_samples, deriv_function = None, noise_params = {}, normalize = False, lam = 1.0
):
    """takes a :
        * function : f(x : array), return y 
        * deriv_function (the function derivative) : f'(i: int), takes i as parameter and returns another function that takes x and returns df(x)/dx[i] = dy/dx[i].
        * ndim : dimension of x
        * min_x, max_x : the boundaries of the domain in which the points will be generated 
        * num_samples : the number of examples (n) to be generated
        * noise = {"get_noise_function" : gaussian_noise, "noise_params" : {'loc' : 0.0, 'scale' : 1.0} } for example
    and returns (xi, yi, [dydx[j], j=1...dim_x]), i = 1….n"""
    
    if deriv_function is None :
        def deriv_function_(i : int) : return lambda x : x
    else :
        deriv_function_ = deriv_function

    X, Y, dYdX = [], [], []
    for n in range(num_samples):
        x = np.array([random.uniform(min_x, max_x) for i in range(ndim)])
        y = function(x)
        dydx = np.array([deriv_function_(i)(x) for i in range(ndim)])
        X.append(x)
        Y.append(y)
        dYdX.append(dydx)
        
    Y = np.array(Y)
    Y = Y + get_noise_function(noise_params)(Y.shape)

    X = np.array(X)
    dYdX = np.array(dYdX)

    if normalize :
        _, n = X.shape
        alpha = 1.0 / (1.0 + lam * n)

        (x_mean, x_std, X), (y_mean, y_std, Y), (dYdX, lambda_j) = normalize_data(x_raw = X, y_raw = Y, dydx_raw=None if deriv_function is None else dYdX, crop=None)
        data_config = {
            "x_mean" : torch.tensor(x_mean), "x_std" : torch.tensor(x_std), 
            "y_mean" : torch.tensor(y_mean), "y_std" : torch.tensor(y_std), 
            "n" : n, "lambda_j" : torch.tensor(lambda_j),
            "alpha" : torch.tensor(alpha), "beta" : torch.tensor(1.0 - alpha)
        }
    else :
        _, n = np.array(X).shape
        data_config = {
            "x_mean" : torch.tensor(0.0), "x_std" : torch.tensor(1.0), 
            "y_mean" : torch.tensor(0.0), "y_std" : torch.tensor(1.0), 
            "n" : n, "lambda_j" : torch.tensor(1.0), 
            "alpha" : torch.tensor(1.0), "beta" : torch.tensor(1.0)
        }

    return  X, Y, dYdX, AttrDict(data_config)

# We define a data constructor that we can use for various purposes later.
def get_dataloader(
    function, ndim, min_x, max_x, num_samples, train_pct, deriv_function = None, noise_params = {}, 
    batch_size=256, num_workers=4, normalize = False, lam = 1.0
):
    X, Y, dYdX, data_config = genData(
        function, ndim, min_x, max_x, num_samples, deriv_function = deriv_function, noise_params = noise_params, normalize = normalize,
        lam = lam
    )
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    if deriv_function is None : dataset = TensorDataset(X, Y)
    else : dataset = TensorDataset(X, Y, torch.from_numpy(dYdX).float())
    
    n = len(dataset)
    train_size = train_pct * n // 100
    val_size = n - train_size

    #print(f"train_size, val_size : {train_size}, {val_size}")

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=min(batch_size, train_size), shuffle=True, drop_last=False, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=min(batch_size, val_size), shuffle=False, drop_last=False, num_workers=num_workers)

    dataloader = DataLoader(dataset, batch_size=min(batch_size, n), shuffle=False, drop_last=False, num_workers=num_workers)

    data_infos = {
        "train_size":train_size, "val_size":val_size,
        "train_batch_size" : min(batch_size, train_size), "val_batch_size" : min(batch_size, val_size), 
        "train_n_batchs":len(train_loader), "val_n_batchs":len(val_loader)
    }
    print(data_infos, "\n")

    return train_loader, val_loader, dataloader, data_infos, data_config

if __name__ == "__main__":
    function = lambda x : x.sum()

    #deriv_function = None
    def deriv_function(i : int) :  return lambda x : 1

    ndim = 2
    min_x, max_x = 0, 5
    num_samples = 10

    train_pct = 80

    #noise_params = None
    noise_params = {"distribution" : "normal", "loc" : 0.0, "scale" : 1.0}
    
    train_loader, val_loader, dataloader, data_infos, data_config = get_dataloader(
        function, ndim, min_x, max_x, num_samples, train_pct, deriv_function = deriv_function, 
        noise_params = noise_params, batch_size=256, num_workers=4,
        normalize = False,  lam = 1.0
    )

    print(data_infos)
    
    if deriv_function is None :
        x, y = next(iter(dataloader))
        dxdy = None
    else :
        x, y, dxdy = next(iter(dataloader))

    print(x, y, dxdy)
