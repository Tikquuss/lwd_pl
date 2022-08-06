import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 

###### TOCORRECT
from .utils_derivs import gradient

def plt2arr(fig, draw=True):
    # https://gist.github.com/orm011/c674f55566fd83609ba6c41699acb728
    if draw:
        fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w,h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
    return rgba_arr

def plotFunction(
    function = None, model = None, 
    min_x = -5, max_x = 5, step_x = 0.25, 
    min_y = -5, max_y = 5, step_y = 0.25,        
    title = None, figsize=(5,5), save_to = None, show = False
) :
    """plot the given function/model"""
    sns.set_theme(style="white", palette=None)
    
    assert function or model
    x = np.arange(start = min_x, stop = max_x, step = step_x, dtype = np.float64)
    y = np.arange(start = min_y, stop = max_y, step = step_y, dtype = np.float64)
    x, y = np.meshgrid(x, y)
    X = []
    for i in range(len(x)):
        for j in range(len(x[0])):
            X.append([x[i][j], y[i][j]])
    X = np.array(X)
    z = []
    if model :
        model.eval()
        X = torch.FloatTensor(X)
        ds = TensorDataset(X, torch.ones_like(X)) 
        dsloader = DataLoader(ds, batch_size = 1, )
        with torch.no_grad():
            for batch, _ in dsloader :
                x_ = batch[0].to(model.device)
                #x_.requires_grad_(True)
                y_pred, _ = model(x_)
                z.append(y_pred.detach().cpu().squeeze().numpy())
    else :    
        for k in range(len(X)):
            z.append(function(X[k]))

    z = np.array(z).reshape((len(x), len(x[0])))

    plt.figure(figsize=figsize)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5)
   
    if title is not None : plt.title(title, y=-0.15)
    if save_to is not None : plt.savefig(save_to, bbox_inches='tight')
    sns.set()

    if show : plt.show()
    else : plt.close()

    return fig, plt2arr(fig, draw=True)

def plotGrad(
    deriv_function = None, model = None, 
    min_x = -5, max_x = 5, step_x = 0.25, 
    min_y = -5, max_y = 5, step_y = 0.25,
    title = None, figsize=(5,5), save_to = None, show = False
):
    """plot the gradient of the function/model"""
    sns.set_theme(style="white", palette=None)
    assert deriv_function or model
    x = np.arange(start = min_x, stop = max_x, step = step_x, dtype = np.float64)
    y = np.arange(start = min_y, stop = max_y, step = step_y, dtype = np.float64)
    x, y = np.meshgrid(x, y)
    X = []
    for i in range(len(x)):
        for j in range(len(x[0])):
            X.append([x[i][j], y[i][j]])
    X = np.array(X)
    z = []
    if model :
        model.train()
        X = torch.FloatTensor(X)
        dsloader = DataLoader(TensorDataset(X, torch.ones_like(X)), batch_size = 1)
        for batch, _ in dsloader :
            x_ = batch[0].to(model.device)
            x_.requires_grad_(True)
            y_pred, _ = model(x_)
            dydx_pred = gradient(y_pred, x_)
            z.append(dydx_pred.detach().cpu().numpy())
    else :
        grad1 = deriv_function(index = 0)
        grad2 = deriv_function(index = 1)
        for k in range(len(X)):
            z.append([grad1(X[k]), grad2(X[k])])
    z = np.array(z)
    z = np.array(z).reshape((len(x), len(x[0]), 2))

    plt.figure(figsize=figsize)
    fig = plt.figure()
    plt.title(title)
    dz = plt.quiver(x, y, z[:, :, 0], z[:, :, 1])

    if title is not None : plt.title(title, y=-0.15)
    if save_to is not None : plt.savefig(save_to, bbox_inches='tight')
    sns.set()

    if show : plt.show()
    else : plt.close()

    return fig, plt2arr(fig, draw=True)

if __name__ == "__main__":
    from functions import get_function
    from utils import AttrDict
    from modeling import Model

    f_name = "Styblinski-Tang"
    f_name = "Sum"
    params = get_function(f_name)

    plotFunction(
        function = params.callable_function, model = None, 
        min_x = params.min_x, max_x = params.max_x, step_x = 0.25, 
        min_y = params.min_y, max_y = params.max_y, step_y = 0.25,
        title = f"{f_name} Function", figsize=(5,5), save_to = f'./../{f_name}.png', show = False
        )

    plotGrad(
        deriv_function = params.callable_function_deriv, model = None, 
        min_x = params.min_x, max_x = params.max_x, step_x = 0.25, 
        min_y = params.min_y, max_y = params.max_y, step_y = 0.25,
        title = f"{f_name} Function deriv", figsize=(5,5), save_to = f'./../{f_name}_deriv.png', show = False
    )

    ##
    ndim = 2
    model_params = AttrDict({
        "ndim" : ndim,
        "hidden_dim" : 105,  
        "n_layers" : 1,
        "dropout" : 0.0,

        "use_wandb" : False,

        "early_stopping_grokking" : {"patience" : 1000, "metric" : "val_loss_y", "metric_threshold" : 90.0},
        "optimizer" : "adam,lr=0.001,weight_decay=0.0,beta1=0.9,beta2=0.99,eps=0.00000001",
        "lr_scheduler" : None,
        #"lr_scheduler" : "reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss_y",
        
        "ID_params" : {},
        #"ID_params": {"method" : "twonn"},

        "alpha_beta" : {"alpha" : 1.0, "beta" : 1.0}, 

        "max_epochs" : 100,

        "data_config" : AttrDict({
            "x_mean" : 0.0, "x_std" : 1.0, "y_mean" : 0.0, "y_std" : 1.0, "lambda_j" : 1.0, "n" : -1,
            "get_alpha_beta" : lambda lam : (1.0, 1.0)
        }),
        "normalize" : False
    })

    model = Model(model_params)

    plotFunction(
        function = None, model = model, 
        min_x = params.min_x, max_x = params.max_x, step_x = 0.25, 
        min_y = params.min_y, max_y = params.max_y, step_y = 0.25,
        title = f"{f_name} Function pred", figsize=(5,5), save_to = f'./../{f_name}_model.png', show = False
        )

    plotGrad(
        deriv_function = None, model = model, 
        min_x = params.min_x, max_x = params.max_x, step_x = 0.25, 
        min_y = params.min_y, max_y = params.max_y, step_y = 0.25,
        title = f"{f_name} Function deriv pred", figsize=(5,5), save_to = f'./../{f_name}_deriv_model.png', show = False
    )