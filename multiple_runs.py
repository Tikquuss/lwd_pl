import itertools
import numpy as np
import matplotlib.pyplot as plt
import wandb

from src.utils import AttrDict, GROUP_VARS
from src.dataset import get_dataloader
from src.utils import get_group_name
from src.trainer import train
from src.functions import get_function

def plot_results(
    params, model_dict, hparms_1, hparms_2, s1, s2,
    title = None, save_to = None, show = True
):
    """
    2D plot of train&val acc&loss as a function of two parameters use for phase diagram
    """
    fig = plt.figure()
    fig.suptitle("Grokking")

    figsize=(2*8, 6)
    plt.gcf().set_size_inches(figsize)

    i = 1
    for metric in ["loss_y", "loss_dydx"]  :
        ax = fig.add_subplot(1, 2, i, projection='3d')
        i += 1 
        xs, ys, zs = [], [], []
        for split, (m, zlow, zhigh) in zip(["val", "train"], [('o', -50, -25), ('^', -30, -5)]) :
            for a, b in itertools.product(hparms_1, hparms_2) :
                k = f"{s1}={a},{s2}={b}"
                if k in model_dict.keys():
                    xs.append(a)
                    ys.append(b)
                    #print(k, f"{split}_{metric}", model_dict[k]["result"][split][f"{split}_{metric}"])
                    zs.append(model_dict[k]["result"][split][f"{split}_{metric}"])

            ax.scatter(xs, ys, zs, marker=m, label = split)

        ax.set_xlabel(s1)
        ax.set_ylabel(s2)
        ax.set_zlabel(metric)
        ax.set_title(metric, fontsize=14)
        ax.legend()

    if title is not None : plt.title(title, y=-0.15)
    if save_to is not None : plt.savefig(save_to, bbox_inches='tight')

    if show : plt.show()
    else : plt.close()

if __name__ == "__main__":

    f_name="Styblinski-Tang"
    train_pct=80
    weight_decay=0.0
    lr=0.001
    dropout=0.0
    opt="adam"
    ndim=2
    group_name=f"{f_name}:ndim={ndim}-tdf={train_pct}-wd={weight_decay}-lr={lr}-d={dropout}-opt={opt}"
    random_seed=0
    log_dir="../log_files"
    alpha=1.0
    beta=1.0
    params = AttrDict({
        ### Main parameters
        "exp_id" : f"{group_name}",
        "log_dir" : f"{log_dir}/{random_seed}",

        ### Model
        "hidden_dim" : 512,  
        "n_layers" : 1,
        "dropout" : dropout,

        ### Dataset
        "func_params": AttrDict({"f_name" : f_name}),
        #"func_params" : AttrDict({"f_name" : f_name, "min_x": -5, "max_x" : 5, "min_y" : -5, "max_y" : 5, "step_x" : 0.25, "step_y" : 0.25}),
        "ndim" : ndim,
        "num_samples" : 1000,
        "noise_params" : None,
        #"noise_params" : {"distribution" : "normal", "loc" : 0.0, "scale" : 1.0},
        "normalize" : False,
        "train_pct" : train_pct,
        "batch_size" : 512,
       
        ### Optimizer
        "optimizer" : f"{opt},lr={lr},weight_decay={weight_decay},beta1=0.9,beta2=0.99,eps=0.00000001",
        #"alpha_beta" : None, 
        "alpha_beta" : {"alpha" : alpha, "beta" : alpha}, 
 
        ### LR Scheduler
        "lr_scheduler" : None,
        #"lr_scheduler" : "reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss_y",
        
        ### Training
        "max_epochs" : 10000, 
        "validation_metrics" : "val_loss_y",
        "checkpoint_path" : None, 
        "model_name": "None", 
        "every_n_epochs":100, 
        "every_n_epochs_show":200, 
        "early_stopping_patience":1e9, 
        "save_top_k":-1,

        # Wandb 
        "use_wandb" : False,
        "wandb_entity" : "grokking_ppsp",
        "wandb_project" : f"learning_with_derivative",
        "group_name" : group_name,

        "group_vars" : None,

        ### Intrinsic Dimension Estimation
        #"ID_params" : {},
        #"ID_params": {"method" : "mle", "k":2},
        "ID_params": {"method" : "twonn"},
        
        ### Devices & Seed
        "accelerator" : "auto",
        "devices" : "auto",
        "random_seed": random_seed,

        ### Early_stopping (for grokking) : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold` 
        #"early_stopping_grokking" : None,
        "early_stopping_grokking" : "patience=int(1000),metric=str(val_loss_y),metric_threshold=float(0.0)"
    })
    if params.alpha_beta is not None : 
        params.group_name=f"{params.group_name}-alpha={alpha}-beta={beta}"
        params.exp_id=params.group_name
    params["weight_decay"] = weight_decay
    params["f_name"] = f_name
    func_params = get_function(params.func_params)
    params.func_params = func_params
    train_loader, val_loader, dataloader, data_infos, data_config = get_dataloader(
        func_params.callable_function, params.ndim, func_params.min_x, func_params.max_x, params.num_samples, params.train_pct, 
        deriv_function = getattr(func_params, "callable_function_deriv", None), noise_params=params.noise_params, 
        batch_size=params.batch_size, num_workers=2, normalize=params.normalize
    )
    params["data_infos"] = data_infos
    params["data_config"] = data_config

    ######## Example : phase diagram with representation_lr and weight_decay

    lrs = [1e-3]
    #lrs = [1e-2, 1e-3, 1e-4, 1e-5] 
    #lrs = np.linspace(start=1e-1, stop=1e-5, num=10)

    weight_decays = [0.0]
    #weight_decays = list(range(20))
    #weight_decays =  np.linspace(start=0, stop=20, num=21)

    print(lrs, weight_decays)

    model_dict = {}
    i = 0
    for a, b in itertools.product(lrs, weight_decays) :

        params["lr"] = a 
        params["optimizer"] = params["optimizer"].replace(f"weight_decay={weight_decay}", f"weight_decay={b}")
    
        name = f"lr={a},weight_decay={b}"
        params.exp_id = name
        
        #group_vars = GROUP_VARS + ["lr", s]
        group_vars = ["lr", "weight_decay"]
        group_vars = list(set(group_vars))
        params["group_name"] = get_group_name(params, group_vars = None)
        
        print("*"*10, i, name, "*"*10)
        i+=1

        model, result = train(params, train_loader, val_loader)
        
        model_dict[name] = {"model": model, "result": result}

    ########

    print(model_dict.keys())
    val_loss = [model_dict[k]["result"]["val"]["val_loss_y"] for k in model_dict]
    val_loss_dydx = [model_dict[k]["result"]["val"]["val_loss_dydx"] for k in model_dict]
    print(val_loss, val_loss_dydx)

    plot_results(params, model_dict, 
        hparms_1 = lrs, hparms_2 = weight_decays,
        s1 = 'lr', s2 = "weight_decay",
        title = None, save_to = f"{params.log_dir}/result_multiple_run.png", show = True
    )

    ########

    # for k in model_dict :
    #     print("*"*10, k, "*"*10)
    #     model = model_dict[k]["model"]
    #     # TODO