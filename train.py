import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from src.dataset import get_dataloader
from src.utils import bool_flag, to_none, str2dic_all, str2list, intorstr
from src.trainer import train
from src.functions import get_function

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = ArgumentParser(description="Grokking for MLP")

    # Main parameters
    parser.add_argument("--exp_id", type=str, help="Experiment id")
    parser.add_argument("--log_dir", type=str, help="Experiment dump path") # trainer

    # Model
    parser.add_argument("--hidden_dim", type=int, help="") 
    parser.add_argument("--n_layers", type=int, help="") 
    parser.add_argument("--dropout", type=float, default=0.0, help="")

    # Dataset
        # Intrinsic Dimension Estimation
    parser.add_argument("--func_params", type=str2dic_all, help="""
        * eg : "f_name=str(Styblinski-Tang)"
        * eg : "f_name=str(Styblinski-Tang),min_x=int(-5),max_x=int(5),min_y=int(-5),max_y=int(5),step_x=int(0.25),step_y=int(0.25)"
        * f_name : Styblinski-Tang, Ackley, Beale, Booth, Bukin, McCormick, Rosenbrock, Sum, Prod
        """)
    parser.add_argument("--ndim", type=int, default=2, help="dimension of the input space") 
    parser.add_argument("--num_samples", type=int, help="Dataset size")
    parser.add_argument("--normalize", type=bool_flag, default=False, help="Normalize the data or not") 
    parser.add_argument("--noise_params", type=str2dic_all, default="", help="""
        See https://numpy.org/doc/1.16/reference/routines.random.html#distributions
        eg :
            * distribution=str(normal),loc=float(0.0),scale=float(1.0)
            * distribution=str(uniform),low=float(0.0),high=float(1.0)
            * distribution=str(beta),a=float(1.0),b=float(1.0)
            * distribution=str(binomial),n=int(10),p=float(0.5)
            * distribution=str(chisquare),df=int(2)
            * distribution=str(exponential),scale=float(1.0)
            * distribution=str(gamma),shape=float(2.0),scale=float(1.0)
            * distribution=str(logistic),loc=float(0.0),scale=float(1.0)
    """)
    parser.add_argument("--train_pct", type=int, help="Training data fraction, \in ]0, 100[") 
    parser.add_argument("--batch_size", type=int, help="Batch size") 

    parser.add_argument("--limit_train_batches", type=float, default=1., help="limit batches for training data")
    parser.add_argument("--limit_val_batches", type=float, default=1., help="limit batches for validation data")
    parser.add_argument("--limit_test_batches", type=float, default=1., help="limit batches for test data")
    
    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.001,beta1=0.9,beta2=0.99,eps=0.00000001", help="""
                - optimizer parameters : adam_inverse_sqrt,beta1=0.9,beta2=0.99,eps=0.00000001 ...
                - classes : CustomAdam (custom_adam), Adam (adam), AdamInverseSqrtWithWarmup (adam_inverse_sqrt), AdamCosineWithWarmup (adam_cosine),
                            Adadelta (adadelta), Adagrad (adagrad), Adamax (adamax), ASGD (asgd), SGD (sgd), RMSprop (rmsprop), Rprop (rprop)
                - See https://pytorch.org/docs/stable/optim.html for optimizers parameters
    """) # optim : training
    parser.add_argument("--alpha_beta", type=str2dic_all, default="", help="""
            * eg : "alpha=float(1.0),beta=float(1.0)"
            * loss = alpha*loss_y + beta*loss_dydx
    """)

    # LR Scheduler
    parser.add_argument("--lr_scheduler", type=to_none, default="", help="""
                eg : 
                    - reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss
                    - constant_lr,factor=0.33,total_iters=5,last_epoch=-1
                Using a scheduler is optional but can be helpful.
                The scheduler reduces the LR if the validation performance hasn't improved for the last N (patience) epochs.
                class : reduce_lr_on_plateau, constant_lr, linear_lr, cosine_annealing_lr, exponential_lr, lambda_lr, 
                multiplicative_lr, step_lr, multi_step_lr, cyclic_lr, one_cycle_lr, cosine_annealing_warm_restarts            
    """)

    # Training
    parser.add_argument("--max_epochs", type=int, help="Maximun number of epoch")

    parser.add_argument("--validation_metrics", type=str, default="val_loss_y", help="Validation metrics : val_acc, val_loss, val_ppl ...") # trainer

    parser.add_argument("--checkpoint_path", type=to_none, default=None, help="Load pretrained model from checkpoint and continued training")
    parser.add_argument("--model_name", type=str, default="", 
                        help="Load a model from root directory (`log_dir/exp_id`), eg : epoch=88-val_loss=13.6392.ckpt")
    parser.add_argument("--every_n_epochs", type=int, 
                        help="Frequency at which the images of the representations and the last layer weights are sent to wandb")
    parser.add_argument("--every_n_epochs_show", type=int, 
                        help="Frequency at which the images of the representations and the last layer weights are displayed. \
                              It must be a multiple of `every_n_epochs`")
    parser.add_argument("--early_stopping_patience", type=int, default=1e9, 
                        help="Early stopping patience : If the model does not converge during these numbers of steps, stop the training") 
    parser.add_argument("--save_top_k", type=int, default=-1, 
                        help="The best `save_top_k` models according to the quantity monitored will be saved.\
                              If save_top_k == 0, no models are saved. if save_top_k == -1, all models are saved.")

    # Wandb
    parser.add_argument("--use_wandb", type=bool_flag, default=False, help="") 
    parser.add_argument("--wandb_entity", type=str, help="") 
    parser.add_argument("--wandb_project", type=str, help="")
    parser.add_argument("--group_name", type=to_none, default=None, help="")
    parser.add_argument("--group_vars", type=str2list, help="") 

    # Intrinsic Dimension Estimation
    parser.add_argument("--ID_params", type=str2dic_all, default="", help="""
        * TWO-NN : "method=str(twonn)"
        * Maximum Likelihood Estimation appoach : "method=str(mle),k=int(2),averaging_of_inverses=bool(True)"
        """)

    # Devices & Seed
    parser.add_argument("--accelerator", type=str, default="auto", help="accelerator types : cpu, gpu, tpu, ipu, auto") 
    parser.add_argument("--devices", type=intorstr, default="auto", help="number of cpu processes, of gpu/tpu cores ...")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproductibility")

    # Early_stopping (stop training after grokking)
    parser.add_argument("--early_stopping_grokking", type=str2dic_all, default="", help="""
        * eg. : "patience=int(1000),metric=str(val_accuracy),metric_threshold=float(90.0)"
        * Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold`"
        """)

    return parser

def main(params) :

    pl.seed_everything(params.random_seed, workers=True)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Dataset
    func_params = get_function(params.func_params)
    params.func_params = func_params
    setattr(params, "f_name", func_params.f_name)
    train_loader, val_loader, _, data_infos, data_config = get_dataloader(
        func_params.callable_function, params.ndim, func_params.min_x, func_params.max_x, params.num_samples, params.train_pct, 
        deriv_function = getattr(func_params, "callable_function_deriv", None), noise_params=params.noise_params, 
        batch_size=params.batch_size, 
        num_workers=2,
        #num_workers = os.cpu_count(),
        normalize = params.normalize
    )
    setattr(params, "data_infos", data_infos)
    setattr(params, "data_config", data_config)

    # Train
    model, result = train(params, train_loader, val_loader)
    
    print("\n********")
    print(result)
    print("********\n")

if __name__ == "__main__":
    # generate parser / parse parameters
    params = get_parser().parse_args()
    # run experiment
    main(params)

