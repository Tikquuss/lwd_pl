# torch & torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from copy import copy

import torchvision
import torchvision.transforms.functional as F

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

import wandb

import cv2
import os
import re 

from .visualize import plotFunction, plotGrad
from .modeling import Model
from .utils import get_group_name, init_wandb


# Thanks https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py
plt.rcParams["savefig.bbox"] = 'tight'
def do_show(imgs, figsize=(14,14), show = True, title = None, save_to = None):
    sns.set_theme(style="white", palette=None)
    if not isinstance(imgs, list): imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if title is not None : plt.title(title, y=-0.13)
    if save_to is not None : plt.savefig(save_to)
    sns.set()

    if show : plt.show()
    else : plt.close()


class GenerateCallback(pl.Callback):
    """Use to plot the learned input embeddings at different training stages"""
    
    def __init__(self, log_dir, func_params, every_n_epochs=1, every_n_epochs_show=10, format="png"):
        super().__init__()
        self.func_params = func_params
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs_show = every_n_epochs_show
        self.log_dir = os.path.join(log_dir, "images")
        self.log_dir_pred = os.path.join(self.log_dir, "pred")
        self.log_dir_pred_grad = os.path.join(self.log_dir, "pred_grad")

        for dir in [self.log_dir, self.log_dir_pred, self.log_dir_pred_grad] : 
            os.makedirs(dir, exist_ok = True)
        
        self.format = format

        self.show = False
        #self.show = not pl_module.use_wandb
        
        params = self.func_params
        self.params = params
        
        _, self.image1 = plotFunction(
            function = params.callable_function, model = None, 
            min_x = params.min_x, max_x = params.max_x, step_x = params.step_x, 
            min_y = params.min_y, max_y = params.max_y, step_y = params.step_y,
            title = f"{params.f_name} Function", figsize=(5,5), 
            save_to = os.path.join(self.log_dir, f"{params.f_name}.{self.format}"), 
            show = False
        )

        _, self.image2 = plotGrad(
            deriv_function = params.callable_function_deriv, model = None, 
            min_x = params.min_x, max_x = params.max_x, step_x = params.step_x, 
            min_y = params.min_y, max_y = params.max_y, step_y = params.step_y,
            title = f"{params.f_name} Function gradient", figsize=(5,5), 
            save_to = os.path.join(self.log_dir, f"{params.f_name}_deriv.{self.format}"), 
            show = False
        )

    def on_epoch_end(self, trainer, pl_module):
    #def on_train_epoch_end(self, trainer, pl_module) :
    #def on_validation_epoch_end(self, trainer, pl_module) :
        current_epoch = trainer.current_epoch
        if current_epoch % self.every_n_epochs == 0 :
            params = self.params
            _, image1 = plotFunction(
                function = None, model = pl_module, 
                min_x = params.min_x, max_x = params.max_x, step_x = params.step_x, 
                min_y = params.min_y, max_y = params.max_y, step_y = params.step_y,
                title = f"{params.f_name} Function pred", figsize=(5,5), 
                save_to = os.path.join(self.log_dir_pred, f"{current_epoch}.{self.format}"), 
                show = False
                )

            _, image2 = plotGrad(
                deriv_function = None, model = pl_module, 
                min_x = params.min_x, max_x = params.max_x, step_x = params.step_x, 
                min_y = params.min_y, max_y = params.max_y, step_y = params.step_y,
                title = f"{params.f_name} Function gradient pred", figsize=(5,5), 
                save_to = os.path.join(self.log_dir_pred_grad, f"{current_epoch}.{self.format}"), 
                show = False
            )

            H, W, _ = image1.shape
            images_ = [image1, image2, self.image1, self.image2, ] # H x W x C
            if True :
                images = [img.transpose(2, 0, 1) for img in images_] # C x H x W
                imgs = [torch.from_numpy(img) for img in images] # B x C x H x W, B = 1
            else :
                imgs = [torch.from_numpy(img).transpose(2, 0).transpose(1, 2) for img in images_] # B x C x H x W, B = 1

            grid = torchvision.utils.make_grid(imgs, nrow=4)
            trainer.logger.experiment.add_image("representation", grid, global_step=trainer.global_step)

            do_show(
                grid, 
                #show = True, 
                show = current_epoch % self.every_n_epochs_show == 0,
                title = f"epoch={current_epoch}", 
                save_to = os.path.join(self.log_dir, f"grid_{current_epoch}.png")
            )

            if pl_module.use_wandb : wandb.log({"representation":  [wandb.Image(img) for img in images_]})


def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.
    https://stackoverflow.com/a/2669120/11814682
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    
def images_to_vidoe(image_folder, video_path, format="png") :
    """Thanks https://stackoverflow.com/a/44948030/11814682"""
    images = [img for img in sorted_nicely(os.listdir(image_folder))  if img.endswith(f".{format}")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def train(params, train_loader, val_loader):

    print()
    for k, v in vars(params).items() : print(k, " --> ", v)
    print()

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(params.log_dir, params.exp_id) 

    trainer_config = {
        "max_epochs": params.max_epochs,
        "default_root_dir" : root_dir,

        "accelerator" : params.accelerator,
        "devices" : params.devices,
        #"reload_dataloaders_every_n_epochs" : True,
        "weights_summary":"full", # "top", None,

        # "log_every_n_steps" : max(len(train_loader) // params.batch_size, 0),
        # "weights_save_path" : os.path.join(root_dir, "weights"),
        # "auto_scale_batch_size" : True, # None
        # "auto_select_gpus" : True,
        # "auto_lr_find": True,
        # "benchmark" : False,
        # "deterministic" : True,
        # "val_check_interval" : 1.,
        # "accumulate_grad_batches" : False,
        # "strategy": "ddp", # "ddp_spaw"
    }

    pp = vars(params)
    for split in ["train", "val", "test"] :
        v = pp.get(f"limit_{split}_batches", 1.0)
        if v != 1.0 : trainer_config[f"limit_{split}_batches"] = v

    validation_metrics = params.validation_metrics
    mode = (lambda s : "min" if 'loss' in s else 'max')(validation_metrics)
    early_stopping_callback = EarlyStopping(
        monitor=validation_metrics, patience=params.early_stopping_patience, verbose=False, strict=True,
        mode = mode
    )

    model_checkpoint_callback = ModelCheckpoint(
            dirpath=root_dir,
            save_weights_only=True,
            filename="{epoch}-{%s:.4f}"%validation_metrics,
            mode = mode,
            monitor=validation_metrics,
            save_top_k=params.save_top_k,
    )

    trainer_config["callbacks"] = [
        early_stopping_callback, 
        model_checkpoint_callback,
        GenerateCallback(
            log_dir = root_dir,
            func_params = params.func_params,
            every_n_epochs = params.every_n_epochs,
            every_n_epochs_show = params.every_n_epochs_show
        ), 
        LearningRateMonitor("epoch")
    ]

    trainer = pl.Trainer(**trainer_config)
    
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = root_dir + params.model_name
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model %s, loading..."%pretrained_filename)
        model = Model.load_from_checkpoint(pretrained_filename)
        print(model)
    else:
        # Initialize wandb
        if params.group_name is None : params.group_name = get_group_name(params, group_vars = params.group_vars)
        init_wandb(params.use_wandb, wandb_project = params.wandb_project, group_name = params.group_name, wandb_entity = params.wandb_entity)

        model = Model(params)
        print(model)
        trainer.fit(model, train_loader, val_loader, ckpt_path=params.checkpoint_path)
        model = Model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        #try : wandb.finish(exit_code = None, quiet = None)
        #except : pass

    # Test best model on validation set
    val_result = trainer.test(model, val_loader, verbose=False)
    train_result = trainer.test(model, train_loader, verbose=False)

    result = {"train": train_result, "val": val_result}
    for k1, v1 in copy(result).items() :
        #for k2 in v1[0] : result[k1][0][k2.replace("test", k1)] = round(result[k1][0].pop(k2), 4)
        result[k1] = {k2.replace("test", k1): round(result[k1][0][k2], 4) for k2 in v1[0]}
    
    return model, result