# torch
import torch
import torch.nn as nn

# PyTorch Lightning
import pytorch_lightning as pl

# wandb
import wandb

import itertools
import math

## ID
from intrinsics_dimension import twonn_pytorch, mle_id
ID_functions = {"twonn" : twonn_pytorch, "mle" : mle_id}

possible_metrics = ["%s_%s"%(i, j) for i, j in itertools.product(["train", "val"], ["loss_y", "loss_dydx"])]

###### TOCORRECT
from .optim import configure_optimizers
from .utils_derivs import gradient

def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

class Linear(nn.Module):
    """costomized linear layer"""
    def __init__(self, in_features, out_features, bias = True, activation_function = None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias = bias)
        self.activation_function = activation_function if activation_function else lambda x : x
    def forward(self, x):
        return self.activation_function(self.linear(x))

class MLP(nn.Module):
    """Multi-layer perceptron"""
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, activation_function = nn.LeakyReLU()):
        super(MLP, self).__init__()
        if hidden_layers == 0 :
            self.net = Linear(in_features, out_features, True, None)
        else :
            net = [Linear(in_features, hidden_features, True, activation_function)]
            net += [Linear(hidden_features, hidden_features, True, activation_function) for _ in range(hidden_layers-1)]  
            net.append(Linear(hidden_features, out_features, True, None))
            self.net = nn.Sequential(*net)
        self.hidden_layers = hidden_layers

    def forward(self, x, return_layers = False):
        zs = []
        if not return_layers or self.hidden_layers == 0 :
            return self.net(x), zs
        z = x + 0.0
        for linear_layer in self.net :
            z = linear_layer(z)
            zs.append(z.detach()) 
        return z, zs[:-1]

class Model(pl.LightningModule):
    """
    params : 
        - ndim (int), hidden_dim (int),  n_layers (int)
        - use_wandb (bool, optional, False)
        - lr (float, optional, 1e-3), weight_decay (float, optional, 0) 
        - patience (float, optional, 20), min_lr (float, optional, 5e-5)
        - ID_params (dict, optional, None)
        - lambda_j (float), alpha (float), beta (float)
        - early_stopping_grokking (dict)
    """
    def __init__(self, params):
        """
        Transformer model 
        """
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters(params) 

        #self.mlp = make_mlp([self.hparams.ndim] + [self.hparams.hidden_dim] * self.hparams.n_layers + [1])  
        self.mlp = MLP(
            in_features = self.hparams.ndim, hidden_features = self.hparams.hidden_dim, 
            hidden_layers = self.hparams.n_layers, out_features = 1, activation_function = nn.LeakyReLU()
        )
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.criterion = nn.MSELoss() 

        self.data_config = self.hparams.data_config        
        alpha_beta = getattr(self.hparams, "alpha_beta", {})
        if  alpha_beta is None or not alpha_beta :
            self.alpha, self.beta = self.data_config.alpha, self.data_config.beta
        else : 
            self.alpha, self.beta = self.hparams.alpha_beta["alpha"], self.hparams.alpha_beta["beta"]
        
        if self.hparams.get("ID_params") is None : self.hparams["ID_params"] = {}
        ID_params = {**{}, **self.hparams.get("ID_params", {"method" : "mle", "k":2})}
        #ID_params = {"method" : "twonn"}
        id_funct = ID_functions.get(ID_params.pop("method", None), None)
        self.ID_function = id_funct
        setattr(self, "ID", id_funct is not None and self.hparams.hidden_dim != 0)
        self.ID_params = ID_params

        self.use_wandb = self.hparams.use_wandb

        # State
        self.grok = False
        self.comprehension = False
        self.memorization = False
        self.confusion = True
        self.comp_epoch = float("inf")
        self.memo_epoch = float("inf")

        # Early stopping grokking : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold`
        early_stopping_grokking = self.hparams.early_stopping_grokking
        if type(early_stopping_grokking) != dict : early_stopping_grokking = {} 
        self.es_patience = early_stopping_grokking.get("patience", self.hparams.max_epochs)
        self.es_metric = early_stopping_grokking.get("metric", "val_loss_y") 
        assert self.es_metric in possible_metrics
        self.es_metric_threshold = early_stopping_grokking.get("metric_threshold", 0.0 if 'loss' in self.es_metric else 99.0) 
        self.es_mode = (lambda s : "min" if 'loss' in s else 'max')(self.es_metric)
        self.es_step = 0
        self.reached_limit = False

    def configure_optimizers(self):
        return configure_optimizers(self.mlp.parameters(), self.hparams.optimizer, self.hparams.lr_scheduler)

    def forward(self, x):
        """
        Inputs: LongTensor(bs, ndim)
        """ 
        y_pred, zs = self.mlp(x, return_layers = self.ID)
        y_pred = self.dropout(y_pred) 
        return y_pred, zs
    
    def _get_loss(self, batch, prefix):
        """
        Given a batch of data, this function returns the loss (MSE)
        """
        x, y, dydx = batch 
        x.requires_grad_(True)
        y_pred, zs = self.forward(x)
        loss_y = self.criterion(input = y_pred.squeeze(), target = y)
        dydx_pred = gradient(y_pred, x)
        lambda_j = self.data_config.lambda_j.to(dydx_pred.device)
        loss_dydx = self.criterion(input = lambda_j * dydx_pred, target = lambda_j * dydx)
        loss = self.alpha * loss_y + self.beta * loss_dydx 
        output = {
            #f'{prefix}loss' : loss, 
            f'{prefix}loss_y' : loss_y, f'{prefix}loss_dydx' : loss_dydx
        }
        if self.hparams.normalize :
            y_mean = self.data_config.y_mean.to(dydx_pred.device)
            y_std = self.data_config.y_std.to(dydx_pred.device)
            x_std = self.data_config.x_std.to(dydx_pred.device)
            y_pred_ = y_mean + y_std * y_pred
            dydx_pred_ = (y_std / x_std) * dydx_pred
            y_ = y_mean + y_std * y
            dydx_ = (y_std / x_std) * dydx
            output[f"{prefix}loss_y_no_scaled"] = self.criterion(input = y_pred_.squeeze(), target = y_)#.item()
            output[f"{prefix}loss_dydx_no_scaled"] = self.criterion(input = dydx_, target = dydx_pred_)#.item()

        for k, v in output.items() : self.log(k, v, prog_bar=True)
        output["zs"] = zs

        return loss, output, zs
    
    def training_step(self, batch, batch_idx):
        loss, output, _ = self._get_loss(batch, prefix="train_")  
        output["loss"] = loss
        return output 
    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        _, output, _ = self._get_loss(batch, prefix="val_")
        return output 
    
    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        _, output, _ = self._get_loss(batch, prefix="test_")
        return output 

    def _group_zs(self, outputs):
        """
        Merges the embedding representation from all batches in one epoch.
        """ 
        # zs = torch.cat([output["zs"] # (batch_size, embed_dim)
        #                          for output in outputs], dim=0) # (_, embed_dim)
        zs = [
            torch.cat([output["zs"][l] for output in outputs], dim=0) # (batch_size, embed_dim)
            for l in range(len(outputs[0]["zs"]))
        ] # _ x (_, embed_dim)

        return zs

    def compute_intrinsic_dimension(self, outputs, prefix, batch_size = 1000):
        """
        Estimate intrinsic dimensions using all bottlenecks collected across one epoch
        bottlenecks : (n, latent_dim)    
        """
        zs = self._group_zs(outputs)
        result = {}
        for l in range(len(zs)): 
            z = zs[l] # (n, embed_dim)
            if False :
                int_dim = self.ID_function(data=z, **self.hparams.ID_params)
            else :
                z = z[:5000] # to save time, juste use 5000 samples
                try : int_dim = self.ID_function(data=z, **self.ID_params)
                except RuntimeError: #CUDA out of memory
                    # batchification
                    i, int_dim = 0, []
                    while i < z.size(0) : int_dim.append(self.ID_function(data=zs[i: i + batch_size], **self.ID_params))
                    int_dim = sum(int_dim) / len(int_dim)
            result[f"{prefix}ID_layer_{l}"] = int_dim
        return result

    def increase_es_limit(self, logs):
        es_metric = logs[self.es_metric]
        self.reached_limit = self.reached_limit or (es_metric >= self.es_metric_threshold if self.es_mode == "max" 
                                                    else es_metric <= self.es_metric_threshold)
        if self.reached_limit : self.es_step+=1
        return self.es_step

    def group_metrics(self, outputs, prefix : str) :
        loss_y = torch.stack([x[f"{prefix}loss_y"] for x in outputs]).mean()
        loss_dydx = torch.stack([x[f"{prefix}loss_dydx"] for x in outputs]).mean()
        logs = {
            f"{prefix}loss_y": loss_y, 
            f"{prefix}loss_dydx": loss_dydx
        }
        if self.hparams.normalize :
            logs[f"{prefix}loss_y_no_scaled"] = torch.stack([x[f"{prefix}loss_y_no_scaled"] for x in outputs]).mean()
            logs[f"{prefix}loss_dydx_no_scaled"] = torch.stack([x[f"{prefix}loss_dydx_no_scaled"] for x in outputs]).mean()

        if self.ID : 
            id_output = self.compute_intrinsic_dimension(outputs, prefix, batch_size = 1000)
            logs = {**logs, **id_output}
        
        return logs, loss_y

    def training_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        """
        #loss_y = torch.stack([x["train_loss_y"] for x in outputs]).mean()
        #loss_dydx = torch.stack([x["train_loss_dydx"] for x in outputs]).mean()
        #logs = {"train_loss_y": loss_y, "train_loss_dydx": loss_dydx}

        logs, loss_y = self.group_metrics(outputs, prefix = 'train_')

        if 'train' in self.es_metric : logs["e_step"] = self.increase_es_limit(logs)

        memo_condition = round(loss_y.item(), 10) == 0.0

        self.memorization = self.memorization or memo_condition
        if memo_condition : self.memo_epoch = min(self.current_epoch, self.memo_epoch)
               
        logs["train_epoch"]  = self.current_epoch

        schedulers = self.lr_schedulers()
        if schedulers is not None :
            try : scheduler = schedulers[0]
            except TypeError: scheduler = schedulers # 'xxx' object is not subscriptable
            param_groups = scheduler.optimizer.param_groups
            logs["lr"] = param_groups[0]["lr"]

        for k, v in logs.items() : self.log(k, v, prog_bar=True)
        if self.use_wandb: wandb.log(logs)

    def validation_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        """
        logs, loss_y = self.group_metrics(outputs, prefix = 'val_')

        comp_condition = round(loss_y.item(), 10) == 0.0
 
        self.comprehension = self.comprehension or comp_condition
        if comp_condition : self.comp_epoch = min(self.current_epoch, self.comp_epoch)

        if 'val' in self.es_metric : logs["e_step"] = self.increase_es_limit(logs)

        for k, v in logs.items() : self.log(k, v, prog_bar=True)
        if self.use_wandb: wandb.log(logs)

        self.grok = self.comprehension and True # and long step of training
        self.memorization = (not self.comprehension) and self.memorization
        self.confusion = (not self.comprehension) and (not self.memorization)

        # diff_epoch = self.comp_epoch - self.memo_epoch
        # if not math.isnan(diff_epoch) : 
        #     self.grok = diff_epoch >= 100
        #     self.comprehension = not self.grok

        self.states = {
            "grok":self.grok, "comprehension":self.comprehension, "memorization": self.memorization, "confusion":self.confusion,
            "comprehension_epoch":self.comp_epoch, "memorization_epoch":self.memo_epoch
        }

    def send_dict_to_wandb(self, data, label, title) :
        if self.hparams.use_wandb:  
            labels = data.keys()
            values = data.values()
            data = [[label, val] for (label, val) in zip(labels, values)]
            table = wandb.Table(data=data, columns = ["label", "value"])
            wandb.log({label : wandb.plot.bar(table, "label", "value", title=title)})
    
    def on_train_start(self):
        db_data = getattr(self.hparams, "data_infos", None)
        if db_data is not None : self.send_dict_to_wandb(db_data, label = "data_info", title="Dataset Informations")

    def on_train_end(self) :

        # diff_epoch = self.comp_epoch - self.memo_epoch
        # if not math.isnan(diff_epoch) : 
        #     self.grok = diff_epoch >= 100
        #     self.comprehension = not self.grok

        states = {
            "grok":int(self.grok), "comprehension":int(self.comprehension), "memorization":int(self.memorization), "confusion":int(self.confusion),
            "comprehension_epoch":self.comp_epoch, "memorization_epoch":self.memo_epoch
        }
        self.send_dict_to_wandb(states, label = "states_info", title="Phase Informations")

if __name__ == "__main__":

    from utils import AttrDict

    ndim = 3
    params = AttrDict({
        "ndim" : ndim,
        "hidden_dim" : 105,  
        "n_layers" : 1,
        "dropout" : 0.0,

        "use_wandb" : False,

        "early_stopping_grokking" : {"patience" : 1000, "metric" : "val_loss_y", "metric_threshold" : 90.0},
        "optimizer" : "adam,lr=0.001,weight_decay=0.0,beta1=0.9,beta2=0.99,eps=0.00000001",
        #"lr_scheduler" : None,
        "lr_scheduler" : "reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss_y",
        
        #"ID_params" : {},
        "ID_params": {"method" : "twonn"},

        "alpha_beta" : {"alpha" : 1.0, "beta" : 1.0}, 

        "max_epochs" : 100,

        "data_config" : AttrDict({
            "x_mean" : 0.0, "x_std" : 1.0, "y_mean" : 0.0, "y_std" : 1.0, "lambda_j" : 1.0, "n" : -1,
            "get_alpha_beta" : lambda lam : (1.0, 1.0)
        }),
        "normalize" : False
    })

    model = Model(params)
    print(model, "\n", model.ID, "\n")    

    bs = 4
    x = torch.zeros(size=(bs, ndim), dtype=torch.float)
    y = x.sum(1) 
    dydx = torch.ones_like(x)
    print(y.shape)

    y_pred, zs = model(x)
    print(y_pred.shape)
    print([z.shape for z in zs])
    
    loss, output, zs = model._get_loss(batch = (x, y, dydx), prefix="train_")
    print(loss, output, len(zs))
    print([z.shape for z in zs])