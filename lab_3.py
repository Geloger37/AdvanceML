# %%

import comet_ml
import os
import torch
import lightning
import itertools
from torch.utils.data import DataLoader
from torch import optim
from torchmetrics import ConfusionMatrix
from modules.dataset import EMODataset 
from modules.resnet import generate_model
from modules.headless_resnet import generate_model as headless_model
from sklearn.model_selection import ParameterGrid
from lightning.pytorch.loggers import CometLogger

# %%
train_ds = EMODataset(img_txt_dir='CREMA_D_img_txt/', subset='train', shape=(300, 400), max_length=8, padding=True)
test_ds = EMODataset(img_txt_dir='CREMA_D_img_txt/', subset='test', shape=(300, 400), max_length=8, padding=True)

# %%
class EmotionClassifier(lightning.LightningModule):
    def __init__(self, lr: int = None, n_classes: int = None, type_of_resnet: int = None):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.n_classes = n_classes
        self.type_of_resnet = type_of_resnet

        self.resnet = generate_model(type_of_resnet, n_classes=n_classes)

        self.loss = torch.nn.CrossEntropyLoss()

        self.conf_mat = ConfusionMatrix(task='multiclass', num_classes=n_classes, normalize='true')

    def forward(self, x):
        x = self.resnet(x)
        return x

    def training_step(self, batch, batch_idx):
        data, labels = batch

        outs = self(data)

        loss = self.loss(outs, labels)
        self.log("loss/train", loss.detach().cpu().item(), prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        
        outs = self(data)

        self.val_outs = torch.cat((self.val_outs, torch.argmax(outs, -1)))
        self.val_labels = torch.cat((self.val_labels, labels))

        val_loss = self.loss(outs, labels)
        self.log("loss/val", val_loss.detach().cpu().item(), prog_bar=True)
        
        return val_loss

    def on_validation_epoch_start(self):
        self.val_outs = torch.empty(0, device=self.device)
        self.val_labels = torch.empty(0, device=self.device)

    def on_validation_epoch_end(self):
        conf_matrix = self.conf_mat(self.val_outs, self.val_labels)
        
        uar = torch.mean(torch.diagonal(conf_matrix).float()).item()

        self.logger.experiment.log_confusion_matrix(y_true=self.val_labels.detach().cpu().numpy().astype(int), y_predicted=self.val_outs.detach().cpu().numpy().astype(int), labels=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'])
        self.log('uar/val', uar, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

# %%
class HeadlessEmotionClassifier(lightning.LightningModule):
    def __init__(self, lr: int = None, type_of_encoder: str = None, type_of_resnet: int = None, n_classes: int = None):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.n_classes = n_classes
        self.type_of_resnet = type_of_resnet
        self.type_of_encoder = type_of_encoder
        
        self.resnet = headless_model(type_of_resnet)

        self.loss = torch.nn.CrossEntropyLoss()

        dim = 512
        len_of_sequence = 8

        if type_of_encoder == 'lstm':
            class Post_lstm_layer(torch.nn.Module):
                def forward(self, x):
                    return x[0]

            self.encoder_layer = torch.nn.Sequential(torch.nn.LSTM(512, dim, 1, batch_first=True),
                                                     Post_lstm_layer(),
                                                     torch.nn.Flatten(start_dim=1),
                                                     torch.nn.Linear(len_of_sequence * dim, n_classes))
        elif type_of_encoder == 'transformer':
            self.encoder_layer = torch.nn.Sequential(torch.nn.TransformerEncoderLayer(d_model=dim, nhead=1, batch_first=True),
                                                     torch.nn.Flatten(start_dim=1),
                                                     torch.nn.Linear(len_of_sequence * dim, n_classes))

        self.conf_mat = ConfusionMatrix(task='multiclass', num_classes=n_classes, normalize='true')

    def forward(self, x):
        x = self.resnet(x)

        x = x.squeeze((3, 4)).permute((0, 2, 1))

        x = self.encoder_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        data, labels = batch

        outs = self(data)

        loss = self.loss(outs, labels)
        self.log("loss/train", loss.detach().cpu().item(), prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        
        outs = self(data)

        self.val_outs = torch.cat((self.val_outs, torch.argmax(outs, -1)))
        self.val_labels = torch.cat((self.val_labels, labels))

        val_loss = self.loss(outs, labels)
        self.log("loss/val", val_loss.detach().cpu().item(), prog_bar=True)
        
        return val_loss
        
    def on_validation_epoch_start(self):
        self.val_outs = torch.empty(0, device=self.device)
        self.val_labels = torch.empty(0, device=self.device)

    def on_validation_epoch_end(self):
        conf_matrix = self.conf_mat(self.val_outs, self.val_labels)
        
        uar = torch.mean(torch.diagonal(conf_matrix).float()).item()

        self.logger.experiment.log_confusion_matrix(y_true=self.val_labels.detach().cpu().numpy().astype(int), y_predicted=self.val_outs.detach().cpu().numpy().astype(int), labels=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'])
        self.log('uar/val', uar, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

# %%
def param_grid_iter(grid):
    for param in itertools.product(*grid.values()):
        yield dict(zip(grid.keys(), param))

hyperparameters = {
    'lr': [1e-4, 2e-5],
    'epochs': [10, 15],
    'batch_size': [16],
    'n_classes': [6],
    'type_of_resnet': [18]
}

train_dataloader = DataLoader(train_ds, batch_size=hyperparameters['batch_size'][0], num_workers=os.cpu_count())
val_dataloader = DataLoader(test_ds, batch_size=hyperparameters['batch_size'][0], num_workers=os.cpu_count())

# %%

for params in param_grid_iter(hyperparameters):
    comet_logger = CometLogger(
        save_dir="comet_logs",
        api_key='mir2VfuhUuhr28pomNEh9y7XX',
        project_name='AdvancedML-lab-3',
        experiment_name=f'RESNET {params}',
        offline=True
    )

    net = EmotionClassifier(lr = params['lr'], n_classes = params['n_classes'], type_of_resnet = params['type_of_resnet'])

    comet_logger.log_hyperparams(params)

    trainer = lightning.Trainer(max_epochs=params['epochs'], logger=comet_logger)

    trainer.fit(net, 
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)

# %%
hyperparameters = {
    'lr': [1e-4, 2e-5],
    'epochs': [10, 15],
    'batch_size': [16],
    'n_classes': [6],
    'type_of_resnet': [18],
    'type_of_encoder': ['lstm']
}

for params in param_grid_iter(hyperparameters):
    comet_logger = CometLogger(
        save_dir="comet_logs",
        api_key='mir2VfuhUuhr28pomNEh9y7XX',
        project_name='AdvancedML-lab-3',
        experiment_name=f'RESNET-LSTM {params}',
        offline=True
    )

    net = HeadlessEmotionClassifier(lr = params['lr'], n_classes = params['n_classes'], type_of_resnet=params['type_of_resnet'], type_of_encoder=params['type_of_encoder'])

    comet_logger.log_hyperparams(params)

    trainer = lightning.Trainer(max_epochs=params['epochs'], logger=comet_logger)

    trainer.fit(net, 
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)

# %%
hyperparameters = {
    'lr': [1e-4, 2e-5],
    'epochs': [10, 15],
    'batch_size': [16],
    'n_classes': [6],
    'type_of_resnet': [18],
    'type_of_encoder': ['transformer']
}

for params in param_grid_iter(hyperparameters):
    comet_logger = CometLogger(
        save_dir="comet_logs",
        api_key='mir2VfuhUuhr28pomNEh9y7XX',
        project_name='AdvancedML-lab-3',
        experiment_name=f'RESNET-transformer {params}',
        offline=True
    )

    net = HeadlessEmotionClassifier(lr = params['lr'], n_classes = params['n_classes'], type_of_resnet=params['type_of_resnet'], type_of_encoder=params['type_of_encoder'])

    comet_logger.log_hyperparams(params)

    trainer = lightning.Trainer(max_epochs=params['epochs'], logger=comet_logger)

    trainer.fit(net, 
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)