import torch
import numpy as np
import pytorch_lightning as pl
from torch import optim, nn
import itertools
from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Smoother(pl.LightningModule):
    def __init__(self, exp_encoder, tpy_encoder, exp_decoder, noise_separator, n_exp, n_tpy):
        super(Smoother, self).__init__()

        self.n_exp = n_exp
        self.n_tpy = n_tpy

        if not isinstance(exp_encoder, nn.Module):
            raise Exception('exp_encoder must be a nn.Module')
        if not isinstance(tpy_encoder, nn.Module):
            raise Exception('tpy_encoder must be a nn.Module')
        if not isinstance(exp_decoder, nn.Module):
            raise Exception('exp_decoder must be a nn.Module')

        self.exp_encoder = exp_encoder
        self.tpy_encoder = tpy_encoder
        self.exp_decoder = exp_decoder
        self.noise_separator = noise_separator

        self.mse_loss = nn.MSELoss()
        self.bec_loss = nn.BCEWithLogitsLoss()

    def forward(self, exp_x, tpy_x):
        mask = torch.where(exp_x <= 0., True, False)
        # To mix the influence between exp and its tpy.
        exp_y = self.exp_encoder(exp_x)
        tpy_y = self.tpy_encoder(tpy_x)
        fused_features = torch.cat([exp_y, tpy_y], dim=1)
        # To recover the exp and split noise from it.
        recover_features = self.exp_decoder(fused_features)
        #print(recover_features.shape)
        #f_features = recover_features
        recover_features = torch.masked_fill(recover_features, mask, 0.)
        recover_features, noise = self.noise_separator(recover_features)
        # For calculation, add a tiny increment.
        epsilon = torch.ones(
            (recover_features.shape[0], recover_features.shape[1]), dtype=torch.float, device=self.device
        ) * 1e-3
        recover_features_ = recover_features + epsilon
        recover_sim = self.__euclidean_distance(recover_features_)
        return recover_features, recover_sim, noise #f_features

    def training_step(self, batch, batch_idx):
        exp_x, tpy_x = batch
        epsilon = torch.ones((exp_x.shape[0], exp_x.shape[1]), dtype=torch.float, device=self.device) * 1e-3
        exp_x_ = exp_x + epsilon
        sim_y = self.__euclidean_distance(exp_x_)

        exp_y, tpy_sim, noise= self(exp_x, tpy_x)
        mae_loss = self.mse_loss(exp_y, exp_x)
        bec_loss = self.bec_loss(tpy_sim.view(-1), sim_y.view(-1))
        loss = mae_loss + bec_loss
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return optimizer

    def backward(self, loss, **kwargs):
        loss.backward()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        exp_x, tpy_x = batch
        exp_y, _, noise = self(exp_x, tpy_x)

        return exp_y, noise

    def __euclidean_distance(self, x):
        return torch.pairwise_distance(x.t(), x.t())


class ExpEncoder(nn.Module):
    def __init__(self, in_features: int = 0, out_features: int = 0, hid_features=None):
        super(ExpEncoder, self).__init__()

        if hid_features is None:
            hid_features = []
        if not isinstance(in_features, int):
            raise Exception('in_features must be an int')
        if not isinstance(out_features, int):
            raise Exception('out_features must be an int')
        if not isinstance(hid_features, list):
            raise Exception('hid_features must be a list')

        self.module = nn.Sequential()
        prev = in_features
        for nf in hid_features:
            self.module.append(BaseBlock(prev, nf))
            prev = nf
        self.module.append(BaseBlock(prev, out_features))

    def forward(self, x):
        return self.module(x)


class TpyEncoder(nn.Module):
    def __init__(self, in_features: int = 0, out_features: int = 0, layer_depth: int = 0):
        super(TpyEncoder, self).__init__()

        if not isinstance(in_features, int):
            raise Exception('in_features object type error')
        if not isinstance(out_features, int):
            raise Exception('in_features object type error')
        if not isinstance(layer_depth, int):
            raise Exception('layer_depth object type error')

        self.module = nn.Sequential()
        for d in range(layer_depth):
            self.module.append(ResBlock(in_features, out_features))

    def forward(self, x):
        return self.module(x)


class BaseBlock(nn.Module):
    def __init__(self, in_features: int = 0, out_features: int = 0):
        super(BaseBlock, self).__init__()

        if not isinstance(in_features, int):
            raise Exception('in_features must be a int')
        if not isinstance(out_features, int):
            raise Exception('out_features must be a int')

        self.module = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        return self.module(x)


class ResBlock(nn.Module):
    def __init__(self, in_features: int = 0, out_features: int = 0):
        super(ResBlock, self).__init__()

        if not isinstance(in_features, int):
            raise Exception('in_features must be a int')
        if not isinstance(out_features, int):
            raise Exception('out_features must be a int')

        self.module = BaseBlock(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        output = self.module(x)
        output = output + x
        output = self.norm(output)

        return output


class ExpDecoder(nn.Module):
    def __init__(self, in_features: int = 0, out_features: int = 0, hid_features=None):
        super(ExpDecoder, self).__init__()

        if hid_features is None:
            hid_features = []
        if not isinstance(in_features, int):
            raise Exception('in_features must be an int')
        if not isinstance(out_features, int):
            raise Exception('in_features must be an int')
        if not isinstance(hid_features, list):
            raise Exception('hid_features must be a list')

        self.module = nn.Sequential()
        prev = in_features
        for nf in hid_features:
            self.module.append(BaseBlock(prev, nf))
            prev = nf
        self.module.append(BaseBlock(prev, out_features))

    def forward(self, x):
        return self.module(x)


class NoiseSeparator(pl.LightningModule):
    def __init__(self, n_feat, batch_size):
        super(NoiseSeparator, self).__init__()

        self.n_feat = n_feat
        self.batch_size = batch_size
        self.noise = nn.Parameter(0.1 * torch.rand(batch_size, n_feat))

        self.module = nn.Sequential()
        for _ in range(3):
            self.module.append(ResBlock(n_feat, n_feat))

    def forward(self, x):
        if x.shape[0] < self.batch_size:
            out = x - self.noise[0: x.shape[0], :]
        else:
            out = x - self.noise
        for m in self.module:
            out = m(out)

        return out, self.noise



class ScDataset(Dataset):
    def __init__(self, exp, tpy, batch_size):
        super(ScDataset, self).__init__()

        if not isinstance(exp, torch.Tensor):
            raise Exception('exp must be a tensor')
        if not isinstance(tpy, torch.Tensor):
            raise Exception('tpy must be a tensor')
        if not isinstance(batch_size, int):
            raise Exception('smi must be a int')

        self.exp = exp
        self.tpy = tpy
        self.batch_size = batch_size
        self.idx = []

    def __getitem__(self, item):
        return self.exp[item, :], self.tpy[item, :]

    def __len__(self):
        return self.exp.size(0)

    def __re_idx(self):
        return itertools.product(self.idx, self.idx)



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_float32_matmul_precision('high')


def normalize_data(X):
    X = X.T
    X_norm = X.copy()
    X_norm = X_norm.astype('float32')
    libs = X.sum(axis=1)
    med = np.median(libs)
    norm_factor = np.diag(med / libs)
    data_norm = np.dot(norm_factor, X_norm)
    data_norm = pd.DataFrame(data_norm, index=None, columns=None)
    return data_norm.T


def logtranform_data(X):
    Y = X.copy()
    Y = np.log10(Y + 1.)
    return Y

processed_data_file = '../data/Clustering/Pollen_HVG.csv'
module_path = '../data/Clustering/Pollen_module.pt'
exp_save_path = '../data/Clustering/Pollen_scDMAE_f.csv'
noise_save_path = '../data/Clustering/Pollen_noise.csv'
fusefeature_save_path = '../data/Clustering/Pollen_fusefeature.csv'

print('Preprocess data')
process_data = pd.read_csv(processed_data_file, header=0, index_col=0)

n_gene = process_data.shape[0]
n_cell = process_data.shape[1]
process_data = logtranform_data(normalize_data(process_data))

feature_exp = torch.from_numpy(np.array(process_data.values))
feature_exp = feature_exp.to(torch.float)

n_tpy = 32
feature_tpy = torch.rand((n_gene, n_tpy))
feature_tpy = feature_tpy.to(torch.float)


batch_size = 256  #1024
max_epochs = 500

sc_dataset = ScDataset(feature_exp, feature_tpy, batch_size)
sc_dataloader = DataLoader(
    sc_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
exp_encoder = ExpEncoder(in_features=n_cell, out_features=n_tpy, hid_features=[128, 64])
tpy_encoder = TpyEncoder(in_features=n_tpy, out_features=n_tpy, layer_depth=3)
exp_decoder = ExpDecoder(in_features=n_tpy + n_tpy, out_features=n_cell, hid_features=[64, 128])
noise_separator = NoiseSeparator(n_feat=n_cell, batch_size=batch_size)
smoother = Smoother(exp_encoder, tpy_encoder, exp_decoder, noise_separator, n_cell, n_tpy)
logger = TensorBoardLogger('./tf-logs', name='smoother')
trainer = Trainer(
    accelerator='gpu', max_epochs=max_epochs, devices='auto', logger=False, fast_dev_run=False,
    callbacks=[EarlyStopping(monitor='loss', mode="min", patience=5)]
)

trainer.fit(smoother, sc_dataloader)

torch.save(smoother.state_dict(), module_path)
smoother.eval()


predict_dataloader = DataLoader(
    sc_dataset, batch_size=batch_size, shuffle=False, num_workers=8
)
results = trainer.predict(smoother, predict_dataloader)

exp_y = torch.cat([results[i][0] for i in range(len(results))], dim=0)
exp_y = pd.DataFrame(exp_y.detach().cpu().numpy(), columns=None, index=None)
exp_y.to_csv(exp_save_path)
noise = torch.cat([results[i][1] for i in range(len(results))], dim=0)
noise = pd.DataFrame(noise.detach().cpu().numpy(), columns=None, index=None)
noise.to_csv(noise_save_path)

