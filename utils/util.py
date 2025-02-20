import os
import csv
import time
import torch
import torch.nn as nn
from model import GrammarVAE
import lightning.pytorch as pl
from collections import defaultdict
from lightning.pytorch.loggers import Logger
from torch.utils.data import DataLoader, TensorDataset, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch.nn.functional as F
from torch.autograd import Variable

class MyPerpLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, true_binary, input_logits):
        ctx.save_for_backward(true_binary, input_logits)
        # print('shape',input_logits.shape)
        b = torch.max(input_logits, 1, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = torch.exp(raw_logits) + 1e-9  # Small constant to avoid log(0)

        norm = torch.sum(exp_pred, 1, keepdim=True)
        prob = exp_pred / norm

        ll = torch.abs(torch.sum(true_binary * prob, 1))
        
        logll = torch.log(ll)

        loss = -torch.sum(logll) / true_binary.size(1)
        
        if input_logits.is_cuda:
            return torch.Tensor([loss]).cuda()
        else:
            return torch.Tensor([loss])

    @staticmethod
    def backward(ctx, grad_output):
        true_binary, input_logits = ctx.saved_tensors

        b = torch.max(input_logits, 1, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = torch.exp(raw_logits) + 1e-9

        norm = torch.sum(exp_pred, 1, keepdim=True)
        prob = exp_pred / norm

        grad_matrix3 = prob - true_binary
        grad_matrix3 = grad_matrix3 * grad_output.data / true_binary.size(1)

        return None, Variable(grad_matrix3)

def my_perp_loss(true_binary, input_logits):
    return MyPerpLoss.apply(true_binary, input_logits)



import h5py
from nltk import Tree, Nonterminal

class Timer:
	"""A simple timer to use during training"""
	def __init__(self):
		self.time0 = time.time()

	def elapsed(self):
		time1 = time.time()
		elapsed = time1 - self.time0
		self.time0 = time1
		return elapsed

class AnnealKL:
	"""Anneal the KL for VAE based training"""
	def __init__(self, step=1e-3, rate=1000): #change the step to 1e-6 from 1e-3
		self.rate = rate
		self.step = step

	def alpha(self, update):
		n, _ = divmod(update, self.rate)
		# return min(1., n*self.step) #initial 
		# return min(1/43*0.1, n*self.step) # we want tighter distribution 
		return 1/30 * 0.01 # we want tighter distribution 

def load_data(data_path):
	"""Returns the h5 dataset as numpy array"""
	f = h5py.File(data_path, 'r')
	return f['data'][:]

def make_nltk_tree(derivation):
	"""return a nltk Tree object based on the derivation (list or tuple of Rules)."""
	d = defaultdict(None, ((r.lhs(), r.rhs()) for r in derrivation))
	def make_tree(lhs, rhs):
		return Tree(lhs, (child if child not in d else make_tree(child) for child in d[lhs]))
		return Tree(lhs,
				(child if not isinstance(child, Nonterminal) else make_tree(child)
					for child in rhs))

	return make_tree(r.lhs(), r.rhs())

class CSVLogger(Logger):
	def __init__(self, save_dir, name='logs', version=None):
		super().__init__()
		self._save_dir = save_dir
		self._name = name
		self._version = version
		self.metrics = []
		self.fieldnames = set()
		os.makedirs(self.save_dir, exist_ok=True)        
	@property
	def save_dir(self):
		return self._save_dir
	@property
	def name(self):
		return self._name
    
      
	@property
	def version(self):
		return self._version
	 
	def log_metrics(self, metrics, step=None):
		self.fieldnames.update(metrics.keys())
		self.metrics.append(metrics)

	def save(self):
		if self.metrics:
			keys = list(self.fieldnames)
			with open(os.path.join(self.save_dir, f'{self.name}_metrics-pde.csv'), 'w', newline='') as output_file:
				dict_writer = csv.DictWriter(output_file, fieldnames=keys)
				dict_writer.writeheader()
				for metric in self.metrics:
					row = {key: metric.get(key, None) for key in keys}
					dict_writer.writerow(row)

	def experiment(self):
		pass

	def log_hyperparams(self, params):
		pass

	def finalize(self, status):
		self.save()

class GrammarVAEModel(pl.LightningModule):
    def __init__(self, model_cfg,training_cfg):
        super(GrammarVAEModel, self).__init__()
        self.model = GrammarVAE(
            model_cfg['encoder_hidden'], 
            model_cfg['z_size'], 
            model_cfg['decoder_hidden'], 
            model_cfg['output_size'], 
            model_cfg['rnn_type'], 
            model_cfg['max_length'], 
            model_cfg['dropout'],
			model_cfg['tightness_param'], 
            model_cfg['dec_layers'] 
        )
        self.criterion = nn.CrossEntropyLoss()
        self.anneal = AnnealKL(step=1e-3, rate=100)
        self.validation_split = training_cfg['validation_split']
        self.lr = training_cfg['learning_rate']
        self.batch = training_cfg['batch_size']
        self.save_hyperparameters()

    def forward(self, x):
        x = x.to(self.device)
        mu, sigma = self.model.encoder(x)
        z = self.model.sample(mu, sigma)
        logits = self.model.decoder(z)
        return logits, mu, sigma

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits, mu, sigma = self(x)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
		# #Convert y to the expected format for MyPerpLoss
        # true_binary = F.one_hot(y, num_classes=logits.size(-1)).float()

        # # Compute the custom loss using the simplified MyPerpLoss
        # loss = my_perp_loss(true_binary, logits)
        loss = self.criterion(logits, y)
        kl = self.model.kl(mu, sigma)
        alpha = self.anneal.alpha(self.global_step)
        elbo = loss + alpha * kl

        acc = self.accuracy(logits, y)
        perplexity = torch.exp(loss)
        self.log('train_loss', loss)
        self.log('train_kl', kl)
        self.log('train_elbo', elbo)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_perplexity', perplexity, prog_bar=True)
        
        return elbo

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits, mu, sigma = self(x)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = self.criterion(logits, y)
        kl = self.model.kl(mu, sigma)
        alpha = self.anneal.alpha(self.global_step)
        elbo = loss + alpha * kl

        acc = self.accuracy(logits, y)
        perplexity = torch.exp(loss)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_kl', kl, prog_bar=True)
        self.log('val_elbo', elbo, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)
        
        return elbo

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_elbo',
            },
        }

    def accuracy(self, logits, y):
        _, y_pred = logits.max(-1)
        acc = (y == y_pred).float().mean()
        return 100 * acc

    def setup(self, stage=None):
        data_path = '/cluster/scratch/ooikonomou/latest/filtered_100_80_euler_sorted_1terminal-new-20-no0_in_const20k-2terminals-latest more-correct-120-no-e-no-3decimals.h5'
        data = load_data(data_path)
        data = torch.from_numpy(data).float()
        data = data.transpose(1, 2)
        targets = data.argmax(1)
        dataset = TensorDataset(data, targets)
        train_size = int((1 - self.validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch, num_workers=3)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
# data_path = '/cluster/scratch/ooikonomou/filtered_100_60_euler_sorted_1terminal-new-20-no0_in_const20k-2terminals.h5'
        