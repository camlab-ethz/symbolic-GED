import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal, Categorical

from encoder_pde import Encoder

# class Decoder(nn.Module):
#     """RNN decoder that reconstructs the sequence of rules from laten z"""
#     def __init__(self, input_size, hidden_size, output_size, rnn_type='lstm'):
#         super(Decoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.rnn_type = rnn_type

#         self.linear_in = nn.Linear(input_size, hidden_size)
#         self.linear_out = nn.Linear(hidden_size, output_size)

#         if rnn_type == 'lstm':
#             self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
#         elif rnn_type == 'gru':
#             self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
#         else:
#             raise ValueError('Select rnn_type from [lstm, gru]')

#         self.relu = nn.ReLU()

#     def forward(self, z, max_length):
#         """The forward pass used for training the Grammar VAE.

#         For the rnn we follow the same convention as the official keras
#         implementaion: the latent z is the input to the rnn at each timestep.
#         See line 138 of
#             https://github.com/mkusner/grammarVAE/blob/master/models/model_eq.py
#         for reference.
#         """
#         x = self.linear_in(z)
#         x = self.relu(x)

#         # The input to the rnn is the same for each timestep: it is z.
#         x = x.unsqueeze(1).expand(-1, max_length, -1)
#         hx = Variable(torch.zeros(x.size(0), self.hidden_size))
#         hx = (hx, hx) if self.rnn_type == 'lstm' else hx

#         x, _ = self.rnn(x, hx)

#         x = self.relu(x)
#         x = self.linear_out(x)
#         return x

class Decoder(nn.Module):
    """RNN decoder that reconstructs the sequence of rules from latent z."""
    def __init__(self, latent_rep_size, hidden_size, output_size, max_length, rnn_type='gru', num_layers=3, dropout=0.5):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.max_length = max_length
        self.num_layers = num_layers

        self.batch_norm = nn.BatchNorm1d(latent_rep_size)
        self.linear_in = nn.Linear(latent_rep_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        if rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.time_distributed_dense = nn.Linear(hidden_size*2, output_size)
        # self.time_distributed_dense = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        """
        The forward pass used for training the Grammar VAE.

        The latent z is repeated across the sequence length and passed through GRU layers.
        """
        h = self.relu(self.linear_in(z))
        # h = self.dropout(h)
        
        # Repeat the latent vector max_length times
        h = h.unsqueeze(1).repeat(1, self.max_length, 1)

        h, _ = self.rnn(h)
        # h = self.layer_norm(h)
        h = self.relu(h)
        # h = self.dropout(h)
        
        # Apply the output dense layer across all time steps
        h = self.time_distributed_dense(h)
        return h
if __name__ == '__main__':
    import h5py

    # # First run the encoder
    # Z_DIM = 2
    # BATCH_SIZE = 100
    # MAX_LENGTH = 15
    # OUTPUT_SIZE = 12

    # # Load data
    # data_path = '../data/eq2_grammar_dataset.h5'
    # f = h5py.File(data_path, 'r')
    # data = f['data']

    # # Create encoder
    # encoder = Encoder(100, Z_DIM)

    # # Pass through some data
    # x = torch.from_numpy(data[:BATCH_SIZE]).transpose(-2, -1).float() # shape [batch, 12, 15]
    # x = Variable(x)
    # _, y = x.max(1) # The rule index


    # mu, sigma = encoder(x)
    # z = encoder.sample(mu, sigma)
    
    # kl = encoder.kl(mu, sigma)

    # decoder = Decoder(Z_DIM, 100, 12,15)
    # logits = decoder(z)

    # criterion = torch.nn.CrossEntropyLoss()
    # logits = logits.view(-1, logits.size(-1))
    # y = y.view(-1)
    # loss = criterion(logits, y)

    # print(loss)
