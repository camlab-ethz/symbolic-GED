import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal, Categorical

# class Encoder(nn.Module):
#     """Convolutional encoder for Grammar VAE.

#     Applies a series of one-dimensional convolutions to a batch
#     of one-hot encodings of the sequence of rules that generate
#     an arithmetic expression.
#     """
#     def __init__(self, hidden_dim=100, z_dim=2, conv_size='large', dropout=0.5):
#         super(Encoder, self).__init__()
#         if conv_size == 'small':
#             # 12 rules, so 12 input channels
#             self.conv1 = nn.Conv1d(12, 2, kernel_size=2)
#             self.conv2 = nn.Conv1d(2, 3, kernel_size=3)
#             self.conv3 = nn.Conv1d(3, 4, kernel_size=4)
#             self.linear = nn.Linear(36, hidden_dim)
#             self.batch_norm1 = nn.BatchNorm1d(2)
#             self.batch_norm2 = nn.BatchNorm1d(3)
#             self.batch_norm3 = nn.BatchNorm1d(4)
#         elif conv_size == 'large':
#             self.conv1 = nn.Conv1d(12, 24, kernel_size=2)
#             self.conv2 = nn.Conv1d(24, 12, kernel_size=3)
#             self.conv3 = nn.Conv1d(12, 12, kernel_size=4)
#             self.linear = nn.Linear(108, hidden_dim)
#             self.batch_norm1 = nn.BatchNorm1d(24)
#             self.batch_norm2 = nn.BatchNorm1d(12)
#             self.batch_norm3 = nn.BatchNorm1d(12)
#         else:
#             raise ValueError('Invalid value for `conv_size`: {}.'
#                              ' Must be in [small, large]'.format(conv_size))

#         self.mu = nn.Linear(hidden_dim, z_dim)
#         self.sigma = nn.Linear(hidden_dim, z_dim)
#         self.dense = nn.Linear(12 * 9, hidden_dim)  # Corrected input size for the dense layer large
#         self.relu = nn.ReLU()
#         self.softplus = nn.Softplus()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         """Encode x into a mean and variance of a Normal"""
#         h = self.conv1(x)
#         h = self.batch_norm1(h)
#         h = self.relu(h)
#         h = self.dropout(h)
#         h = self.conv2(h)
#         h = self.batch_norm2(h)
#         h = self.relu(h)
#         h = self.dropout(h)
#         h = self.conv3(h)
#         h = self.batch_norm3(h)
#         h = self.relu(h)
#         h = self.dropout(h)
#         h = h.view(x.size(0), -1)  # flatten
#         h = self.dense(h)
#         h = self.relu(h)
#         mu = self.mu(h)
#         sigma = self.softplus(self.sigma(h))
#         return mu, sigma



# # dsds
# class Encoder(nn.Module):
#     """Convolutional encoder for Grammar VAE.

#     Applies a series of one-dimensional convolutions to a batch
#     of one-hot encodings of the sequence of rules that generate
#     an arithmetic expression.
#     """
#     def __init__(self, input_dim=12, hidden_dim=100, z_dim=2, conv_size='large', dropout=0.5):
#         super(Encoder, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.z_dim = z_dim
#         self.conv_size = conv_size
#         self.dropout_rate = dropout

#         if conv_size == 'small':
#             self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=4, stride=2, padding=1)
#             self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)
#             self.linear = nn.Linear(64 * ((input_dim // 4) // 2), hidden_dim)
#             self.batch_norm1 = nn.BatchNorm1d(32)
#             self.batch_norm2 = nn.BatchNorm1d(64)
#         elif conv_size == 'large':
#             self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=4, stride=2, padding=1)
#             self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
#             self.linear = nn.Linear(128 * ((input_dim // 4) // 2), hidden_dim)
#             self.batch_norm1 = nn.BatchNorm1d(64)
#             self.batch_norm2 = nn.BatchNorm1d(128)
#         else:
#             raise ValueError('Invalid value for `conv_size`: {}. Must be in [small, large]'.format(conv_size))

#         self.mu = nn.Linear(hidden_dim, z_dim)
#         self.sigma = nn.Linear(hidden_dim, z_dim)
#         self.relu = nn.ReLU()
#         self.softplus = nn.Softplus()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         """Encode x into a mean and variance of a Normal"""
#         h = self.conv1(x)
#         h = self.batch_norm1(h)
#         h = self.relu(h)
#         h = self.dropout(h)
#         h = self.conv2(h)
#         h = self.batch_norm2(h)
#         h = self.relu(h)
#         h = self.dropout(h)
#         h = h.view(h.size(0), -1)  # flatten
#         h = self.linear(h)
#         h = self.relu(h)
#         mu = self.mu(h)
#         sigma = self.softplus(self.sigma(h))
#         return mu, sigma

#     def sample(self, mu, sigma):
#         """Reparameterized sample from a N(mu, sigma) distribution"""
#         normal = Normal(torch.zeros(mu.shape).to(mu.device), torch.ones(sigma.shape).to(sigma.device))
#         eps = Variable(normal.sample())
#         z = mu + eps * torch.sqrt(sigma)
#         return z

#     def kl(self, mu, sigma):
#         """KL divergence between two normal distributions"""
#         return torch.mean(-0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), 1))


import torch
import torch.nn as nn
class Encoder(nn.Module):
    """Encoder for Grammar VAE with both Convolutional and GRU layers."""

    def __init__(self, input_dim=23, hidden_dim=100, z_dim=2, conv_size='mid', dropout=0.5, num_layers=2):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.conv_size = conv_size

        if conv_size == 'small':
            self.conv1 = nn.Conv1d(12, 2, kernel_size=2)
            self.conv2 = nn.Conv1d(2, 3, kernel_size=3)
            self.conv3 = nn.Conv1d(3, 4, kernel_size=4)
            self.linear = nn.Linear(36, hidden_dim)
            self.batch_norm1 = nn.BatchNorm1d(2)
            self.batch_norm2 = nn.BatchNorm1d(3)
            self.batch_norm3 = nn.BatchNorm1d(4)
        elif conv_size == 'mid':
            self.conv1 = nn.Conv1d(39, 128, kernel_size=2)
            self.conv2 = nn.Conv1d(128, 256, kernel_size=3)
            self.conv3 = nn.Conv1d(256, 512, kernel_size=4)
            # self.conv4 = nn.Conv1d(256, 512, kernel_size=5)
            
            self.batch_norm1 = nn.BatchNorm1d(128)
            self.batch_norm2 = nn.BatchNorm1d(256)
            self.batch_norm3 = nn.BatchNorm1d(512)
            # self.batch_norm4 = nn.BatchNorm1d(512)
            
            # Assuming input length is 15 for calculation purposes
            
            length = 80
            length = length - 2 + 1  # conv1 with kernel_size=2
            length = length - 3 + 1  # conv2 with kernel_size=3
            output_length = length - 4 + 1  # conv3 with kernel_size=4
            # output_length = length - 5 + 1  # conv3 with kernel_size=4
            
            self.linear = nn.Linear(output_length * 512, hidden_dim)  # Adjusted for the new size

   
        # elif conv_size == 'large':
        #     self.conv1 = nn.Conv1d(12, 48, kernel_size=2)  # Increased from 24 to 48
        #     self.conv2 = nn.Conv1d(48, 24, kernel_size=3)  # Increased input from 24 to 48, output increased to 24
        #     self.conv3 = nn.Conv1d(24, 24, kernel_size=4)  # Increased input from 12 to 24, output remains 24
        #     self.linear = nn.Linear(216, hidden_dim)       # Adjusted input size accordingly
        #     self.batch_norm1 = nn.BatchNorm1d(48)          # Batch normalization for 48 channels
        #     self.batch_norm2 = nn.BatchNorm1d(24)          # Batch normalization for 24 channels
        #     self.batch_norm3 = nn.BatchNorm1d(24)          # Batch normalization for 24 channels
        else:
            raise ValueError('Invalid value for `conv_size`: {}. Must be in [small, large]'.format(conv_size))
        
        

        # self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply convolutional layers
        h = self.conv1(x)
        h = self.batch_norm1(h)
        h = self.relu(h)
        # h = self.dropout(h)

        h = self.conv2(h)
        h = self.batch_norm2(h)
        h = self.relu(h)
        # h = self.dropout(h)

        h = self.conv3(h)
        h = self.batch_norm3(h)
        h = self.relu(h)
        # h = self.dropout(h)
        # h = self.conv4(h)
        # h = self.batch_norm4(h)
        # h = self.relu(h)

        # Flatten output of convolutions
        h = h.view(x.size(0), -1)

        # Apply linear layer
        h = self.linear(h)
        h = self.relu(h)

        # # Transpose x to match the expected shape for GRU input (batch_first=True)
        # x_gru = x.permute(0, 2, 1)  # Now x_gru is [batch_size, seq_len, input_dim]
        
        # # # Apply GRU
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial hidden state
        # out, _ = self.gru(x_gru, h0)

        # # # Use the last hidden state for encoding
        # h_gru = out[:, -1, :]

        # Compute mu and sigma
        mu = self.mu(h)
        sigma = self.softplus(self.sigma(h))

        return mu, sigma


    
# class Encoder(nn.Module):
#     """
#     Encoder for Grammar VAE with convolutional layers.
#     """
#     def __init__(self, input_dim=12, hidden_dim=100, z_dim=2, conv_size='mid', dropout=0.5, num_layers=2):
#         super(Encoder, self).__init__()
#         self.latent_dim = 2
#         self.max_len = 15

#         self.conv1 = nn.Conv1d(12, 7, 7)
#         self.conv2 = nn.Conv1d(7, 8, 8)
#         self.conv3 = nn.Conv1d(8, 9, 9)

#         self.last_conv_size = 15 - 7 + 1 - 8 + 1 - 9 + 1
#         self.w1 = nn.Linear(self.last_conv_size * 9, hidden_dim)
#         self.mean_w = nn.Linear(hidden_dim, self.latent_dim)
#         self.log_var_w = nn.Linear(hidden_dim, self.latent_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x_cpu):
#         if x_cpu.device.type == 'cpu':
#             batch_input = x_cpu
#         else:
#             batch_input = x_cpu.cuda()

#         h1 = self.conv1(batch_input)
#         h1 = self.relu(h1)
#         h2 = self.conv2(h1)
#         h2 = self.relu(h2)
#         h3 = self.conv3(h2)
#         h3 = self.relu(h3)

#         flatten = h3.view(x_cpu.shape[0], -1)
#         h = self.w1(flatten)
#         h = self.relu(h)

#         z_mean = self.mean_w(h)
#         z_log_var = self.log_var_w(h)
        
#         return z_mean, z_log_var