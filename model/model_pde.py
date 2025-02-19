import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal, Categorical
from nltk import Nonterminal
from encoder_pde import Encoder
from decoder_pde import Decoder
from stack import Stack
from grammar_pde import GCFG, S, get_mask

class GrammarVAE(nn.Module):
    """Grammar Variational Autoencoder"""
    def __init__(self, hidden_encoder_size, z_dim, hidden_decoder_size, output_size, rnn_type, max_length, dropout=0.5, tightness_param=1e-3, num_layers=3):
        super(GrammarVAE, self).__init__()
        self.encoder = Encoder(input_dim=19, hidden_dim=hidden_encoder_size, z_dim=z_dim, num_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(latent_rep_size=z_dim, hidden_size=hidden_decoder_size, output_size=output_size, max_length=max_length, rnn_type=rnn_type, num_layers=num_layers, dropout=dropout)
        self.tightness_param = tightness_param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def sample(self, mu, sigma):
        """Reparametrized sample from a N(mu, sigma) distribution"""
        device = mu.device
        normal = Normal(torch.zeros(mu.shape).to(device), torch.ones(sigma.shape).to(device))
        eps = Variable(normal.sample()).to(device)
        z = mu + eps * torch.sqrt(sigma)*self.tightness_param
        return z

    def kl(self, mu, sigma):
        """KL divergence between two normal distributions"""
        return torch.mean(-0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), 1))

    def forward(self, x):
        x = x.to(self.device)
        mu, sigma = self.encoder(x)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        z = self.sample(mu, sigma)
        logits = self.decoder(z.to(self.device))
        return logits

    def generate(self, z, sample=False, max_length=15):
        """Generate a valid expression from z using the decoder"""
        stack = Stack(grammar=GCFG, start_symbol=S)
        logits = self.decoder(z).squeeze().to(self.device)
        rules = []
        t = 0
        # print(max_length)
        while stack.nonempty:
            
            alpha = stack.pop()
            # print('alpha',alpha)
            mask = (get_mask(alpha, stack.grammar, as_variable=True)).to(self.device)
            # print('mask',mask)
            # print(t)
            # print(logits[t])
            
            probs = mask * logits[t].exp()
            probs = probs / probs.sum()
            if sample:
                m = Categorical(probs)
                i = m.sample()
            else:
                _, i = probs.max(-1)  # argmax
            # convert PyTorch Variable to regular integer
            i = i.item()  # changed to `.item()` for compatibility with newer PyTorch versions
            # select rule i
            rule = stack.grammar.productions()[i]
            rules.append(rule)
            # add rhs nonterminals to stack in reversed order
            for symbol in reversed(rule.rhs()):
                if isinstance(symbol, Nonterminal):
                    stack.push(symbol)
            t += 1
            # print('t',t)
            if t == max_length:
                if len(stack) !=0 :
                    rules = None
                break
        return rules

   