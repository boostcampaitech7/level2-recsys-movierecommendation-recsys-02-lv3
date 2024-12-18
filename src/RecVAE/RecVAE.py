from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())

class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
          super(Encoder, self).__init__()
          
          self.fc1 = nn.Linear(input_dim, hidden_dim)
          self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
          self.fc2 = nn.Linear(hidden_dim, hidden_dim)
          self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
          self.fc3 = nn.Linear(hidden_dim, hidden_dim)
          self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
          self.fc4 = nn.Linear(hidden_dim, hidden_dim)
          self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
          self.fc5 = nn.Linear(hidden_dim, hidden_dim)
          self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
          self.fc_mu = nn.Linear(hidden_dim, latent_dim)
          self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
          
    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1, keepdim=True).sqrt()
        x = x / norm
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        def swish(x):
            return x.mul(torch.sigmoid(x))
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)

class Compositeprior(nn.Module):
      # mixture of Gaussian prior and latent code distribution with params fixed from previous iteration
        def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
            super(Compositeprior, self).__init__()
            self.mixture_weights = mixture_weights
            
            self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
            self.mu_prior.data.fill_(0)
            
            self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
            self.logvar_prior.data.fill_(0)
            
            self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
            self.logvar_uniform_prior.data.fill_(10)
            
            self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
            self.encoder_old.requires_grad_(False)

        def forward(self,x,z):
            post_mu, post_logvar = self.encoder_old(x, 0)

            stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
            post_prior = log_norm_pdf(z, post_mu, post_logvar)
            unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
            
            gaussians = [stnd_prior, post_prior, unif_prior]
            gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
            
            density_per_gaussian = torch.stack(gaussians, dim=-1)
                    
            return torch.logsumexp(density_per_gaussian, dim=-1)
        
class RecVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(RecVAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder = Encoder(self.hidden_dim, self.latent_dim, self.input_dim)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim)
        self.prior = Compositeprior(self.hidden_dim, self.latent_dim, self.input_dim)

    def forward(self, user_ratings, calculate_loss=True, beta=None, gamma=1, dropout_rate=0.5):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)

        def reparameterize(self, mu, logvar):
          if self.training:
              std = torch.exp(0.5*logvar)
              eps = torch.randn_like(std)
              return eps.mul(std).add_(mu)
          else:
              return mu
        
        z = reparameterize(self, mu, logvar)
        pred = self.decoder(z)

        if calculate_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta
            mll = (F.log_softmax(pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)
            
            return (mll, kld), negative_elbo
            
        else:
            return pred
      
    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))   