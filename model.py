import torch
from torch import nn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

from tqdm import tqdm

class VAE(nn.Module):
    def __init__(self, input_size, z_dim, hidden_dim):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.hidden_linear = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            # nn.Softplus(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Softplus()
            nn.ReLU(),
        )
        self.z_mu = nn.Linear(hidden_dim, z_dim)
        self.z_logvar = nn.Linear(hidden_dim, z_dim)
        self.x_mu = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            # nn.Softplus(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Softplus(),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
            nn.Sigmoid()
        )

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        bs = x.shape[0]

        x = self.hidden_linear(x)
        mu = self.z_mu(x)
        logvar = self.z_logvar(x)

        epsilon = torch.randn((bs, self.z_dim), device=x.device)
        z_sampled = mu + torch.exp(0.5 * logvar) * epsilon

        return z_sampled, mu, logvar
    
    def decode(self, z):
        return self.x_mu(z)
    
    def calc_elbo(self, x, z, z_mu, z_logvar):
        # version 1 Gaussian assumption, X ~ N(mu, sigma)
        x_recon = self.decode(z)

        # log_likelihood, reconstruction loss. Omit constant item.
        # log_likelihood = - torch.sum(0.5 * (x - mu) ** 2 / (sigma ** 2 + 1e-8), dim=-1) \
        #                  - 0.5 * torch.sum(torch.log(sigma ** 2 + 1e-8), dim=-1)
        log_likelihood = - torch.sum((x - x_recon) ** 2, dim=-1)
 
        # Actually, if we think sigma is a constant, this will become a MSE reconstruction loss.
        # KL(N(mu, sigma^2)|| N(0,1)) = -0.5 \sum (1 + 2log(mu) - mu^2 - sigma^2)
        kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mu ** 2 - z_logvar.exp(), dim=-1)

        log_likelihood = log_likelihood.mean()
        kl_divergence = kl_divergence.mean()
        # print(f'{log_likelihood=}')
        # print(f'{kl_divergence=}')
        
        elbo = log_likelihood - kl_divergence
        return elbo
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        z, mu, logvar = self.encode(x)
        elbo = self.calc_elbo(x, z, mu, logvar)
        loss = -elbo
        return loss
    
    
def build_svm(kernel, C):
    svm_model = SVC(kernel=kernel, C=C, gamma='scale')
    return svm_model

class SVMPipeline():
    def __init__(self, vae, svm, device) -> None:
        self.vae = vae
        self.svm = svm
        self.device = device

    def extract_feature(self, data):
        Z = []
        X = []
        Y = []

        self.vae = self.vae.to(self.device)

        # extract feature
        with torch.no_grad():
            for i in tqdm(range(len(data)), desc="Extracting latent representation"):
                input, label = data[i]  # `inputs` is `x` (features) and `labels` is `y`
                input = input.to(self.device)
                x = input.flatten()
                _, z, _ = self.vae.encode(x.unsqueeze(0)) # tensor [1, c]
                z_np = z[0].cpu().numpy()
                x_np = x.cpu().numpy()
                X.append(x_np)
                Z.append(z_np)
                Y.append(label)
        
        return Z, X, Y

    def train_and_eval_list(self, train_list, eval_set):
        # extract feature for eval
        z_eval, x_eval, y_eval = self.extract_feature(eval_set)
        for train_data in train_list:
            z_train, x_train, y_train = self.extract_feature(train_data)

            self.svm.fit(z_train, y_train)
            y_pred = self.svm.predict(z_eval)
            accuracy = accuracy_score(y_eval, y_pred)
            f1 = f1_score(y_eval, y_pred, average='macro')
            print(f"{len(train_data)} samples: VAE+SVM accuracy: {accuracy}")
            print(f"{len(train_data)} samples: VAE+SVM F1 score: {f1}")

            self.svm.fit(x_train, y_train)
            y_pred = self.svm.predict(x_eval)
            accuracy = accuracy_score(y_eval, y_pred)
            f1 = f1_score(y_eval, y_pred, average='macro')
            print(f"{len(train_data)} samples: SVM accuracy: {accuracy}")
            print(f"{len(train_data)} samples: SVM F1 score: {f1}")
            


    def predict(self, image):
        self.vae = self.vae.to(self.device)
        
        image = image.to(self.device)
        with torch.no_grad():
            z = self.vae.encode(image.flatten().unsqueeze(0))
            z_np = z[0].cpu().numpy()

        # Use the trained SVM to predict the label
        label = self.svm.predict([z_np])
        return label
    
if __name__ == '__main__':
    input_size = 28 * 28
    z_dim = 50
    hidden_dim = 600

    vae = VAE(input_size, z_dim, hidden_dim)
    x = torch.randn((128, 28, 28))
    loss = vae(x)
    print(f"{loss=}")