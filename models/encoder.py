import os
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import numpy as np
from joblib import dump, load
# from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from time import time


class Encoder():
    def __init__(self, n_components):
        self.n_components = n_components
        pass

    def train(self, dataset):
        pass

    def get_latent_space(self, dataset):
        pass


class TBVAE(nn.Module):
    """
    Variational Autoencoder for translating turbulent flow fields into a latent space. Such latent space is then used to classify the flow fields by their characteristics.
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers=3, dropout=0.5, **kwargs):
        super(TBVAE, self).__init__() 
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        for _ in range(num_layers - 1):
            self.encoder.add_module(
                f'hidden_{_}',
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)

        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        for _ in range(num_layers - 1):
            self.decoder.add_module(
                f'hidden_{_}',
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.decoder.add_module(
            'output',
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.encoder_mu(h), self.encoder_logvar(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class PCAEncoder(Encoder):
    def __init__(self, n_components, **kwargs):
        super(PCAEncoder, self).__init__(n_components)
        self.model = PCA(n_components=n_components)

    def train(self, dataset, save_model=False, path=None):
        # print(type(dataset[0]))
        # split training data by graph and matrix format and call the appropriate training function
        # if dataset is a torch dataset, send to _train_matrix, if dataset is a torch_geometric InMemoryDataset, send to _train_graph
        # if type(dataset[0][0]) == list:
        self._train_graph(dataset, save_model, path)
        # else:
        #     self._train_graph(dataset, save_model, path)

    def _train_graph(self, dataset, save_model=False, path=None):
        data_space = []
        for data in dataset:
            x = data.x.cpu().detach().numpy()
            data_space.append(x)

        # ensure all data has the same shape by cutting the data length to the minimum length
        min_length = min(data.shape[0] for data in data_space)
        # min_length = 562
        print(f'Min length: {min_length}')
        data_space = [data[:min_length, :].reshape(-1) for data in data_space]
        print(f'PCA input shape: {np.array(data_space).shape}')
        # print(f'Data space shape: {np.array(data_space).shape}')
        self.model.fit(np.array(data_space))
        if save_model:
            self._save_model(path)

    def _train_matrix(self, dataset, save_model=False, path=None):
        data_space = []
        for data in dataset:
            try:
                x = data[0][0][:, :, 0]
            except:
                x, y = data
            data_space.append(x.cpu().detach().numpy().reshape(-1))
        self.model.fit(np.array(data_space))
        # dump(self.model, 'logs/models/collection_fno_jhtdb/pca_encoder.joblib')
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'pca_encoder.joblib'))

    def get_latent_space(self, dataset):
        data_space = []
        for data in dataset:
            x = data.x.cpu().detach().numpy()

            data_space.append(x)

        # ensure all data has the same shape by cutting the data length to the minimum length
        min_length = min(data.shape[0] for data in data_space)
        min_length = 280
        print(f'Min length: {min_length}')
        data_space = [data[:min_length, :].reshape(-1) for data in data_space]
        print(f'PCA input shape: {np.array(data_space).shape}')
        latent_space = self.model.transform(np.array(data_space))
        return latent_space
    
    def load_model(self, path):
        self.model = load(os.path.join(path, 'pca_encoder.joblib'))
    

class VAEEncoder(Encoder):
    def __init__(self, n_components, **kwargs):
        super(VAEEncoder, self).__init__(n_components)
        self.model = TBVAE(input_dim=kwargs['input_dim'], latent_dim=kwargs['n_components'], hidden_dim=128, num_layers=kwargs['num_layers'], dropout=kwargs['dropout'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, dataset):
        def loss_function(x_hat, x, mu, logvar):
            BCE = nn.functional.mse_loss(x_hat, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return BCE + KLD
        
        device = self.device
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        for data in dataset:
            x, y = data
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = self.model(x)
            loss = loss_function(x_hat, x, mu, logvar)
            loss.backward()
            optimizer.step()

    def get_latent_space(self, dataset):
        device = self.device
        self.model.to(device)
        self.model.eval()
        latent_space = []
        with torch.no_grad():
            for data in dataset:
                x, y = data
                x = x.to(device)
                mu, logvar = self.model.encode(x)
                z = self.model.reparameterize(mu, logvar)
                latent_space.append(z.cpu().detach().numpy())
        return np.array(latent_space)
    

class SpectrumEncoder(Encoder):
    def __init__(self, n_components, **kwargs):
        super(SpectrumEncoder, self).__init__(n_components)
        self.n_components = n_components
        self.domain_size = kwargs['domain_size']

    def train(self, dataset, save_model=False, path=None):
        pass
    
    @staticmethod
    def _compute_tke_spectrum(u):
        """
        Given velocity fields u and v, computes the turbulent kinetic energy spectrum. The function computes in three steps:
        1. Compute velocity spectrum with fft, returns uf, vf.
        2. Compute the point-wise turbulent kinetic energy Ef=0.5*(uf, vf)*conj(uf, vf).
        3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy 
        Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
        the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
        E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).
        """
        u, lx, ly = u
        # check if u is a data that contains x and y or just x
        if len(u) == 2:
            u = u[0]
        # print(u.shape)
        nx = u.shape[0]
        ny = u.shape[1]

        nt = nx * ny
        # Compute velocity spectrum
        if len(u.shape) == 2:
            uf = np.fft.fft2(u, axes=(0, 1))
        elif len(u.shape) == 3:
            uf = np.fft.fft2(u[:, :, 0].unsqueeze(-1), axes=(0, 1))
        else:
            print(u.shape)
            raise ValueError('Invalid input shape')

        # Compute the point-wise turbulent kinetic energy
        Ef = 0.5 * (uf * np.conj(uf)).real
        # kx = 2 * np.pi / lx 
        # ky = 2 * np.pi / ly
        # knorm = np.sqrt(kx ** 2 + ky ** 2)
        kxmax = nx / 2
        kymax = ny / 2
        # wave_numbers = knorm * np.arange(0, nx)
        tke_spectrum = np.zeros(nx)
        for i in range(nx):
            rkx = i
            if i > kxmax:
                rkx = rkx - nx
            for j in range(ny):
                rky = j
                if j > kymax:
                    rky = rky - ny
                rk = np.sqrt(rkx * rkx + rky * rky)
                k_index = int(np.round(rk))
                tke_spectrum[k_index] += Ef[i, j]
        # k = torch.fft.fftfreq(nx, lx / nx)

        # plt.loglog(wave_numbers[1:], tke_spectrum[1:])
        # plt.savefig('tke_spectrum.png')
        # normalize the spectrum between 0 and 1
        tke_spectrum = np.log(tke_spectrum[1:] + 1e-8)
        tke_spectrum = (tke_spectrum - np.min(tke_spectrum)) / (np.max(tke_spectrum) - np.min(tke_spectrum))
        return tke_spectrum
    
    @staticmethod
    def _compute_tke_spectrum_3d(points, physics, grid_resolution):
        """
        Computes the turbulent kinetic energy spectrum from sparse 3D data.

        Args:
            points (np.ndarray): Array of shape [num_points, 3] with 3D coordinates (x, y, z).
            physics (np.ndarray): Array of shape [num_points, 1] with physics data (e.g., velocity magnitude).
            grid_resolution (tuple): Resolution of the regular grid (nx, ny, nz).
        
        Returns:
            np.ndarray: Normalized 1D TKE spectrum.
        """
        # 1. Define a regular grid
        x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), grid_resolution[0])
        y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), grid_resolution[1])
        z = np.linspace(np.min(points[:, 2]), np.max(points[:, 2]), grid_resolution[2])
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

        # 2. Interpolate sparse data onto the regular grid
        grid_values = griddata(points, physics.flatten(), (grid_x, grid_y, grid_z), method='linear', fill_value=0)

        # 3. Compute 3D FFT
        uf = np.fft.fftn(grid_values, axes=(0, 1, 2))

        # 4. Compute point-wise turbulent kinetic energy spectrum
        Ef = 0.5 * (uf * np.conj(uf)).real

        # 5. Integrate over spherical shells in 3D
        nx, ny, nz = grid_resolution
        kxmax, kymax, kzmax = nx // 2, ny // 2, nz // 2
        tke_spectrum = np.zeros(nx // 2)  # Only need half due to symmetry

        for i in range(nx):
            rkx = i - nx if i > kxmax else i
            for j in range(ny):
                rky = j - ny if j > kymax else j
                for k in range(nz):
                    rkz = k - nz if k > kzmax else k
                    rk = np.sqrt(rkx**2 + rky**2 + rkz**2)
                    k_index = int(np.round(rk))
                    if k_index < len(tke_spectrum):
                        tke_spectrum[k_index] += Ef[i, j, k]

        # 6. Normalize spectrum
        tke_spectrum = np.log(tke_spectrum[1:] + 1e-8)
        tke_spectrum = (tke_spectrum - np.min(tke_spectrum)) / (np.max(tke_spectrum) - np.min(tke_spectrum))

        return tke_spectrum
    
    def get_latent_space(self, dataset):
        # determine if the dataset is a torch matrix dataset or a torch_geometric dataset
        # if type(dataset[0][0]) == list:
        try:
            dataset = [[data[0][0][:, :, 0].numpy(), self.domain_size, self.domain_size] for data in dataset]
        except:
            dataset = [[data[0].numpy(), self.domain_size.numpy(), self.domain_size] for data in dataset]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            latent_space = list(executor.map(self._compute_tke_spectrum, dataset))

        # else:
        #     latent_space = []
        #     for data in dataset:
        #         points = data.pos.cpu().detach().numpy()
        #         physics = data.y.cpu().detach().numpy()
        #         latent_space.append(self._compute_tke_spectrum_3d(points, physics, (self.domain_size, self.domain_size, self.domain_size)))

        return np.array(latent_space)

    def get_latent(self, x):
        # determine if the data object is a torch matrix or a torch_geometric object
        # if type(x) == tuple:

        # print('Computing latent space')
        # time_start = time()
        x = [(data, self.domain_size, self.domain_size) for data in x]
        # domain_size_list = [self.domain_size for _ in range(len(x))]
        # time_sep = time()
        # print(f'Time to separate data: {time_sep - time_start}')
        with ProcessPoolExecutor(max_workers=16) as executor:
            latent_space = list(executor.map(self._compute_tke_spectrum, x))
            # latent_space = p.starmap(self._compute_tke_spectrum, zip(x, domain_size_list, domain_size_list))
        # time_end = time()
        # print(f'Time to compute latent space: {time_end - time_sep}')
        # else:
        #     latent_space = []
        #     for data in x:
        #         points = data.pos.cpu().detach().numpy()
        #         physics = data.y.cpu().detach().numpy()
        #         latent_space.append(self._compute_tke_spectrum_3d(points, physics, (self.domain_size, self.domain_size, self.domain_size)))
        return np.array(latent_space)

    def load_model(self, path):
        pass


class DMDEncoder(Encoder):
    """
    Employs the Dynamic Mode Decomposition to extract the latent space of the flow fields.
    """
    def __init__(self, n_components, **kwargs):
        super(DMDEncoder, self).__init__(n_components)
        self.n_components = n_components

    def train(self, dataset, save_model=False, path=None):
        pass
    
    def get_latent_space(self, dataset):
        pass

    def get_latent(self, x):
        pass

    def load_model(self, path):
        pass