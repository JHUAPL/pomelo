import logging
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from harddisc.feature_extraction.reducer import Reducer

logger = logging.getLogger(__name__)


class AddGaussianNoise:
    """
    A class that creates adds Gaussian Noise to a pytorch tensor
    Attributes
    ----------
    mean: float
        mean of gaussian distribution
        optional
        default 0.0
    std: float
        standard deviation away from mean
        optional
        default 1.0
    Methods
    -------
    __call__(tensor: torch.Tensor)
        adds gaussian noise to a pytorch tensor
        output: torch.Tensor
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """Initializes Additive Gaussian Noise class

        Parameters
        ----------
        mean : float, optional
            Mean of Gaussian distribution, by default 0.0
        std : float, optional
            Standard Deviation of Gaussian distribution, by default 1.0
        """
        logger.debug("Initializing AddGaussianNoise")
        self.std = std
        self.mean = mean
        logger.debug("Finished initializing AddGaussianNoise")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Adds gaussian noise to a pytorch tensor

        Parameters
        ----------
        tensor : torch.Tensor
            A tensor that needs to have noise added

        Returns
        -------
        torch.Tensor
            Noised tensor
        """
        return tensor + torch.randn(tensor.size()).to(tensor) * self.std + self.mean


class AutoEncoderProcessorDataset(Dataset):
    """
    A class that wraps the autoencoder dataset into a pytorch usable form

    Attributes
    ----------
    input: torch.Tensor
        required
        tensor of dataset

    Methods
    -------

    __len__(tensor: torch.Tensor)
        gets the length of dataset
        output: int

    __getitem__(index: int)
        returns data at index in dataset
        output: Any
    """

    def __init__(self, input: torch.Tensor):
        """Initializes auto encoder processor dataset

        Parameters
        ----------
        input : torch.Tensor
            Dataset to fit
        """
        logger.debug(
            f"Initializing AutoEncoderProcessorDataset of length: {len(input)}"
        )
        self.input = input
        logger.debug("Finished initializing AutoEncoderProcessorDataset")

    def __len__(self) -> int:
        """Returns length of dataset

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.input)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Retrives item from dataset at index

        Parameters
        ----------
        index : int
            Index to retrieve item in dataset at

        Returns
        -------
        torch.Tensor
            Dataset entry at the index provided
        """
        return self.input[index]


class Encoder(nn.Module):
    """
    A class for the encoder part of an autoencoder

    Attributes
    ----------
    dim_sizes: List[int]
        a list of sizes for each layer of the autoencoder
        must have zero at the beginning to accomdate for different sizes of inputs
    hidden: List[nn.Module]
        list of pytorch layers
    model: nn.Sequential
        model in Sequential form meaning it can be used to run data through it

    Methods
    -------
    __init__(dim_sizes: List[int])
        builds the encoder with the sizes of the list provided
        each layer is Linear->ReLu
    forward(x: torch.Tensor)
        puts data through encoder
        output: torch.Tensor
    """

    def __init__(self, dim_sizes: List[int]) -> None:
        """Initializes autoencoder encoder half

        Parameters
        ----------
        dim_sizes : List[int]
            List of hidden dimension sizes for encoder
        """
        super().__init__()
        logger.debug(f"Initializing Autoencoder Encoder")
        self.dim_sizes = dim_sizes

        self.hidden: List[nn.Module] = []
        # loops through list of sizes of layers and creates linear layer relu combos
        # ex. [300, 200, 100] would make a 2 layer neural net with layer 300->200 relu 200->100 relu
        for k in range(len(self.dim_sizes) - 1):
            self.hidden.append(nn.Linear(self.dim_sizes[k], self.dim_sizes[k + 1]))
            self.hidden.append(nn.ReLU())

        self.model = nn.Sequential(*self.hidden)

        logger.debug(f"Finished initializing Autoencoder Encoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Puts data through encoder

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Transformed output tensor
        """
        return self.model(x)

    def change_device(self, device: str) -> None:
        """Moves the model to the device specified

        Parameters
        ----------
        device : str
            Device to move the model to
        """
        self.model.to(device)


class Decoder(nn.Module):
    """
    A class for the decoder part of an autoencoder or a variational autoencoder

    Attributes
    ----------
    dim_sizes: List[int]
        a list of sizes for each layer of the autoencoder
        must have zero at the end to accomdate for different sizes of inputs
    hidden: List[nn.Module]
        list of pytorch layers
    model: nn.Sequential
        model in Sequential form meaning it can be used to run data through it

    Methods
    -------
    __init__(dim_sizes: List[int])
        builds the decoder with the sizes of the list provided
        each layer is Linear->ReLu
    forward(x: torch.Tensor)
        puts data through decoder
        output: torch.Tensor
    """

    def __init__(self, dim_sizes: List[int]) -> None:
        """Initializes autoencoder decoder half

        Parameters
        ----------
        dim_sizes : List[int]
            List of hidden dimension sizes for decoder
        """
        super().__init__()
        logger.debug(f"Initializing Autoencoder Decoder")
        self.dim_sizes = dim_sizes

        self.hidden: List[nn.Module] = []
        # loops through list of sizes of layers and creates linear layer relu combos except at end where there is not relu
        # ex. [100, 200, 300] would make a 2 layer neural net with layer 100->200 relu 200->300
        for k in range(len(self.dim_sizes) - 1):
            self.hidden.append(nn.Linear(self.dim_sizes[k], self.dim_sizes[k + 1]))
            if k != len(dim_sizes) - 2:
                self.hidden.append(nn.ReLU())

        self.model = nn.Sequential(*self.hidden)

        logger.debug(f"Finished initializing Autoencoder Decoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Puts data through decoder

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Transformed output tensor
        """
        return self.model(x)

    def change_device(self, device: str) -> None:
        """Moves the model to the device specified

        Parameters
        ----------
        device : str
            Device to move the model to
        """
        self.model.to(device)


class VariationalEncoder(nn.Module):
    """
    A class for the variational encoder part of an variational autoencoder

    Attributes
    ----------
    dim_sizes: List[int]
        a list of sizes for each layer of the autoencoder
        must have zero at the end to accomdate for different sizes of inputs
    encoder_mean: nn.Linear
        linear layer to take data from encoder and make it the mean of the gaussian
    encoder_variance: nn.Linear
        linear layer to take data from encoder and make it the variance of the gaussian
    distrib: torch.distributions.Normal
        normal distribution to act as the variational part of the encoder
        mean 0 std 1
    kl: float
        KL divergence: used for loss calculations

    Methods
    -------
    __init__(dim_sizes: List[int])
        builds the variational encoder with the sizes of the list provided
        each layer is Linear->ReLu
    forward(x: torch.Tensor)
        puts data through variational encoder
        output: torch.Tensor
    """

    def __init__(self, dim_sizes: List[int]) -> None:
        """Initializes variational autoencoder encoder half

        Parameters
        ----------
        dim_sizes : List[int]
            List of hidden dimension sizes for variational autoencoder encoder
        """
        super().__init__()
        logger.debug(f"Initializing Autoencoder VariationalEncoder")
        self.dim_sizes = dim_sizes

        # loops through list of sizes of layers and creates linear layer relu combos
        # ex. [300, 200, 100] would make a 2 layer neural net with layer 300->200 relu 200->100 relu

        self.hidden: List[nn.Module] = []
        for k in range(len(self.dim_sizes) - 1):
            self.hidden.append(nn.Linear(self.dim_sizes[k], self.dim_sizes[k + 1]))
            self.hidden.append(nn.ReLU())

        self.model = nn.Sequential(*self.hidden)

        # variational part of encoder
        # creates multivariate gaussian using the output of the model

        # creates linear
        self.encoder_mean = nn.Linear(
            in_features=self.dim_sizes[-1], out_features=self.dim_sizes[-1]
        )

        self.encoder_variance = nn.Linear(
            in_features=self.dim_sizes[-1], out_features=self.dim_sizes[-1]
        )

        # creates a normal distribution to sample from

        self.distrib = torch.distributions.Normal(0, 1)

        # used for loss

        self.kl: torch.Tensor

        logger.debug(f"Finished initializing Autoencoder VariationalEncoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Puts data through variational autoencoder

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor

        Returns
        -------
        torch.Tensor
            Transformed output tensor
        """
        # get the output from encoder
        activation = self.model(x)

        # put it into our layers to get the mean and variation
        mu = self.encoder_mean(activation)
        sigma = torch.exp(self.encoder_variance(activation))

        # reparameterization trick
        z = mu + sigma * self.distrib.sample(mu.shape).to(mu)

        # calculate loss
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

        return z

    def change_device(self, device: str) -> None:
        """Moves the model to the device specified

        Parameters
        ----------
        device : str
            Device to move the model to
        """
        self.model.to(device)
        self.encoder_mean.to(device)
        self.encoder_variance.to(device)


class AutoEncoderProcessor(nn.Module, Reducer):
    """
    A class to wrap encoder and decoder parts of an autoencoder

    Attributes
    ----------
    encoder_dim_size: List[int]
        a list of sizes for each layer of the autoencoder
        must have zero at the beginning to accomdate for different sizes of inputs

    decoder_dim_size: List[int]
        a list of sizes for each layer of the autoencoder
        must have zero at the end to accomdate for different sizes of inputs

    noise: bool
        switch to have the noise added at the beginning of the model

    variational: bool
        switch to have the encoder be variational

    encoder: Union[Encoder, VariationalEncoder]
        built encoder or variationalencoder class

    decoder: Decoder
        built decoder class

    GaussianNoise: AddGaussianNoise
        built class for adding noise

    Methods
    -------
    __init__(encoder_dim_size: List[int], decoder_dim_size: List[int],noise: bool, variational: bool)
        builds the autoencoder with parameters described

        encoder_dim_size: List[int]
            a list of sizes for each layer of the autoencoder
            must have zero at the beginning to accomdate for different sizes of inputs

        decoder_dim_size: List[int]
            a list of sizes for each layer of the autoencoder
            must have zero at the end to accomdate for different sizes of inputs

        noise: bool
            switch to have the noise added at the beginning of the model

        variational: bool
            switch to have the encoder be variational
    reduce(self, data: np.ndarray)
        Reduces dimensionality of 2d numpy array using a/an AE or VAE
        output: np.ndarray
    forward(x: torch.Tensor)
        puts data through autoencoder
        output: torch.Tensor
    compute_l1_loss(w: torch.Tensor)
        computes the l1 loss
        output: float
    train_autoencoder(data: np.ndarray, epochs: int, batch_size: int, lr: float, l2_regularization: float, l1_regularization: float)
        trains the autoencoder
    encode_dataset(data: np.ndarray, batch_size: int)
        puts data through the encoder portion to use as features
    """

    def __init__(
        self,
        encoding_layers: List[int],
        decoding_layers: List[int],
        train_batch_size: int = 10,
        dev_batch_size: int = 10,
        epochs: int = 100,
        noise: bool = False,
        variational: bool = False,
        lr: float = 0.001,
        l2: float = 0,
        l1: float = 1.0,
    ) -> None:
        """Initializes autoencoder module

        Parameters
        ----------
        encoding_layers : List[int]
            List of hidden dimension sizes for autoencoder encoder
        decoding_layers : List[int]
            List of hidden dimension sizes for autoencoder decoder
        train_batch_size : int, optional
            Size of batches during training, by default 10
        dev_batch_size : int, optional
            Size of batches during dev step, by default 10
        epochs : int, optional
            Number of epochs to train the model, by default 100
        noise : bool, optional
            Whether to make it a noised autoencoder, by default False
        variational : bool, optional
            Whether the model will be variational, by default False
        lr : float, optional
            Learning rate during training, by default 0.001
        l2 : float, optional
            L2/weight decay penalty, by default 0
        l1 : float, optional
            L1 penalty, by default 1.0

        Raises
        ------
        ValueError
            Encoder and decoder latent dimensions must match
        """
        super().__init__()
        logger.debug(f"Initializing AutoEncoderProcessor")
        if encoding_layers is None:
            logger.warning(f"No encoding layers provided using default [0, 300, 200]")
            encoding_layers = [0, 300, 200]

        if decoding_layers is None:
            logger.warning(f"No decoding layers provided using default [200, 300, 0]")
            decoding_layers = [200, 300, 0]

        self.encoder_dim_size = encoding_layers
        self.decoder_dim_size = decoding_layers

        if self.encoder_dim_size[-1] != self.decoder_dim_size[0]:
            raise ValueError(  # noqa: TRY003
                f"Encoder and Decoder Latent Dims must match! {self.encoder_dim_size[-1]} != {self.decoder_dim_size[0]}"
            )

        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size

        self.epochs = epochs

        self.lr = lr
        self.l2 = l2
        self.l1 = l1

        self.noise = noise
        self.variational = variational

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if self.noise:
            logger.debug(f"Using Noised AutoEncoder")
            self.GaussianNoise = AddGaussianNoise()

        self.encoder: Union[VariationalEncoder, Encoder]
        if self.variational:
            logger.debug(f"Using VariationalEncoder")
            self.encoder = VariationalEncoder(self.encoder_dim_size)
        else:
            logger.debug(f"Using regular AutoEncoder encoder")
            self.encoder = Encoder(self.encoder_dim_size)

        self.decoder = Decoder(self.decoder_dim_size)

        self.encoder.change_device(self.device)

        self.decoder.change_device(self.device)

        logger.debug(f"Finished initializing AutoEncoderProcessor")

    def reduce(self, data: np.ndarray) -> np.ndarray:
        """Reduces dimensionality of 2d numpy array using a/an AE or VAE

        Parameters
        ----------
        data : np.ndarray
            2d array of data each row representing one data point

        Returns
        -------
        np.ndarray
            Reduced representation of 2d array by AE or VAE
        """
        self.encoder.model[0] = nn.Linear(data.shape[1], self.encoder.dim_sizes[1]).to(
            self.device
        )

        self.decoder.model[-1] = nn.Linear(
            self.decoder.dim_sizes[-2], data.shape[1]
        ).to(self.device)

        self.train_autoencoder(data)

        return self.encode_dataset(data, self.dev_batch_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Passes data through autoencoder

        Parameters
        ----------
        features : torch.Tensor
            Input Tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if self.noise:
            features = self.GaussianNoise(features)

        encoded = self.encoder(features)

        return self.decoder(encoded)

    def compute_l1_loss(self, w: torch.Tensor) -> torch.Tensor:
        """Computes the L1 loss which is the sum of the absolute values of the weights

        Parameters
        ----------
        w : torch.Tensor
            Weights

        Returns
        -------
        torch.Tensor
            L1 penalty
        """
        return torch.abs(w).sum()

    def train_autoencoder(
        self,
        data: np.ndarray,
    ) -> None:
        """Trains the autoencoder with adam optimizer and MSE loss

        Parameters
        ----------
        data : np.ndarray
            input features as a 2d array
        """

        logger.debug(f"Training AutoEncoder with {len(data)} instances")
        # convert to data to tensor
        tensor_data = torch.Tensor(data)

        # load dataset into torch readable format
        dataset = AutoEncoderProcessorDataset(tensor_data)

        dataloader = DataLoader(dataset, batch_size=self.train_batch_size)

        # creates adam optimization
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

        # mean-squared error loss
        criterion = nn.MSELoss()

        # training loop
        for epoch in range(self.epochs):
            loss: float = 0.0
            for i, batch_features in enumerate(dataloader):
                # zeros optimizer
                optimizer.zero_grad()

                batch_features = batch_features.to(self.device)

                # passes data through model
                outputs = self(batch_features)

                # calculates loss between the output and the input
                # autoencoders are unsupervised!
                train_loss = criterion(outputs, batch_features)

                # calculates the l1 loss
                l1: torch.Tensor = torch.zeros(1)
                if self.l1 != 1.0:
                    l1_parameters = []
                    for parameter in self.parameters():
                        l1_parameters.append(parameter.view(-1))
                    l1 = self.l1 * self.compute_l1_loss(torch.cat(l1_parameters))

                # variational autoencoders need to have kl divergence because they use a distribution
                # kl helps guide the variational autoencoder to the "ideal" distribution
                if self.variational:
                    train_loss += self.encoder.kl.cpu()

                # add the l1
                train_loss += l1.item()

                # step through

                train_loss.backward()

                optimizer.step()

                loss += train_loss.item()

                if i % 10 == 0 and i != 0:
                    logging.info(
                        f"Training autoencoder epoch {epoch+1}: {i}/{len(dataloader)} batches complete"
                    )

            loss = loss / len(dataloader)

            logging.info(f"epoch : {epoch + 1}/{self.epochs}, loss = {loss:.6f}")

        logger.debug(f"Finished training AutoEncoder")

    def encode_dataset(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """Passes data through trained encoder to use as features

        Parameters
        ----------
        data : np.ndarray
            Input dataset
        batch_size : int
            Size of batches to pass through the

        Returns
        -------
        np.ndarray
            Transformed dataset
        """
        logging.debug(f"Reducing {len(data)} instances with AutoEncoder")
        # convert to data to tensor
        tensor_data = torch.Tensor(data)

        # load dataset into torch readable format
        dataset = AutoEncoderProcessorDataset(tensor_data)

        dataloader = DataLoader(dataset, batch_size=batch_size)

        # pass dataset through trained encoder
        encoded_features = []
        self.eval()

        with torch.no_grad():
            for i, batch_features in enumerate(dataloader):
                batch_features = batch_features.to(self.device)
                outputs = self.encoder(batch_features)
                encoded_features.append(outputs)

                if i % 10 == 0 and i != 0:
                    logging.info(
                        f"Prediction with AutoEncoder: {i}/{len(dataloader)} batches complete"
                    )

        logging.debug(f"Finished reducing with AutoEncoder")

        output = torch.cat(encoded_features).detach().cpu().numpy()

        return output
