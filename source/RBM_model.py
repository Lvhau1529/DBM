import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import time
import torch.optim as opt
import torch.nn.functional as F

from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, Optional, Tuple

class Model(torch.nn.Module):
    """The Model class is the basis for any custom model.
    One can configure, if necessary, different properties or methods that
    can be used throughout all childs.
    """

    def __init__(self, use_gpu: Optional[bool] = False) -> None:
        """Initialization method.
        Args:
            use_gpu: Whether GPU should be used or not.
        """

        super(Model, self).__init__()

        self.device = "cpu"
        if torch.cuda.is_available() and use_gpu:
            self.device = "cuda"

        self.history = {}

        # Sets default tensor type to float to avoid value errors
        torch.set_default_tensor_type(torch.FloatTensor)


    @property
    def device(self) -> str:
        """Indicates which device is being used for computation."""

        return self._device

    @device.setter
    def device(self, device: str) -> None:
        if device not in ["cpu", "cuda"]:
            raise e.TypeError("`device` should be `cpu` or `cuda`")

        self._device = device

    @property
    def history(self) -> Dict[str, Any]:
        """Dictionary containing historical values from the model."""

        return self._history

    @history.setter
    def history(self, history: Dict[str, Any]) -> None:
        self._history = history

    def dump(self, **kwargs) -> None:
        """Dumps any amount of keyword documents to lists in the history property."""

        for k, v in kwargs.items():
            if k not in self.history.keys():
                self.history[k] = []

            self.history[k].append(v)


class RBM(Model):
    def __init__(
        self,
        n_visible: Optional[int] = 128,
        n_hidden: Optional[int] = 128,
        steps: Optional[int] = 1,
        learning_rate: Optional[float] = 0.1,
        momentum: Optional[float] = 0.0,
        decay: Optional[float] = 0.0,
        temperature: Optional[float] = 1.0,
        use_gpu: Optional[bool] = False,
    ) -> None:

        super(RBM, self).__init__(use_gpu=use_gpu)

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.steps = steps
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.T = temperature

        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))

        self.optimizer = opt.SGD(
            self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay
        )

        if self.device == "cuda":
            self.cuda()

    def hidden_sampling(
        self, v: torch.Tensor, scale: Optional[bool] = False
    ) -> torch.Tensor:
        activations = F.linear(v, self.W.t(), self.b)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def visible_sampling(
        self, h: torch.Tensor, scale: Optional[bool] = False
    ) -> torch.Tensor:
        activations = F.linear(h, self.W, self.a)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def gibbs_sampling(
        self, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v)
        neg_hidden_states = pos_hidden_states

        # Performing the Contrastive Divergence
        for _ in range(self.steps):
            _, visible_states = self.visible_sampling(neg_hidden_states, True)
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, True
            )

        return (
            pos_hidden_probs,
            pos_hidden_states,
            neg_hidden_probs,
            neg_hidden_states,
            visible_states,
        )

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        activations = F.linear(samples, self.W.t(), self.b)

        # Creates a Softplus function for numerical stability
        s = nn.Softplus()

        h = torch.sum(s(activations), dim=1)
        v = torch.mv(samples, self.a)

        energy = -v - h

        return energy

    def pseudo_likelihood(self, samples: torch.Tensor) -> torch.Tensor:
        # Calculates the energy of samples before flipping the bits
        samples_binary = torch.round(samples)
        energy = self.energy(samples_binary)

        # Samples an array of indexes to flip the bits
        indexes = torch.randint(
            0, self.n_visible, size=(samples.size(0), 1), device=self.device
        )
        bits = torch.zeros(samples.size(0), samples.size(1), device=self.device)
        bits = bits.scatter_(1, indexes, 1)

        # Calculates the energy after flipping the bits
        samples_binary = torch.where(bits == 0, samples_binary, 1 - samples_binary)
        energy1 = self.energy(samples_binary)

        # Calculate the logarithm of the pseudo-likelihood
        pl = torch.mean(
            self.n_visible * torch.log(torch.sigmoid(energy1 - energy) + 1e-10)
        )

        return pl

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 10,
    ) -> Tuple[float, float]:

        batches = dataset

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))

            start = time.time()

            mse, pl = 0, 0

            for samples, _ in tqdm(batches):
                samples = samples.reshape(len(samples), self.n_visible)
                if self.device == "cuda":
                    samples = samples.cuda()

                _, _, _, _, visible_states = self.gibbs_sampling(samples)
                visible_states = visible_states.detach()

                cost = torch.mean(self.energy(samples)) - torch.mean(
                    self.energy(visible_states)
                )

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                batch_size = samples.size(0)

                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size
                ).detach()
                batch_pl = self.pseudo_likelihood(samples).detach()

                mse += batch_mse
                pl += batch_pl

            mse /= len(batches)
            pl /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), pl=pl.item(), time=end - start)

        return mse, pl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.hidden_sampling(x)

        return x