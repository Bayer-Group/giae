"""adapted form https://github.com/e3nn/e3nn"""
import torch
from torch.utils.data import Dataset

from torch_geometric.utils import remove_self_loops
try:
    from torch_geometric.loader import DataLoader
except ModuleNotFoundError:
    from torch_geometric.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset

from pytorch_lightning import LightningDataModule
import math


def rand_matrix(*shape, requires_grad=False, dtype=None, device=None):
    r"""random rotation matrix
    Parameters
    ----------
    *shape : int
    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape}, 3, 3)`
    """
    R = angles_to_matrix(*rand_angles(*shape, dtype=dtype, device=device))
    return R.detach().requires_grad_(requires_grad)


def rand_angles(*shape, requires_grad=False, dtype=None, device=None):
    r"""random rotation angles
    Parameters
    ----------
    *shape : int
    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape})`
    beta : `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape})`
    gamma : `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape})`
    """
    alpha, gamma = 2 * math.pi * torch.rand(2, *shape, dtype=dtype, device=device)
    beta = torch.rand(shape, dtype=dtype, device=device).mul(2).sub(1).acos()
    alpha = alpha.detach().requires_grad_(requires_grad)
    beta = beta.detach().requires_grad_(requires_grad)
    gamma = gamma.detach().requires_grad_(requires_grad)
    return alpha, beta, gamma


def angles_to_matrix(alpha, beta, gamma):
    r"""conversion from angles to matrix
    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)


def matrix_x(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around X axis
    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([o, z, z], dim=-1),
        torch.stack([z, c, -s], dim=-1),
        torch.stack([z, s, c], dim=-1),
    ], dim=-2)


def matrix_y(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around Y axis
    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([c, z, s], dim=-1),
        torch.stack([z, o, z], dim=-1),
        torch.stack([-s, z, c], dim=-1),
    ], dim=-2)


def matrix_z(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around Z axis
    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([c, -s, z], dim=-1),
        torch.stack([s, c, z], dim=-1),
        torch.stack([z, z, o], dim=-1)
    ], dim=-2)


class DataModule(LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, num_eval_samples, train_samples=1000000):
        super().__init__()
        self.dataset = dataset
        self.train_samples = train_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_eval_samples = num_eval_samples

    def setup(self, stage=None):
        pass

    def train_dataloader(self, shuffle=False):
        dataset = self.dataset(num_elements=self.train_samples)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,   # True
            shuffle=shuffle
        )
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset(num_elements=self.num_eval_samples)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,   # True
        )
        return dataloader


class TetrisDataset(Dataset):
    def __init__(self, num_elements,
                 rotate: bool = True,
                 noise_level: float = 0.01,
                 translation_level: float = 5.0):
        self.num_elements = num_elements
        self.pos = [
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
        ]
        self.pos = torch.tensor(self.pos, dtype=torch.get_default_dtype())

        # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
        self.labels = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
            [0, 1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
            [0, 0, 1, 0, 0, 0, 0, 0],  # square
            [0, 0, 0, 1, 0, 0, 0, 0],  # line
            [0, 0, 0, 0, 1, 0, 0, 0],  # corner
            [0, 0, 0, 0, 0, 1, 0, 0],  # L
            [0, 0, 0, 0, 0, 0, 1, 0],  # T
            [0, 0, 0, 0, 0, 0, 0, 1],  # zigzag
        ], dtype=torch.get_default_dtype())

        self.rotate = rotate
        self.noise_level = noise_level
        self.translation_level = translation_level


    def __len__(self):
        return self.num_elements

    def __getitem__(self, item):
        i = torch.randint(0, len(self.pos), (1,))
        pos, label = self.pos[i], self.labels[i]
        if self.noise_level > 0.0:
            pos = pos + torch.randn_like(pos) * self.noise_level

        R = rand_matrix(len(pos))
        if self.rotate:
            pos = torch.einsum('zij,zaj->zai', R, pos)

        if self.translation_level > 0.0:
            transl = torch.randn(3, ) * 5.0
            # larger translation
            pos += transl

        return pos.squeeze(), label, R


def to_Data(datadict):
    return Data(**datadict)


class TetrisDatasetPyG(PyGDataset):
    def __init__(self,
                 num_elements: int,
                 rotate: bool = True,
                 noise_level: float = 0.01,
                 translation_level: float = 5.0,
                 transform=to_Data,
                 pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.num_elements = num_elements
        self.dataset = TetrisDataset(num_elements=num_elements,
                                     rotate=rotate,
                                     noise_level=noise_level,
                                     translation_level=translation_level)

    def len(self):
        return self.num_elements

    def get(self, idx):
        # hacky way, idx does not matter as we create random rotations on the fly anyways
        pos, label, rot = self.dataset[idx]
        # create fully-connected edges
        nelements = torch.tensor([pos.size(0)])

        # fully-connected graph
        edge_index = torch.cartesian_prod(torch.arange(nelements.item()), torch.arange(nelements.item())).T
        # remove self-loops
        edge_index, _ = remove_self_loops(edge_index)

        datadict = {"pos": pos.squeeze(), "label": label, "rot": rot,
                    "nelements": nelements, "edge_index": edge_index}
        # __getitem__ handled by parent-class.
        # https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/dataset.py#L184-L203
        return datadict


def _test():

    datamodule = DataModule(dataset=TetrisDatasetPyG,
                            batch_size=32,
                            num_workers=4,
                            num_eval_samples=1000)

    dataloader = datamodule.train_dataloader()
    data = next(iter(dataloader))
    print(data)
    print(data.pos.shape)
    print(data.batch)