"""

Experimental Implementation of a simple Equivariant GNN


"""

from typing import Union, Tuple, Optional
import numpy as np
from torch.nn import Module
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, Batch
from torch_geometric.typing import OptTensor
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.inits import reset


from torch_scatter import scatter

from giae.se3.utils import get_rotation_matrix_from_two_vector


def norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


class GatedEquivLayer(nn.Module):
    """
    Implements a Gated Equivariant Layer.
    Assumes the vector has an input dimensionality
    """

    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        h_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
    ):
        super(GatedEquivLayer, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.h_dim = h_dim or max(self.vi, self.vo)
        self.norm_eps = norm_eps

        if self.vi is not None:
            self.Wv0 = nn.Linear(self.vi, self.h_dim, bias=False)
            sin = self.h_dim + self.si
        else:
            self.register_buffer("Wv0", None)
            sin = self.si

        if self.vo is not None:
            self.Wv1 = nn.Linear(self.vi, self.vo, bias=False)
            vo = self.vo
        else:
            self.register_buffer("Wv1", None)
            vo = 0

        self.Ws0 = nn.Linear(sin, self.h_dim, bias=True)
        self.Ws1 = nn.Linear(self.h_dim, vo + self.so, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.Ws0.reset_parameters()
        self.Ws1.reset_parameters()
        if self.Wv0 is not None:
            self.Wv0.reset_parameters()
        if self.Wv1 is not None:
            self.Wv1.reset_parameters()

    def forward(self, x: Tuple[Tensor, OptTensor]) -> Tuple[Tensor, OptTensor]:
        """
        Computes a forward pass on a scalar-vector object.
        :param x: Tuple with scalar (s) and vector (v) part.
                  Can also handle the case that vector part is non-existent. In this case, this layer is a standard
                  1-layer MLP.
        :return: Tuple of transformed scalar-vector object.
        """

        s, v = x

        # s has shape [..., self.si]
        # if is not None has shape [..., self.vi, 3]

        # process the vector part if existent
        if v is not None:
            assert v.size(-1) == 3
            # fuse information from vector part to scalar part, i.e., using norm on vector part to obtain a
            # rotation-invariant representation from the vector part
            vnorm = self.Wv0(v.transpose(-2, -1)).transpose(-2, -1)
            vnorm = norm_no_nan(vnorm, eps=self.norm_eps, sqrt=True)
            s = torch.cat([s, vnorm], dim=-1)

        # transform the scalar part using a 1-layer MLP
        s = self.Ws1(F.silu(self.Ws0(s)))

        if v is not None:
            if self.vo is not None:
                sgate, s = s.split([self.vo, self.so], dim=-1)
                # fuse information from the scalar part to the vector part via gating
                v = sgate.unsqueeze(1) * self.Wv1(v.transpose(-2, -1))
                v = v.transpose(-2, -1)
            else:
                v = None

        return s, v


class LayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims, elementwise_affine: bool = True):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s, elementwise_affine=elementwise_affine)
        self.reset_parameters()

    def reset_parameters(self):
        self.scalar_norm.reset_parameters()

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


def get_node_mask_batch(batch: Tensor, batch_num_nodes: Tensor, max_num_nodes: int,
                        batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    cum_nodes = torch.cat([batch.new_zeros(1), batch_num_nodes.cumsum(dim=0)])
    idx = torch.arange(batch.size(0), dtype=torch.long, device=batch.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=batch.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return mask


def sparse_sort(src: Tensor, index: Tensor, dim: int = 0,
                descending: bool = False, eps: float = 1e-12,
                dtype: torch.dtype = torch.float32):

    f_src = src.to(dtype)
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min)/(f_max - f_min + eps) + index.to(dtype)*(-1)**int(descending)
    perm = norm.argsort(dim=dim, descending=descending)

    return src[perm], perm


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self.get_sinusoid_encoding_table_torch(n_position, d_hid))

    @classmethod
    def get_sinusoid_encoding_table_torch(self, n_position, d_hid):
        i_hid = torch.arange(d_hid)
        i_pos = torch.arange(n_position)
        power_frequency = 2 * torch.div(i_hid, 2, rounding_mode='trunc') / d_hid
        frequency = torch.pow(10000, power_frequency)
        sinusoid_table = i_pos.unsqueeze(-1) / frequency.unsqueeze(0)

        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table.unsqueeze(0)

    @classmethod
    def get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, batch_size, num_nodes):
        x = self.pos_table[:, :num_nodes].clone().detach()
        x = x.expand(batch_size, -1, -1)
        return x


class Permuter(torch.nn.Module):
    """adapted from https://github.com/jrwnter/pigvae"""
    def __init__(self, input_dim):
        super().__init__()
        self.scoring_fc = Linear(input_dim, 1)

    def score(self, x):
        scores = self.scoring_fc(x)
        return scores

    def soft_sort(self, scores: Tensor, batch: Tensor, tau: float = 1.0, hard: bool = False):
        scores = scores.squeeze()
        scores_sorted, _ = sparse_sort(scores, index=batch, dim=0, descending=False)
        scores_dense, mask = to_dense_batch(scores, batch)
        scores_sorted_dense, mask = to_dense_batch(scores_sorted, batch)

        mask_adj = mask.unsqueeze(dim=1) * mask.unsqueeze(dim=-1)

        pairwise_diff = (scores_dense.unsqueeze(-1) - scores_sorted_dense.unsqueeze(dim=1)).abs().neg() / tau
        pairwise_diff_masked = mask_adj * pairwise_diff

        pairwise_diff_masked[~mask_adj] = -9999.
        perm = pairwise_diff_masked.softmax(dim=-1)
        perm = mask_adj * perm

        if hard:
            perm_ = torch.zeros_like(perm, device=perm.device)
            perm_.scatter_(-1, perm.topk(1, -1)[1], value=1)
            perm = (perm_ - perm).detach() + perm
            perm = mask_adj * perm

        return perm

    def mask_perm(self, perm, mask):
        batch_size, num_nodes = mask.size(0), mask.size(1)
        eye = torch.eye(num_nodes, num_nodes).unsqueeze(0).expand(batch_size, -1, -1).type_as(perm)
        mask = mask.unsqueeze(-1).expand(-1, -1, num_nodes)
        perm = torch.where(mask, perm, eye)
        return perm

    def forward(self, node_features: Tensor, batch: Tensor,  hard=False, tau=1.0):
        # add noise to break symmetry
        node_features = node_features + torch.randn_like(node_features) * 0.05
        scores = self.score(node_features)
        perm = self.soft_sort(scores=scores, batch=batch, hard=hard, tau=tau)
        perm = perm.transpose(2, 1)
        return perm

    @staticmethod
    def permute_node_features(node_features, perm):
        node_features = torch.matmul(perm, node_features)
        return node_features

    @staticmethod
    def permute_edge_features(edge_features, perm):
        edge_features = torch.matmul(perm.unsqueeze(1), edge_features)
        edge_features = torch.matmul(perm.unsqueeze(1), edge_features.permute(0, 2, 1, 3))
        edge_features = edge_features.permute(0, 2, 1, 3)
        return edge_features

    @staticmethod
    def permute_graph(graph, perm):
        graph.node_features = Permuter.permute_node_features(graph.node_features, perm)
        graph.edge_features = Permuter.permute_edge_features(graph.edge_features, perm)
        return graph



class RadialBasis(nn.Module):
    def __init__(self, num_radial: int):
        """
        Instantiates a Radial-Basis-Function that operates on euclidean distances
        :param num_radial:
        """
        super(RadialBasis, self).__init__()
        self.num_radial = num_radial

    def forward(self, dist: Tensor) -> Tensor:
        raise NotImplementedError()


class GaussianRBF(RadialBasis):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 10.0,
        num_radial: int = 20,
    ):
        """
        See https://en.wikipedia.org/wiki/Radial_basis_function_kernel
        :param start: where the rbf reference mean starts
        :param stop:where the rbf reference mean stops
        :param num_radial: the number of RBF functions to use to encode the scalar distance in terms of RBFs
        """
        super(GaussianRBF, self).__init__(num_radial=num_radial)
        assert start >= 0.0, "start must be positive valued"
        assert start < stop
        self.stop = stop
        self.num_radial = num_radial
        # get `num_gaussians` evenly spaced discretized means
        mu = torch.linspace(start, stop, num_radial)
        # as all gaussians are evenly spaced, the width is also equal among all consecutive pairs.
        self.gamma = 0.5 / (mu[1] - mu[0]).item() ** 2
        self.register_buffer("mu", mu)

    def forward(self, dist: Tensor) -> Tensor:
        """
        Encode distances `dist` in terms of Gaussian radial basis functions
        :param dist: [..., 1] tensor where the last axis refers to the distance
        :return: radial basis encoding of distance according to
        rbf = exp(-gamma * (d - mu)**2) where mu and gamma are fixed and d is the 1d-distance
        """
        dist = dist.view(-1, 1) - self.mu.view(1, -1)
        out = torch.exp(-1.0 * self.gamma * torch.pow(dist, 2))
        return out


class EquivConv(MessagePassing):
    def __init__(self, in_features: Tuple[int, int], out_features: Tuple[int, int], aggr: str = "add"):
        super(EquivConv, self).__init__(aggr=aggr, node_dim=0)

        self.si, self.vi = in_features
        self.so, self.vo = out_features
        self.distance_expansion = GaussianRBF(start=0.0, stop=4.0, num_radial=8)
        self.message_kernel = nn.Linear(2*self.si + 8,
                                        2*self.vi + self.si)

        self.Ws = nn.Linear(self.si, self.si, bias=True)
        self.Wv = nn.Linear(self.vi, self.vi, bias=False)
        self.Wu = GatedEquivLayer(in_dims=(self.si, self.vi),
                                  out_dims=(self.so, self.vo)
                                  )


    def forward(self,
                x: Tuple[Tensor, Tensor, None],
                edge_attr: [Tensor, Tensor],
                edge_index: Tensor) -> Tuple[Tensor, Tensor, None]:

        s, v, _ = x

        us = self.Ws(s)
        uv = self.Wv(v.transpose(-2, -1)).transpose(-2, -1)
        ms, mv = self.propagate(xs=s, xv=uv, s=us, edge_index=edge_index, edge_attr=edge_attr)

        ms = ms + s
        mv = mv + v

        us, uv = self.Wu(x=(ms, mv))

        us = us + ms
        uv = uv + mv

        return us, uv, None

    def aggregate(
            self, inputs: Tuple[Tensor, Tensor], index: Tensor
    ) -> Tuple[Tensor, Tensor, OptTensor]:
        msg_s, msg_v = inputs
        msg_s = scatter(msg_s, index, dim=0, reduce=self.aggr)
        msg_v = scatter(msg_v, index, dim=0, reduce=self.aggr)
        return msg_s, msg_v


    def message(self,
                xs_i: Tensor,
                xs_j: Tensor,
                s_j: Tensor,
                xv_i: Tensor,
                xv_j: Tensor,
                edge_attr: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:

        d, r = edge_attr
        msg_in = torch.cat([xs_i, xs_j, self.distance_expansion(d)], dim=-1)
        splitter = [self.si, self.vi, self.vi]
        mij = self.message_kernel(msg_in)

        a_ij, b_ij, c_ij = mij.split(splitter, dim=-1)

        s_j = a_ij * s_j

        v_msg = torch.cross(xv_i, xv_j, dim=-1)
        v0_j = b_ij.unsqueeze(dim=-1) * v_msg
        v1_j = c_ij.unsqueeze(dim=-1) * r.unsqueeze(dim=1)
        v_j = v0_j + v1_j

        return s_j, v_j


class EquivConvWithTransl(MessagePassing):
    def __init__(self, in_features: Tuple[int, int], out_features: Tuple[int, int], aggr: str = "add"):
        super(EquivConvWithTransl, self).__init__(aggr=aggr, node_dim=0)

        self.si, self.vi = in_features
        self.so, self.vo = out_features
        self.distance_expansion = GaussianRBF(start=0.0, stop=4.0, num_radial=8)
        self.message_kernel = nn.Linear(2*self.si + 8,
                                        2*self.vi + self.si + 1)

        self.Ws = nn.Linear(self.si, self.si, bias=True)
        self.Wv = nn.Linear(self.vi, self.vi, bias=False)
        # self.Wv = nn.Identity()

        self.Wu = GatedEquivLayer(in_dims=(self.si, self.vi),
                                  out_dims=(self.so, self.vo)
                                  )


    def forward(self,
                x: Tuple[Tensor, Tensor, Tensor],
                edge_attr: [Tensor, Tensor],
                edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        s, v, pos = x

        us = self.Ws(s)
        uv = self.Wv(v.transpose(-2, -1)).transpose(-2, -1)
        ms, mv, mp = self.propagate(xs=s, xv=uv, s=us, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

        ms = ms + s
        mv = mv + v
        pos = mp + pos
        us, uv = self.Wu(x=(ms, mv))

        us = us + ms
        uv = uv + mv

        return us, uv, pos

    def aggregate(
            self, inputs: Tuple[Tensor, Tensor, Tensor], index: Tensor
    ) -> Tuple[Tensor, Tensor]:
        msg_s, msg_v, msg_pos = inputs
        msg_s = scatter(msg_s, index, dim=0, reduce=self.aggr)
        msg_v = scatter(msg_v, index, dim=0, reduce=self.aggr)
        msg_pos = scatter(msg_pos, index, dim=0, reduce=self.aggr)
        return msg_s, msg_v, msg_pos


    def message(self,
                xs_i: Tensor,
                xs_j: Tensor,
                s_j: Tensor,
                xv_i: Tensor,
                xv_j: Tensor,
                pos_i: Tensor,
                pos_j: Tensor,
                edge_attr: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:

        d, r = edge_attr
        msg_in = torch.cat([xs_i, xs_j, self.distance_expansion(d)], dim=-1)
        splitter = [self.si, self.vi, self.vi, 1]
        mij = self.message_kernel(msg_in)

        a_ij, b_ij, c_ij, d_ij = mij.split(splitter, dim=-1)

        s_j = a_ij * s_j

        v_msg = torch.cross(xv_i, xv_j, dim=-1)
        v0_j = b_ij.unsqueeze(dim=-1) * v_msg
        v1_j = c_ij.unsqueeze(dim=-1) * r.unsqueeze(dim=1)
        v_j = v0_j + v1_j

        msg_pos_j = d_ij * (pos_i - pos_j)

        return s_j, v_j, msg_pos_j



class ShapeGNN(nn.Module):
    # thinks about a increasing/decreasing dimensionality as in a encoder-decoder framework.
    def __init__(self,
                 dims: Tuple[int, int],
                 depth: int,
                 has_transl: bool = False,
                 aggr: str = "add",
                 use_layer_norm: bool = True,
                 affine: bool = True):
        super(ShapeGNN, self).__init__()
        self.dims = dims
        self.depth = depth
        self.use_layer_norm = use_layer_norm
        self.affine = affine
        self.has_transl = has_transl
        self.norms = nn.ModuleList()
        self.convs = nn.ModuleList()

        if has_transl:
            module = EquivConvWithTransl
        else:
            module = EquivConv

        for i in range(depth):
            if use_layer_norm:
                self.norms.append(LayerNorm(dims=dims, elementwise_affine=affine))
            self.convs.append(
                module(
                    in_features=dims,
                    out_features=dims,
                    aggr=aggr,
                )
            )

        self.apply(fn=reset)

    def forward(
            self,
            x: Tuple[Tensor, Tensor, OptTensor],
            edge_index: Tensor,
            edge_attr: Tuple[OptTensor, OptTensor],
    ) -> Tuple[Tensor, Tensor, OptTensor]:

        s, v, pos = x
        for i, conv in enumerate(self.convs):
            if self.use_layer_norm:
                s, v = self.norms[i](x=(s, v))
            s, v, pos = conv(x=(s, v, pos), edge_index=edge_index, edge_attr=edge_attr)
        return s, v, pos


class Encoder(Module):
    def __init__(self, hidden_dim, emb_dim, num_layers,
                 layer_norm: bool = True,
                 num_nearest=16, aggr: str = "add"):
        super().__init__()
        self.num_nearest = num_nearest
        self.hidden_dim = hidden_dim
        self.aggr = aggr

        self.equivariant_gnn = ShapeGNN(dims=(hidden_dim, hidden_dim),
                                        depth=num_layers,
                                        aggr=aggr,
                                        use_layer_norm=layer_norm,
                                        affine=True,
                                        has_transl=True,
                                        )

        self.lin = GatedEquivLayer(in_dims=(hidden_dim, hidden_dim),
                                   out_dims=(2*emb_dim, 2)
                                   )

        self.ohe_lin = Linear(4, hidden_dim)


    def forward(self, pos: Tensor,
                batch: Tensor,
                batch_num_nodes: Tensor,
                edge_index: OptTensor = None,
                use_fc: bool = True):

        # s_in_shape = (pos.size(0), self.hidden_dim)
        v_in_shape = [pos.size(0)] + [self.hidden_dim, 3]
        # s = torch.ones(s_in_shape).type_as(pos) * 0.1
        s = torch.diag(torch.ones((4,), device=pos.device))
        s = s.repeat(len(batch_num_nodes), 1)
        s = self.ohe_lin(s)
        v = torch.zeros(v_in_shape, device=pos.device)

        if use_fc:
            if edge_index is None:
                # assume fully connected graph without self-loops
                edge_index = [torch.ones(n, n) for n in batch_num_nodes.tolist()]
                # below is erroneous if batch consists of samples with different number of nodes...
                # edge_index = torch.ones(size=(batch_size, 4, 4),
                #                         device=s.device, dtype=torch.long)
                edge_index = torch.block_diag(*edge_index).fill_diagonal_(0.0).to(s.device)
                edge_index = edge_index.nonzero().t()
        else:
            edge_index = knn_graph(x=pos, k=self.num_nearest, batch=batch)


        row, col = edge_index
        r = pos[row] - pos[col]
        d = torch.pow(r, 2).sum(-1)
        d = d.clamp(min=1e-4)

        # scalar and vector node embeddings
        sout, vout, transl_out = self.equivariant_gnn(x=(s, v, pos), edge_index=edge_index, edge_attr=(d, r))

        # rotation invariant scalar shape embedding ; rotation equivariant vector shape embedding
        sout, vout = self.lin((sout, vout))
        shape_embed, point_embed = sout.chunk(2, dim=-1)
        shape_embed = scatter(shape_embed, index=batch, dim=0, reduce="mean")
        vout = scatter(vout, index=batch, dim=0, reduce="mean")
        transl_out = scatter(transl_out, index=batch, dim=0, reduce="mean")
        rot = get_rotation_matrix_from_two_vector(vout[:, 0], vout[:, 1])

        return shape_embed, point_embed, rot, transl_out, vout


class Decoder(Module):
    def __init__(self, hidden_dim, emb_dim, num_layers,
                 layer_norm: bool = True,
                 n_position: int = 200, num_nearest=16, aggr: str = "add"):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_nearest = num_nearest

        self.start_pos = nn.Parameter(torch.randn(4, 3))
        self.pos_encoder = PositionalEncoding(d_hid=emb_dim, n_position=n_position)
        self.lin = Linear(emb_dim, hidden_dim)

        self.equivariant_gnn = ShapeGNN(dims=(hidden_dim, hidden_dim),
                                        depth=num_layers,
                                        aggr=aggr,
                                        use_layer_norm=layer_norm,
                                        affine=True, has_transl=False
                                        )

    def forward(self, s: Tensor, rot: Tensor, transl: Tensor, perm: Tensor,
                batch: Tensor, batch_num_nodes: Tensor,
                node_mask: Tensor, use_fc: bool = True):
        batch_size = s.size(0)
        max_num_nodes = int(torch.max(batch_num_nodes))
        gstart = self.pos_encoder(batch_size=batch_size, num_nodes=max_num_nodes)
        gstart = gstart[node_mask].detach()
        s = torch.repeat_interleave(s, batch_num_nodes, dim=0)
        s = gstart + s
        s = self.lin(s)

        s, mask = to_dense_batch(s, batch=batch)
        s = torch.matmul(perm, s)
        s = s.view(-1, self.hidden_dim)

        pos_start = self.start_pos.unsqueeze(0)
        if rot is not None:
            pos_start = torch.einsum('zij,zaj->zai', rot, pos_start)
        else:
            pos_start = pos_start.expand(batch_size, -1, -1)
        pos = pos_start.reshape(-1, 3)

        if use_fc:
            # assume fully connected graph without self-loops
            edge_index = [torch.ones(n, n) for n in batch_num_nodes.tolist()]
            # below is erroneous if batch consists of samples with different number of nodes...
            # edge_index = torch.ones(size=(batch_size, 4, 4),
            #                         device=s.device, dtype=torch.long)
            edge_index = torch.block_diag(*edge_index).fill_diagonal_(0.0).to(s.device)
            edge_index = edge_index.nonzero().t()
        else:
            edge_index = knn_graph(x=pos, k=self.num_nearest, batch=batch)

        row, col = edge_index

        r = pos[row] - pos[col]
        d = torch.pow(r, 2).sum(-1)
        # enforce minimal distance
        d = d.clamp(min=1e-4, max=10.)

        v = torch.zeros(size=(s.size(0), s.size(1), 3), device=s.device)

        sout, vout, _ = self.equivariant_gnn(x=(s, v, None), edge_index=edge_index, edge_attr=(d, r))
        vout = vout.mean(dim=1)
        vout += pos
        transl = torch.repeat_interleave(transl, batch_num_nodes, dim=0)
        vout += transl

        return vout


class Model(Module):
    def __init__(self,
                 hidden_dim,
                 emb_dim,
                 num_layers,
                 layer_norm=True,
                 encoder_nearest=16,
                 decoder_nearest=16,
                 num_points=200,
                 encoder_aggr="mean",
                 decoder_aggr="mean"):
        super().__init__()
        self.encoder = Encoder(
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_nearest=encoder_nearest,
            aggr=encoder_aggr,
            layer_norm=layer_norm,
        )

        self.permuter = Permuter(emb_dim)

        self.decoder = Decoder(
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_nearest=decoder_nearest,
            aggr=decoder_aggr,
            n_position=num_points,  # for positional encoding
            layer_norm=layer_norm,
        )

    def forward(self, data: Union[Data, Batch], tau: float = 1.0, hard: bool = False, do_rot: bool = True, **kwargs):
        pos, batch, edge_index = data.pos, data.batch, data.edge_index
        batch_size = int(batch.max()) + 1
        batch_num_nodes = torch.bincount(batch)
        max_nodes = int(batch_num_nodes.max())
        shape_emb, point_emb, rot, transl, vout = self.encoder(pos=data.pos,
                                                               batch=batch,
                                                               batch_num_nodes=batch_num_nodes,
                                                               edge_index=edge_index,
                                                               use_fc=False)
        perm = self.permuter(point_emb, batch=batch, tau=tau, hard=hard)
        node_mask = get_node_mask_batch(batch=batch,
                                        batch_num_nodes=batch_num_nodes,
                                        batch_size=batch_size, max_num_nodes=max_nodes)

        pos_out = self.decoder(s=shape_emb, perm=perm, rot=rot if do_rot else None, transl=transl,
                               batch=batch, batch_num_nodes=batch_num_nodes,
                               node_mask=node_mask, use_fc=True)

        return pos_out, perm, vout, rot


if __name__ == '__main__':
    from givae.se3.data import TetrisDatasetPyG, DataModule
    datamodule = DataModule(dataset=TetrisDatasetPyG,
                            batch_size=32,
                            num_workers=0,
                            num_eval_samples=1000)

    datamodule.setup()
    loader = datamodule.train_dataloader()
    data = next(iter(loader))
    data = data.to("cuda:0")
    model = Model(hidden_dim=32, emb_dim=2, num_layers=5).to("cuda:0")

    pos_out, perm, vout = model(data, hard=False)
    loss = torch.pow(data.pos - pos_out, 2)
    loss = scatter(loss, data.batch, dim=0, reduce="add")
    loss = loss.mean()
    print(loss)
    loss.backward()


    # check rotation equivariance
    from scipy.spatial.transform import Rotation
    R = torch.from_numpy(Rotation.random().as_matrix()).float().to("cuda:0")
    print("checking invariance and equivariance")
    shape_emb, point_emb, rot, transl, vout = model.encoder(pos=data.pos,
                                                            batch=data.batch,
                                                            batch_num_nodes=torch.bincount(data.batch),
                                                            edge_index=data.edge_index,
                                                            use_fc=False)

    pos_R = data.pos @ R.T

    shape_emb_R, point_emb_R, rot_R, transl_R, vout_R = model.encoder(pos=pos_R,
                                                                      batch=data.batch,
                                                                      batch_num_nodes=torch.bincount(data.batch),
                                                                      edge_index=data.edge_index,
                                                                      use_fc=False)

    # Equivariance wrt to Rotations
    # shape embedding shoould invariant wrt. rotation
    assert torch.allclose(shape_emb, shape_emb_R, atol=1e-5)
    # equivariance for translation vector wrt. to rotation
    assert torch.allclose(transl @ R.T, transl_R, atol=1e-5)
    # equivariance for rotation vectors wrt. to rotation
    assert torch.allclose(vout @ R.T, vout_R, atol=1e-5)
    d = R @ rot - rot_R
    assert d.norm().item() < 1e-3

    # Equivariance wrt to Translations
    shape_emb_transl, point_emb_transl, rot_transl, transl2, vout_transl = model.encoder(pos=data.pos + torch.ones_like(data.pos),
                                                                                         batch=data.batch,
                                                                                         batch_num_nodes=torch.bincount(data.batch),
                                                                                         edge_index=data.edge_index,
                                                                                         use_fc=False)

    assert torch.allclose(shape_emb, shape_emb_transl, atol=1e-5)
    # equivariance for translation vector wrt. to translation
    assert torch.allclose(transl + torch.ones_like(transl), transl2, atol=1e-5)
    # invariance for rotation vectors wrt. to translation
    assert torch.allclose(vout, vout_transl, atol=1e-5)

