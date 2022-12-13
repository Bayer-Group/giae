import torch
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d


class Model(Module):
    def __init__(self, emb_dim, hidden_dim, num_digits, num_classes):
        super().__init__()
        self.encoder = Encoder(
            emb_dim=emb_dim,
            input_dim=num_classes,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        self.decoder = Decoder(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_digits=num_digits,
            num_classes=num_classes
        )
        self.permuter = Permuter(
            input_dim=hidden_dim
        )

    def forward(self, x, hard):
        set_emb, element_emb = self.encoder(x, aggr="sum")
        perm = self.permuter(element_emb, hard=hard)
        y = self.decoder(set_emb)
        y = torch.matmul(perm, y)
        return y, perm, set_emb




class Encoder(Module):
    def __init__(self, emb_dim, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.element_encoder = Sequential(
            Linear(num_classes, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.set_encoder = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, emb_dim),
        )

    def forward(self, x, aggr="sum"):
        # x: [bz, N, D]
        batch_size, num_elements = x.size(0), x.size(1)

        x = x.view(batch_size * num_elements, -1)
        x = self.element_encoder(x)
        x = x.view(batch_size, num_elements, -1)

        if aggr == "mean":
            x_aggr = x.mean(dim=1)
        elif aggr == "sum":
            x_aggr = x.sum(dim=1)
        else:
            raise NotImplementedError
        emb = self.set_encoder(x_aggr)

        return emb, x


class Decoder(Module):
    def __init__(self, emb_dim, hidden_dim, num_digits, num_classes):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_digits = num_digits
        self.num_classes = num_classes

        self.element_decoder = Sequential(
            Linear(self.emb_dim + num_digits, self.hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, num_classes)
        )

    def forward(self, emb):
        # emb: [bz, D]
        batch_size = emb.size(0)
        x = emb.unsqueeze(1).expand(-1, self.num_digits, -1)
        pos_emb = torch.diag(torch.ones(self.num_digits)).unsqueeze(0).expand(emb.size(0), -1, -1).type_as(x)
        x = torch.cat((x, pos_emb), dim=-1)
        x = x.view(batch_size * self.num_digits, self.emb_dim + self.num_digits)
        x = self.element_decoder(x)
        x = x.view(batch_size, self.num_digits, self.num_classes)

        return x


class Decoder2(Module):
    def __init__(self, emb_dim, hidden_dim, num_digits, num_classes):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_digits = num_digits
        self.num_classes = num_classes

        self.element_decoder = Sequential(
            Linear(self.emb_dim, self.hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, num_digits*num_classes)
        )

    def forward(self, emb):
        # emb: [bz, D]
        x = self.element_decoder(emb)
        x = x.view(emb.size(0), self.num_digits, self.num_classes)

        return x


class Permuter(Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scoring_fc = Linear(input_dim, 1)

    def score(self, x, mask=None):
        scores = self.scoring_fc(x)
        #fill_value = scores.min().item() - 1
        #scores = scores.masked_fill(mask.unsqueeze(-1) == 0, fill_value)
        return scores

    def soft_sort(self, scores, hard, tau):
        scores_sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - scores_sorted).abs().neg() / tau
        perm = pairwise_diff.softmax(-1)
        if hard:
            perm_ = torch.zeros_like(perm, device=perm.device)
            perm_.scatter_(-1, perm.topk(1, -1)[1], value=1)
            perm = (perm_ - perm).detach() + perm
        return perm

    def mask_perm(self, perm, mask):
        batch_size, num_nodes = mask.size(0), mask.size(1)
        eye = torch.eye(num_nodes, num_nodes).unsqueeze(0).expand(batch_size, -1, -1).type_as(perm)
        mask = mask.unsqueeze(-1).expand(-1, -1, num_nodes)
        perm = torch.where(mask, perm, eye)
        return perm

    def forward(self, node_features, mask=None, hard=False, tau=1.0):
        # add noise to break symmetry
        node_features = node_features + torch.randn_like(node_features) * 0.05
        scores = self.score(node_features, mask)
        perm = self.soft_sort(scores, hard, tau)
        perm = perm.transpose(2, 1)
        #perm = self.mask_perm(perm, mask)
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


class ClassicalModel(Module):
    def __init__(self, emb_dim, hidden_dim, num_digits, num_classes):
        super().__init__()
        self.num_digits = num_digits
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.element_encoder = torch.nn.Sequential(
            Linear(num_classes, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.set_encoder = torch.nn.Sequential(
            Linear(hidden_dim*self.num_digits, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, emb_dim),
        )
        self.decoder = torch.nn.Sequential(
            Linear(emb_dim, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(self.hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_digits*num_classes),
        )

    def forward(self, x, hard):
        batch_size = x.size(0)
        x = x.view(batch_size * self.num_digits, self.num_classes)
        x = self.element_encoder(x)
        x = x.view(batch_size, self.num_digits * self.hidden_dim)
        emb = self.set_encoder(x)
        y = self.decoder(emb)
        y = y.view(batch_size, self.num_digits, self.num_classes)
        perm = torch.diag(torch.ones(self.num_digits)).unsqueeze(0).expand(batch_size, -1, -1).type_as(x)
        return y, perm, emb
