import torch


def normalize_vector(v, eps=10e-9):
    v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    return v


def orthogonal_projection(v1, v2):
    #  project v2 onto v1
    dot_prod = torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).squeeze()
    v2 = v2 - dot_prod.unsqueeze(-1) * v1
    return v2


def get_rotation_matrix_from_two_vector(v1, v2):
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)
    v2 = orthogonal_projection(v1, v2)
    v2 = normalize_vector(v2)
    v3 = torch.cross(v1, v2, dim=-1)
    rot = torch.stack((v1, v2, v3), dim=-1)
    return rot


