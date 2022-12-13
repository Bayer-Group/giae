"""adapted from https://github.com/QUVA-Lab/e2cnn"""
import torch
import torch.nn.functional as F
from torch.nn import Module
from e2cnn import gspaces
from e2cnn import nn


class Encoder(Module):
    def __init__(self, out_dim, hidden_dim=32):
        super().__init__()
        self.out_dim=out_dim
        self.r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=8)
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type

        # convolution 1
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block1 = nn.SequentialModule(
            #nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 2
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 4
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 6
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        # convolution 7 --> out
        # the old output type is the input type to the next layer
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, out_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, 1 * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field

        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        #x = torch.nn.functional.pad(x, (0, 1, 0, 1), value=0).unsqueeze(1)
        x = nn.GeometricTensor(x, self.input_type)

        x = self.block1(x)
        x = self.block2(x)
        #x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        #x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        #x = self.pool3(x)
        x = self.block7(x)

        #x = x.tensor.squeeze(-1).squeeze(-1)
        x = x.tensor.mean(dim=(2, 3))

        x_0, x_1 = x[:, :self.out_dim], x[:, self.out_dim:]

        return x_0, x_1


"""class Decoder(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=8)

        # the input image is a scalar field, corresponding to the trivial representation
        in_scalar_fields = nn.FieldType(self.r2_act, input_size * [self.r2_act.trivial_repr])
        in_type = in_scalar_fields
        self.input_type = in_type

        # convolution 1
        out_type = nn.FieldType(self.r2_act, hidden_size * [self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        self.upsampling1 = nn.R2Upsampling(out_type, 4)

        # convolution 2
        in_type = out_type
        out_type = nn.FieldType(self.r2_act, hidden_size * [self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        self.upsampling2 = nn.R2Upsampling(out_type, 2)

        # convolution 3
        in_type = out_type
        out_type = nn.FieldType(self.r2_act, hidden_size * [self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        self.upsampling3 = nn.R2Upsampling(out_type, 2)

        # convolution 4
        in_type = out_type
        out_type = nn.FieldType(self.r2_act, hidden_size * [self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        self.upsampling4 = nn.R2Upsampling(out_type, 2)

        # convolution 5
        in_type = out_type
        out_type = nn.FieldType(self.r2_act, hidden_size * [self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # convolution 6
        in_type = out_type
        out_type = nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1).unsqueeze(-1)  # [bz, emb_dim, 1, 1]
        pos_emb = torch.Tensor([[1, 2], [4, 3]]).type_as(x).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1, -1)
        x = x + pos_emb
        x = nn.GeometricTensor(x, self.input_type)

        x = self.block1(x)
        x = self.upsampling1(x)
        x = self.block2(x)
        x = self.upsampling2(x)
        x = self.block3(x)
        x = self.upsampling3(x)
        x = self.block4(x)
        x = self.upsampling4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.tensor[:, :, 2:30, 2:30]
        x = torch.sigmoid(x)
        return x"""

class Decoder(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # convolution 1
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_size, hidden_size, kernel_size=1, padding=0,),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 2
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 3
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 4
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 5
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 6
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, 1, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1).unsqueeze(-1)  # [bz, emb_dim, 1, 1]
        x = x.expand(-1, -1, 2, 2)
        #pos_emb = torch.Tensor([[1, 2], [4, 3]]).type_as(x).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1, -1)
        #x = x + pos_emb

        x = self.block1(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block2(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block3(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block4(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block5(x)
        x = self.block6(x)
        x = x[:, :, 2:30, 2:30]
        x = torch.sigmoid(x)
        return x


class Model(Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder(out_dim=hparams["emb_dim"])
        self.decoder = Decoder(input_size=hparams["emb_dim"], hidden_size=hparams["hidden_dim"])

    def forward(self, x, do_rot=True):
        emb, v = self.encoder(x)
        rot = get_rotation_matrix(v)
        #print(v, rot)
        y = self.decoder(emb)
        if do_rot:
            y = rot_img(y, rot)
        return y, rot, emb



def get_rotation_matrix(v, eps=10e-5):
    v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    rot = torch.stack((
        torch.stack((v[:, 0], v[:, 1]), dim=-1),
        torch.stack((-v[:, 1], v[:, 0]), dim=-1),
        torch.zeros(v.size(0), 2).type_as(v)
    ), dim=-1)
    return rot

def rot_img(x, rot):
    grid = F.affine_grid(rot, x.size(), align_corners=False).type_as(x)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


def get_non_linearity(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    relu = nn.ReLU(scalar_fields)
    norm_relu = nn.NormNonLinearity(vector_fields)
    nonlinearity = nn.MultipleModule(
        out_type,
        ['relu'] * len(scalar_fields) + ['norm'] * len(vector_fields),
        [(relu, 'relu'), (norm_relu, 'norm')]
    )
    return nonlinearity

def get_batch_norm(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    batch_norm = nn.InnerBatchNorm(scalar_fields)
    norm_batch_norm = nn.NormBatchNorm(vector_fields)
    batch_norm = nn.MultipleModule(
        out_type,
        ['bn'] * len(scalar_fields) + ['nbn'] * len(vector_fields),
        [(batch_norm, 'bn'), (norm_batch_norm, 'nbn')]
    )
    return batch_norm



class ClassicalEncoder(Module):
    def __init__(self, out_dim, hidden_dim=32):
        super().__init__()

        # convolution 1

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, hidden_dim, kernel_size=7, padding=1, bias=False),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU()
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU()
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU()
        )
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU()
        )
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU()
        )
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, bias=False),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU()
        )
        self.block7 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, out_dim, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU()
        )

        self.pool1 = torch.nn.MaxPool2d(kernel_size=5, stride=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=5, stride=2)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        #x = torch.nn.functional.pad(x, (0, 1, 0, 1), value=0).unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = x.mean(dim=(2, 3))

        return x, None


class ClassicalModel(Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = ClassicalEncoder(out_dim=hparams["emb_dim"], hidden_dim=hparams["hidden_dim"])
        self.decoder = Decoder(input_size=hparams["emb_dim"], hidden_size=hparams["hidden_dim"])

    def forward(self, x, do_rot=True):
        emb, _ = self.encoder(x)
        y = self.decoder(emb)
        return y, None, emb