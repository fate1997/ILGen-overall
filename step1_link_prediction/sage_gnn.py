import torch
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int
    ):
        super().__init__()
        self.conv1s = SAGEConv(in_channels, hidden_channels)
        self.conv1t = SAGEConv(in_channels, hidden_channels)
        self.conv2s = SAGEConv(hidden_channels, out_channels)
        self.conv2t = SAGEConv(hidden_channels, out_channels)

    def encode(self, data):
        x_s, x_t = data['cation'].x, data['anion'].x
        s2t = data['cation', 'anion'].edge_index
        t2s = data['anion', 'cation'].edge_index
        x_new_s = self.conv1s((x_t, x_s), t2s)
        x_new_t = self.conv1t((x_s, x_t), s2t)
        x_s = self.conv2s((x_new_t, x_new_s), t2s)
        x_t = self.conv2t((x_new_s, x_new_t), s2t)
        return x_s, x_t

    def decode(self, z_s, z_t, edge_label_index):
        return (z_s[edge_label_index[0]] * z_t[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z_s, z_t):
        prob_adj = z_s @ z_t.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()