import torch, torch.nn as nn


class DConv(nn.Module):
    """Diffusion graph convolution: mixes each sensor with its upstream and downstream neighbours (K hops)."""
    def __init__(self, cin, cout, supports, K=2):
        super().__init__()
        self.register_buffer('S', supports)                    # [2, N, N] forward / backward transitions
        self.K = K
        self.lin = nn.Linear(cin * (1 + supports.shape[0] * K), cout)

    def forward(self, x):                                      # x [B, N, C]
        outs = [x]
        for s in range(self.S.shape[0]):
            xk = x
            for _ in range(self.K):
                xk = torch.einsum('nm,bmc->bnc', self.S[s], xk)
                outs.append(xk)
        return self.lin(torch.cat(outs, -1))


class DCGRUCell(nn.Module):
    def __init__(self, cin, hid, supports, K):
        super().__init__()
        self.hid = hid
        self.gates = DConv(cin + hid, 2 * hid, supports, K)
        self.cand = DConv(cin + hid, hid, supports, K)

    def forward(self, x, h):
        zr = torch.sigmoid(self.gates(torch.cat([x, h], -1)))
        z, r = zr[..., :self.hid], zr[..., self.hid:]
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1)))
        return z * h + (1 - z) * c


class TrafficFlowST(nn.Module):
    """Diffusion graph-convolutional GRU encoder + direct 12-step head (+ last-speed skip connection)."""
    def __init__(self, supports, cin=6, hid=48, layers=1, K=2, horizon=12):
        super().__init__()
        self.hid = hid
        self.cells = nn.ModuleList([DCGRUCell(cin if i == 0 else hid, hid, supports, K) for i in range(layers)])
        self.head = nn.Linear(hid, horizon)

    def forward(self, x):                                      # x [B, 12, N, 6]
        B, T, N, _ = x.shape
        hs = [x.new_zeros(B, N, self.hid) for _ in self.cells]
        for t in range(T):
            inp = x[:, t]
            for i, c in enumerate(self.cells):
                hs[i] = c(inp, hs[i])
                inp = hs[i]
        out = self.head(hs[-1]) + x[:, -1, :, 0:1]            # [B, N, 12]
        return out.permute(0, 2, 1)                            # [B, 12, N] normalised speed