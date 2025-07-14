import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, out_len)

    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.lstm(x)  # out: [B, T, H*2]
        last = out[:, -1, :]  # 取最后时刻
        y = self.fc(last)  # [B, out_len]
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))  # 学习型位置编码
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        # x: [B, T, D]; pe[:, :T, :] 形状 [1, T, D]
        return x + self.pe[:, :x.size(1), :]


class TransformerForecaster(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, out_len, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, out_len)

    def forward(self, x):
        # x: [B, T, D]
        x = self.input_proj(x)  # [B, T, d_model]
        x = self.pos_enc(x)  # 加位置编码
        enc = self.encoder(x)  # [B, T, d_model]
        last = enc[:, -1, :]  # 取最后一个时刻
        return self.output_proj(last)  # [B, out_len]


class AdditiveTransformerUQ(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, out_len, dim_feedforward=256, dropout=0.1):
        super().__init__()
        # 输入映射与位置编码
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, 1000, d_model))  # 可学习的位置编码
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 均值与 log-variance 投影
        self.mean_proj = nn.Linear(d_model, out_len)
        self.logvar_proj = nn.Linear(d_model, out_len)

    def forward(self, x):
        # x: [B, T, D]
        B, T, _ = x.shape
        x = self.input_proj(x)  # [B, T, d_model]
        x = x + self.pos_enc[:, :T, :]  # 加位置编码
        enc = self.encoder(x)  # [B, T, d_model]
        last = enc[:, -1, :]  # 取最后时刻特征
        mean = self.mean_proj(last)  # 预测均值 [B, out_len]
        logvar = self.logvar_proj(last)  # 预测 log-variance [B, out_len]
        return mean, logvar


class BayesianTransformerQuantile(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, out_len, quantiles=[0.1, 0.5, 0.9], dim_feedforward=256,
                 dropout=0.1):
        super().__init__()
        self.quantiles = quantiles
        self.K = len(quantiles)
        self.out_len = out_len

        # 输入映射层
        self.input_proj = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Bayesian Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Quantile Regression 输出层
        self.output_proj = nn.Linear(d_model, out_len * self.K)

    def forward(self, x):
        """
        x: [B, T, D]
        return: [B, out_len, K]
        """
        B = x.size(0)

        x = self.input_proj(x)  # [B, T, d_model]
        x = self.pos_enc(x)  # 加位置编码
        enc = self.encoder(x)  # [B, T, d_model]
        enc = self.dropout(enc)  # Bayesian Dropout

        last = enc[:, -1, :]  # 取最后一个时间步 [B, d_model]
        out = self.output_proj(last)  # [B, out_len * K]
        out = out.view(B, self.out_len, self.K)  # [B, out_len, K]
        return out


def quantile_loss(y_pred, y_true, quantiles):
    """
    y_pred: [B, out_len, K]
    y_true: [B, out_len]
    quantiles: list of quantile levels
    return: scalar loss
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[:, :, i]
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss.unsqueeze(2))
    loss_tensor = torch.cat(losses, dim=2)  # [B, out_len, K]
    return loss_tensor.mean()
