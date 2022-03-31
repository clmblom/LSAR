import torch.nn as nn
from .positional_encoding import PositionalEncoding


class HTREncoder(nn.Module):
    def __init__(self, hidden_size=None, feature_size=None, feature_width=None, feature_height=None):
        super(HTREncoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(feature_height*feature_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout=0.1, max_len=feature_width) # feature_width is the width of the feature map from the CNN

    def forward(self, x):
        # x has shape batch, f, h, w
        # output has shape: w, b, hidden_size
        batch, feature, height, width = x.shape
        x = x.permute(3, 0, 1, 2) # shape: w, b, f, h
        x = x.reshape(width, batch, height*feature) # w, b, h x f
        out = self.fc(x) # w, b, hidden_size
        out = self.pe(out) # w, b, hidden_size
        return out
