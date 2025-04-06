import torch
import torch.nn as nn

import config

class Encoder(nn.Module):
    def __init__(self, vocab_size, encoder_embedding_dim=config.encoder_embedding_dim, dropout=0.5):
        super().__init__()
        kernel_size = config.encoder_kernel_size
        padding = int((kernel_size - 1) / 2)
        self.dropout = dropout

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, encoder_embedding_dim)
        
        # 3 Conv Layers
        self.conv1 = nn.Conv1d(encoder_embedding_dim, encoder_embedding_dim, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(encoder_embedding_dim)

        self.conv2 = nn.Conv1d(encoder_embedding_dim, encoder_embedding_dim, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(encoder_embedding_dim)

        self.conv3 = nn.Conv1d(encoder_embedding_dim, encoder_embedding_dim, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm1d(encoder_embedding_dim)

        # Xavier uniform init
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu'))

        # Bidirectional LSTM
        self.lstm = nn.LSTM(encoder_embedding_dim, encoder_embedding_dim//2, bidirectional=True, batch_first=True)

    def forward(self, text, text_lengths):
        # text: (B, T_enc)
        # text_lengths: (B, )

        # Embedding
        x = self.embedding(text) # (B, T_enc, encoder_embedding_dim)
        x = x.transpose(-2, -1) # (B, encoder_embedding_dim, T_enc)
        
        # Convolutions
        x = torch.dropout(torch.relu(self.bn1(self.conv1(x))), self.dropout, train=self.training) # (B, encoder_embedding_dim, T_enc)
        x = torch.dropout(torch.relu(self.bn2(self.conv2(x))), self.dropout, train=self.training) # (B, encoder_embedding_dim, T_enc)
        x = torch.dropout(torch.relu(self.bn3(self.conv3(x))), self.dropout, train=self.training) # (B, encoder_embedding_dim, T_enc)
        x = x.transpose(-2, -1) # (B, T_enc, encoder_embedding_dim)

        # Packing x to skip paddings in LSTM
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=text_lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        self.lstm.flatten_parameters()

        # LSTM
        out, _ = self.lstm(x)

        # Unpacking x to restore paddings
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # (B, T_enc, encoder_embedding_dim)
        return out

    @torch.inference_mode()
    def inference(self, text):
        # text: (B, T_enc)

        # Embedding
        x = self.embedding(text) # (B, T_enc, encoder_embedding_dim)
        x = x.transpose(-2, -1) # (B, encoder_embedding_dim, T_enc)
        
        # Convolutions
        x = torch.dropout(torch.relu(self.bn1(self.conv1(x))), self.dropout, train=self.training) # (B, encoder_embedding_dim, T_enc)
        x = torch.dropout(torch.relu(self.bn2(self.conv2(x))), self.dropout, train=self.training) # (B, encoder_embedding_dim, T_enc)
        x = torch.dropout(torch.relu(self.bn3(self.conv3(x))), self.dropout, train=self.training) # (B, encoder_embedding_dim, T_enc)
        x = x.transpose(-2, -1) # (B, T_enc, encoder_embedding_dim)

        # Packing x to skip paddings in LSTM
        self.lstm.flatten_parameters()

        # LSTM
        out, _ = self.lstm(x)

        return out