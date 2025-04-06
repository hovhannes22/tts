import torch
import torch.nn as nn

import config

class PreNet(nn.Module):
    def __init__(self, n_mels, prenet_hidden_dim, dropout=config.prenet_dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_mels, prenet_hidden_dim, bias=False)
        self.fc2 = nn.Linear(prenet_hidden_dim, prenet_hidden_dim, bias=False)
        self.dropout = dropout

        # Xavier uniform init
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        # x: (B, n_mels)
        x = torch.dropout(torch.relu(self.fc1(x)), p=self.dropout, train=self.training) # (B, prenet_hidden_dim)
        x = torch.dropout(torch.relu(self.fc2(x)), p=self.dropout, train=self.training) # (B, prenet_hidden_dim)
        return x

class PostNet(nn.Module):
    def __init__(self, n_mels, postnet_embedding_dim):
        super().__init__()
        kernel_size = config.postnet_kernel_size
        padding = int((kernel_size - 1) / 2)
        self.dropout = config.postnet_dropout

        self.conv1 = nn.Conv1d(n_mels, postnet_embedding_dim, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(postnet_embedding_dim)

        self.conv2 = nn.Conv1d(postnet_embedding_dim, postnet_embedding_dim, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(postnet_embedding_dim)

        self.conv3 = nn.Conv1d(postnet_embedding_dim, postnet_embedding_dim, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm1d(postnet_embedding_dim)

        self.conv4 = nn.Conv1d(postnet_embedding_dim, postnet_embedding_dim, kernel_size=kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm1d(postnet_embedding_dim)

        self.conv5 = nn.Conv1d(postnet_embedding_dim, n_mels, kernel_size=kernel_size, padding=padding)
        self.bn5 = nn.BatchNorm1d(n_mels)

        # Xavier uniform init
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.conv5.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        # x: (B, n_mels, T_dec) mel-spectogram
        x = torch.dropout(torch.tanh(self.bn1(self.conv1(x))), self.dropout, train=self.training) # (B, postnet_embedding_dim, T_dec)
        x = torch.dropout(torch.tanh(self.bn2(self.conv2(x))), self.dropout, train=self.training) # (B, postnet_embedding_dim, T_dec)
        x = torch.dropout(torch.tanh(self.bn3(self.conv3(x))), self.dropout, train=self.training) # (B, postnet_embedding_dim, T_dec)
        x = torch.dropout(torch.tanh(self.bn4(self.conv4(x))), self.dropout, train=self.training) # (B, postnet_embedding_dim, T_dec)
        x = torch.dropout(self.bn5(self.conv5(x)), self.dropout, train=self.training) # (B, n_mel_channels, T_dec)
        return x

class LocationAttention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, attention_n_filters, attention_kernel_size):
        super().__init__()
        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.weight_vector = nn.Parameter(torch.randn(attention_dim, 1))

        self.location_conv = nn.Conv1d(2, attention_n_filters, kernel_size=attention_kernel_size, padding=int((attention_kernel_size - 1) / 2), bias=False)
        self.location_dense = nn.Linear(attention_n_filters, attention_dim, bias=False)

        # Xavier uniform init
        nn.init.xavier_uniform_(self.query_layer.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.memory_layer.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.location_dense.weight, gain=nn.init.calculate_gain('tanh'))

    def forward(self, query, memory, processed_memory, attention_weights_cat, mask=None):
        # query: (B, attention_rnn_dim) attention hidden state
        # memory: (B, T_enc, encoder_embedding_dim) encoder output
        # processed_memory: (B, T_enc, attention_dim) processed encoder output
        # attention_weights_cat: (B, 2, T_enc) attention_weights_cum and previous attention weights
        # mask: (B, T_enc) mask for padding positions
        
        # Attention energies (scores)
        q = self.query_layer(query.unsqueeze(1)) # (B, 1, attention_dim)

        # Location features from previous attention
        location_features = self.location_conv(attention_weights_cat) # (B, attention_n_filters, T_enc)
        location_features = location_features.transpose(1, 2) # (B, T_enc, attention_n_filters)
        location_features = self.location_dense(location_features) # (B, T_enc, attention_dim)

        # Raw alignment (energies) scores
        energies = torch.tanh(q + processed_memory + location_features) @ self.weight_vector # (B, T_enc, attention_dim) @ (B, attention_dim, 1) -> (B, T_enc, 1)
        energies = energies.squeeze(-1) # (B, T_enc)

        # Mask
        if mask is not None:
            energies.data.masked_fill_(mask, -float('inf')) # (B, T_enc)

        # Attention weights
        attention_weights = torch.softmax(energies, dim=-1) # (B, T_enc)

        # Attention context
        attention_context = attention_weights.unsqueeze(-1) * memory # (B, T_enc, 1) @ (B, T_enc, encoder_embedding_dim) -> (B, T_enc, encoder_embedding_dim)
        attention_context = torch.sum(attention_context, dim=-2) # (B, encoder_embedding_dim)

        return attention_context, attention_weights # (B, encoder_embedding_dim); (B, T_enc)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters
        self.prenet_dim = config.prenet_hidden_dim
        self.postnet_dim = config.postnet_embedding_dim
        self.attention_rnn_dim = config.attention_rnn_dim
        self.decoder_rnn_dim = config.decoder_rnn_dim
        self.encoder_embedding_dim = config.encoder_embedding_dim
        self.n_mels = config.n_mel_channels

        self.gate_threshold = config.gate_threshold
        self.max_decoder_steps = config.max_decoder_steps

        # Prenet
        self.prenet = PreNet(self.n_mels, self.prenet_dim)

        # Location Attention
        self.location_attention = LocationAttention(
            attention_rnn_dim=config.attention_rnn_dim,
            embedding_dim=config.encoder_embedding_dim,
            attention_dim=config.attention_dim,
            attention_n_filters=config.attention_n_filters,
            attention_kernel_size=config.attention_kernel_size
        )
        
        # Attention RNN
        self.attention_rnn_cell = nn.LSTMCell(self.prenet_dim + self.encoder_embedding_dim, self.attention_rnn_dim)

        # Decoder RNN
        self.decoder_rnn_cell = nn.LSTMCell(self.attention_rnn_dim + self.encoder_embedding_dim, self.decoder_rnn_dim)
        
        # Linear Projection to mel-spectrogram
        self.linear_projection = nn.Linear(self.decoder_rnn_dim + self.encoder_embedding_dim, self.n_mels)
        
        # Linear Projection to Stop Token
        self.gate_projection = nn.Linear(self.decoder_rnn_dim + self.encoder_embedding_dim, 1, bias=True)

        # Xavier unifrom init
        nn.init.xavier_uniform_(self.gate_projection.weight, gain=nn.init.calculate_gain('sigmoid'))

    def _init_states(self, encoder_outputs, mask):
        # encoder_outputs: (B, T_enc, encoder_embedding_dim) | T_enc is the maximum time
        B, T_enc, _ = encoder_outputs.shape
        
        # RNN states
        self.attention_hidden = torch.zeros(B, self.attention_rnn_dim).to(encoder_outputs.device) # (B, attention_rnn_dim)
        self.attention_cell = torch.zeros(B, self.attention_rnn_dim).to(encoder_outputs.device) # (B, attention_rnn_dim)

        # Decoder states
        self.decoder_hidden = torch.zeros(B, self.decoder_rnn_dim).to(encoder_outputs.device) # (B, decoder_rnn_dim)
        self.decoder_cell = torch.zeros(B, self.decoder_rnn_dim).to(encoder_outputs.device) # (B, decoder_rnn_dim)

        # Attention weights
        self.attention_weights = torch.zeros(B, T_enc).to(encoder_outputs.device) # (B, T_enc)
        self.attention_weights_cum = torch.zeros(B, T_enc).to(encoder_outputs.device) # (B, T_enc)

        # Context
        self.attention_context = torch.zeros(B, self.encoder_embedding_dim).to(encoder_outputs.device) # (B, encoder_embedding_dim)

        # Memory
        self.memory = encoder_outputs # (B, T_enc, encoder_embedding_dim)
        self.processed_memory = self.location_attention.memory_layer(encoder_outputs) # (B, T_enc, attention_dim)

        # Mask
        self.mask = mask

    def decode_step(self, prev_mel):
        # prev_mel: (B, prenet_hidden_dim) previous mel frame

        # Concat prenet output and attention context (before attention)
        attention_rnn_input = torch.cat([prev_mel, self.attention_context], dim=-1) # (B, prenet_hidden_dim + encoder_embedding_dim)

        # Attention RNN
        self.attention_hidden, self.attention_cell = self.attention_rnn_cell(attention_rnn_input, (self.attention_hidden, self.attention_cell)) # (B, attention_rnn_dim)
        self.attention_hidden = torch.dropout(self.attention_hidden, p=config.p_attention_dropout, train=self.training) # (B, attention_rnn_dim)

        # Location attention weights and context vector
        attention_weights_cat = torch.stack([self.attention_weights, self.attention_weights_cum], dim=1) # (B, 2, T_enc)
        self.attention_context, self.attention_weights = self.location_attention(self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask) # (B, encoder_embedding_dim); # (B, T_enc)

        # Update cumulative attention weights
        self.attention_weights_cum += self.attention_weights # (B, T_enc)

        # Decoder RNN
        decoder_rnn_input = torch.cat([self.attention_hidden, self.attention_context], dim=-1) # (B, attention_rnn_dim + encoder_embedding_dim)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn_cell(decoder_rnn_input, (self.decoder_hidden, self.decoder_cell)) # (B, decoder_rnn_dim)
        self.decoder_hidden = torch.dropout(self.decoder_hidden, p=config.p_decoder_dropout, train=self.training) # (B, decoder_rnn_dim)

        # Projections
        decoder_output = torch.cat([self.decoder_hidden, self.attention_context], dim=-1) # (B, decoder_rnn_dim + encoder_embedding_dim)
        mel_output = self.linear_projection(decoder_output) # (B, n_mels)
        gate_output = self.gate_projection(decoder_output) # (B, 1)

        return mel_output, gate_output, self.attention_weights

    def forward(self, memory, mel_target, memory_lengths):
        # memory: (B, T_enc, encoder_embedding_dim) Encoder outputs
        # mel_target: (B, n_mels, T_mel)
        # memory_lengths: () Encoder output lengths for attention mask
        B, _, T_mel = mel_target.shape

        # Initial mel-spectogram
        prev_mel = torch.zeros(B, self.n_mels, device=memory.device) # (B, n_mels)
        prev_mel = prev_mel.unsqueeze(0) # (1, B, n_mels)

        decoder_inputs = mel_target.permute(2, 0, 1) # (T_mel, B, n_mels)

        decoder_inputs = torch.cat((prev_mel, decoder_inputs), dim=0) # (T_mel + 1, B, n_mels)

        decoder_inputs = self.prenet(decoder_inputs) # (T_mel + 1, B, prenet_hidden_dim)

        # Mask
        # Maximum sequence length in the batch
        max_length = torch.max(memory_lengths).item()
        
        # Generate an array of indices [0, ..., max_length - 1]
        ids = torch.arange(0, max_length, device=memory.device)
        
        # Create mask
        mask = (ids < memory_lengths.unsqueeze(1)).bool()

        # Initialize decoder states with mask
        self._init_states(memory, mask=~mask)

        # Outputs
        mel_outputs, gate_outputs, alignments = [], [], []

        # Loop until we have enough mel outputs
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            # Next input for the decoder
            prev_mel = decoder_inputs[len(mel_outputs)] # (B, prenet_hidden_dim)

            # Run decoder step
            mel_output, gate_output, attention_weights = self.decode_step(prev_mel) # (B, n_mels); (B, 1); (B, T_enc)
            
            # Store outputs
            mel_outputs.append(mel_output.squeeze(-1)) # (B, n_mels)
            gate_outputs.append(gate_output.squeeze(-1)) # (B, )
            alignments.append(attention_weights) # (B, T_enc)

        # Stack outputs
        mel_outputs = torch.stack(mel_outputs).permute(1, 2, 0).contiguous()  # (B, n_mels, T_mel)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()  # (B, T_mel)
        alignments = torch.stack(alignments).transpose(0, 1) # (B, T_enc, T_mel)

        return mel_outputs, gate_outputs, alignments # (B, n_mels, T_mel); (B, T_mel); (B, T_mel, T_enc)
    
    @torch.inference_mode()
    def inference(self, memory):
        # memory: (B, T_enc, encoder_embedding_dim) Encoder outputs

        B = memory.size(0)

        # Initial mel-spectogram
        prev_mel = torch.zeros(B, self.n_mels, device=memory.device) # (B, n_mels)
        prev_mel = prev_mel.unsqueeze(0) # (1, B, n_mels)

        self._init_states(memory)
        
        mel_outputs, gate_outputs, alignments = [], [], []
        
        while True:
            prev_mel = self.prenet(prev_mel)

            # Run decoder step
            mel_output, gate_output, attention_weights = self.decode_step(prev_mel) # (B, n_mels); (B, 1); (B, T_enc)
            
            # Store outputs
            mel_outputs.append(mel_output.squeeze(-1)) # (B, n_mels)
            gate_outputs.append(gate_output.squeeze(-1)) # (B, )
            alignments.append(attention_weights) # (B, T_enc)
            
            # Stop when probability when > gate_threshold
            if torch.sigmoid(gate_output.squeeze()) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Reached max decoder steps")
                break
            
            prev_mel = mel_output  # Use predicted frame for next step
        
        # Stack outputs
        mel_outputs = torch.stack(mel_outputs).permute(1, 2, 0).contiguous()  # (B, n_mels, T_mel)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()  # (B, T_mel)
        alignments = torch.stack(alignments).transpose(0, 1) # (B, T_enc, T_mel)

        return mel_outputs, gate_outputs, alignments # (B, n_mels, T_mel); (B, T_mel); (B, T_mel, T_enc)
        
        # Stack outputs
        mel_outputs = torch.stack(mel_outputs, dim=2)
        gate_outputs = torch.stack(gate_outputs, dim=1)
        alignments = torch.stack(alignments, dim=1)
        
        postnet_outputs = self.postnet(mel_outputs) + mel_outputs
        return postnet_outputs.squeeze(), mel_outputs.squeeze(), alignments.squeeze()