import torch
import torch.nn as nn

# Tacotron2 Loss
class Tacotron2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_loss = nn.MSELoss()  # For spectrogram reconstruction
        self.gate_loss = nn.BCEWithLogitsLoss()  # For stop token prediction

    def forward(self, mel_postnet, mel_outputs, gate_outputs, mel_target, gate_target):
        # mel_postnet: (B, n_mel_channels, T) Post-net processed mel predictions
        # mel_outputs: (B, n_mel_channels, T) Pre-net mel predictions  
        # gate_outputs: (B, T) Stop token logits
        # mel_target: (B, T, n_mel_channels) Ground truth mel
        # gate_target: (B, T) Ground truth stop tokens (0 or 1)

        gate_target = gate_target.view(-1)

        # Mel-spectrogram reconstruction loss (both pre and post-net)
        prenet_mel_loss = self.mel_loss(mel_outputs, mel_target)
        postnet_mel_loss = self.mel_loss(mel_postnet, mel_target)
        total_mel_loss = prenet_mel_loss + postnet_mel_loss

        gate_outputs = gate_outputs.view(-1)

        # Stop token prediction loss
        total_gate_loss = self.gate_loss(gate_outputs, gate_target)

        # Combined loss
        total_loss = total_mel_loss + total_gate_loss
        
        return total_loss, (total_mel_loss, total_gate_loss)