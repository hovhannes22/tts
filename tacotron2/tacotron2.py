import torch
import torch.nn as nn

from encoder import Encoder as Encoder
from decoder import Decoder, PostNet
import loss as loss

import dataloader as data_loader

import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import config

# Boolean mask from sequence lengths
def get_mask(lengths):
    max_len = torch.max(lengths).item()
    device = lengths.device
    indices = torch.arange(0, max_len, device=device, dtype=torch.long)
    return (indices < lengths.unsqueeze(1)).bool() # (B, max_length)

class Tacotron2(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.mask_padding = True
        self.n_mels = config.n_mel_channels

        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder()

        # Postnet
        self.postnet = PostNet(self.n_mels, config.postnet_embedding_dim)

    def forward(self, text, text_lengths, mel_target, mel_lengths):
        encoder_output = self.encoder(text, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_output, mel_target, text_lengths)
        
        # PostNet
        postnet_outputs = self.postnet(mel_outputs) + mel_outputs  # (B, n_mels, T_mel)

        # Apply padding mask directly on the outputs
        if self.mask_padding and mel_lengths is not None:
            mask = ~get_mask(mel_lengths)

            # Expanding the mask to match the mel spectrogram shape
            mask = mask.expand(self.n_mels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            # Apply the mask to the mel spectrogram outputs (set padded regions to 0)
            mel_outputs.data.masked_fill_(mask, 0.0)
            postnet_outputs.data.masked_fill_(mask, 0.0)
            gate_outputs.data.masked_fill_(mask[:, 0, :], 1e3)

        return mel_outputs, postnet_outputs, gate_outputs, alignments

    @torch.inference_mode()
    def inference(self, text):
        encoder_output = self.encoder.inference(text)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_output)

        # PostNet
        postnet_outputs = self.postnet(mel_outputs) + mel_outputs  # (B, n_mels, T_mel)

        return mel_outputs, postnet_outputs, gate_outputs, alignments # (B, n_mels, T_mel); (B, T_mel); (B, T_mel, T_enc)

# Load checkpoint if available
def load_latest_checkpoint(model, optimizer, device):
    dir = os.path.join(BASE_DIR, 'checkpoints')
    checkpoint_files = glob.glob(f"{dir}/checkpoint_step_*.pth")
    if not checkpoint_files:
        print("No checkpoint found. Starting training from scratch.")
        return 0, 0, 0, []  # start_epoch, global_step, start_batch_index
    # Find the checkpoint with the highest global step
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.search(r"checkpoint_step_(\d+).pth", x).group(1)))
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    start_batch_index = checkpoint.get('step_in_epoch', 0)
    loss_history = checkpoint.get("loss_history", [])
    print(f"Loaded checkpoint: {latest_checkpoint} (Epoch: {start_epoch}, Step: {global_step}, Batch in epoch: {start_batch_index})")
    return start_epoch, global_step, start_batch_index, loss_history

# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, step, global_step, loss_history):
    checkpoint = {
        "epoch": epoch,
        "step_in_epoch": step,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history
    }
    checkpoint_path = os.path.join(BASE_DIR, 'checkpoints', f'checkpoint_step_{global_step}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {global_step} (Epoch {epoch})")


def visualize_loss(loss_history, step, x_len=250):
    if len(loss_history) > x_len:
        loss_history = loss_history[-x_len:]

    # Save the loss plot
    plt.figure(figsize=(12, 5))
    plt.plot(loss_history, label="Total Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.grid(0.2)
    plt.savefig(os.path.join(BASE_DIR, 'results', 'loss', f'loss_step_{step}.png'))
    plt.close()

def train(model, train_loader, optimizer):
    # Load checkpoint if available
    start_epoch, global_step, start_batch_index, loss_history = load_latest_checkpoint(model, optimizer, device)
    
    model.train()
    for epoch in range(start_epoch, epochs):
        for step, batch in enumerate(train_loader):
            # If we are still in the resumed epoch, skip batches before start_batch_index.
            if epoch == start_epoch and step <= start_batch_index:
                continue
            
            global_step += 1
            
            # Move data to the device
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel_target = batch['mel'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            gate_target = batch['gate_target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            mel_outputs, mel_postnet, gate_outputs, _ = model(text, text_lengths, mel_target, mel_lengths)

            # Sampling visualization every sample_every steps
            if global_step % sample_every == 0:
                data_loader.tacotron2_visualize_mels(
                    generated_mels=mel_postnet[:n_samples],
                    ground_truth_mels=mel_target[:n_samples],
                    raw_text=batch['raw_text'][:n_samples],
                    text=text[:n_samples],
                    epoch=epoch,
                    step=step,
                    n_samples=n_samples,
                    base_dir=BASE_DIR
                )
                visualize_loss(loss_history, global_step - 1)
            
            # Calculate loss and backward pass
            loss, (total_mel_loss, total_gate_loss) = loss_function(
                mel_postnet, mel_outputs, gate_outputs,
                mel_target, gate_target
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track loss
            loss_history.append(loss.item())
            
            # Save checkpoint every 'save_every' steps
            if global_step % save_every == 0:
                save_checkpoint(model, optimizer, epoch, step, global_step, loss_history)


            print(f"Epoch: {epoch:4}/{epochs:4}; Step: {global_step:6}; Loss: {loss.item():.4f}; Mel Loss: {total_mel_loss.item():.4f}; Gate Loss: {total_gate_loss.item():.4f}")

if __name__ == "__main__":
    # Base dir
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Vocabulary
    vocab = data_loader.build_vocab()
    vocab_size = len(vocab) + 1

    # Paths
    # metadata_path = './data/metadata.csv'
    # audio_dir = './data/wavs/'

    metadata_path = os.path.join(BASE_DIR, '..', 'data', 'metadata.csv')
    audio_dir = os.path.join(BASE_DIR, '..', 'data', 'wavs')

    # Dataloader
    batch_size = config.batch_size
    dataloader = data_loader.get_dataloader_tacotron2(metadata_path, audio_dir, vocab, batch_size=batch_size)
    print(f"Dataloader len: {len(dataloader)}")
    n_samples = config.n_samples # Number of samples to visualize during training

    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config.epochs
    save_every = config.save_every # Saves model every N epochs during training
    sample_every = config.sample_every # Samples every N epochs during training

    model_tacotron2 = Tacotron2(vocab_size=vocab_size).to(device)
    optimizer_tacotron2 = torch.optim.Adam(model_tacotron2.parameters(), lr=config.tacotron2_lr)
    loss_function = loss.Tacotron2Loss()

    train(model_tacotron2, dataloader, optimizer_tacotron2)