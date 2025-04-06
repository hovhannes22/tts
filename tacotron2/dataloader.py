import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import config
from datetime import datetime

class TextToSpeechDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, sample_rate=config.sample_rate, n_mels=config.n_mel_channels):
        # Load dataset metadata
        # self.data = pd.read_csv(metadata_path, delimiter='|', names=['filename', 'text', 'normalized_text'])
        
        # id|transcription
        self.data = pd.read_csv(metadata_path, delimiter='|')

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_wav_value = config.max_wav_value

        # Mel spectrogram parameters
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            f_min=0.0,
            f_max=8000.0,
            norm='slaney',
            mel_scale='slaney',
            power=1.0,
            window_fn=torch.hann_window
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        # Load and preprocess audio
        audio_path = os.path.join(self.audio_dir, f"{row['filename']}")
        if '.wav' not in audio_path:
            audio_path += '.wav'
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if sample rate is different
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Normalize audio
        # waveform = waveform / torch.max(torch.abs(waveform))
        waveform = waveform / self.max_wav_value

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Squeeze channel dimension and convert to log scale
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        return {
            'text': row['normalized_text'],
            'mel': mel_spec.squeeze(0),
        }

def build_vocab():
    # English (lowercase)
    armenian = [chr(c) for c in range(0x0561, 0x0588)]  # ա-ֆ, և
    english = [chr(c) for c in range(ord('a'), ord('z') + 1)] # a-z
    russian = [chr(c) for c in range(0x0430, 0x0450)] + ['ё'] # а-я
    
    # Standard punctuation (shared across languages)
    standard_punct = list('!\'(),-.:;? \"')
    
    # Armenian-specific punctuation (Unicode)
    armenian_punct = ['«', '»', '՞', '՜', '՛', '՚', '՝', '՟', '֊']
    
    # Combine all characters
    all_chars = russian + english + armenian + standard_punct + armenian_punct
    # all_chars = english + standard_punct
    
    # Create vocabulary (char → ID)
    return { char: idx + 1 for idx, char in enumerate(all_chars) }

def normalize_text(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Armenian ligatures
    # text = text.replace('եւ', 'և')
    
    return text

def tokenize_text(text, vocab):
    normalized_text = normalize_text(text)
    tokens = []
    
    for char in normalized_text:
        if char in vocab:
            tokens.append(vocab[char])
    
    return tokens

class CollateFn:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        # Process Texts
        texts = [ item['text'] for item in batch ] # List[str]
        
        # Tokenize to indices: List[List[int]]
        tokenized = [ tokenize_text(text, self.vocab) for text in texts ]
        
        # Pad texts to (batch_size, max_text_len)
        text_lengths = torch.tensor([len(t) for t in tokenized])  # (B,)
        max_text_len = text_lengths.max().item()
        text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)  # (B, T_text)
        for i, seq in enumerate(tokenized):
            text_padded[i, :len(seq)] = torch.tensor(seq)  # Fill valid tokens

        # Mel Spectrograms
        mels = [ item['mel'] for item in batch ]  # List[Tensor] where each is (n_mels, time)
        
        # Time dimensions
        mel_lengths = torch.tensor([mel.shape[1] for mel in mels])
        max_mel_time = mel_lengths.max().item()
        n_mels = mels[0].shape[0]

        # Initialize padded tensors
        mel_padded = torch.zeros(len(batch), n_mels, max_mel_time) # (B, n_mels, T)
        gate_padded = torch.zeros(len(batch), max_mel_time) # (B, T)

        for i, mel in enumerate(mels):
            mel_padded[i, :, :mel.shape[1]] = mel
            gate_padded[i, mel.shape[1]-1:] = 1.0  # Set stop token

        return {
            'text': text_padded, # (B, T_text)
            'text_lengths': text_lengths, # (B,)
            'mel': mel_padded, # (B, n_mels, T_mel)
            'mel_lengths': mel_lengths, # (B,)
            'gate_target': gate_padded, # (B, T_mel)
            'raw_text': texts # List[str]
        }
    
class HifiGanCollator:
    """Custom collator for HiFi-GAN training"""
    def __init__(self, hop_length):
        self.hop_length = hop_length
        
    def __call__(self, batch):
        # Sort batch by mel length (descending) for efficient padding
        batch.sort(key=lambda x: x['mel'].shape[1], reverse=True)
        
        mels = [item['mel'] for item in batch]
        audios = [item['audio'] for item in batch]
        
        # Get dimensions
        max_mel_len = max([mel.shape[1] for mel in mels])
        n_mels = mels[0].shape[0]
        max_audio_len = max_mel_len * self.hop_length
        
        # Create padded tensors
        mel_padded = torch.zeros(len(batch), n_mels, max_mel_len)
        audio_padded = torch.zeros(len(batch), max_audio_len)
        mel_lengths = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long)
        
        for i, (mel, audio) in enumerate(zip(mels, audios)):
            mel_padded[i, :, :mel.shape[1]] = mel
            audio_padded[i, :audio.shape[0]] = audio
            
        return {
            'mels': mel_padded,
            'audios': audio_padded,
            'mel_lengths': mel_lengths
        }

# Tacotron2
def get_dataloader_tacotron2(metadata_path, audio_dir, vocab, batch_size=16):
    dataset = TextToSpeechDataset(metadata_path, audio_dir)
    
    # Create an instance of the collate function
    collate_fn = CollateFn(vocab)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )

def tacotron2_visualize_mels(generated_mels, ground_truth_mels, raw_text=None, text=None, n_samples=4, cmap='magma', vmin=-12, vmax=0, epoch=None, step=None, base_dir=None):
    # Saving directory
    save_dir = os.path.join(base_dir, 'results')

    # Create the output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    B = ground_truth_mels.shape[0]
    n_samples = min(n_samples, B)

    generated_mels = generated_mels.detach().cpu().numpy()
    ground_truth_mels = ground_truth_mels.detach().cpu().numpy()
    # print(raw_text)
    # print(text)
    
    # Create a figure with n_samples rows and 2 columns (ground truth and generated)
    fig, axes = plt.subplots(n_samples, 2, figsize=(4*n_samples, 2*n_samples))
    
    # Ensure axes is always iterable as a list of pairs
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        gt_img = ground_truth_mels[i]
        gen_img = generated_mels[i]
        
        # Plot ground truth
        im = axes[i][0].imshow(gt_img, cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        # axes[i][0].set_title("Ground Truth")
        fig.colorbar(im, ax=axes[i][0])
        # axes[i][0].axis('off')
        
        # Plot generated image
        im = axes[i][1].imshow(gen_img, cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        # axes[i][1].set_title("Generated")
        fig.colorbar(im, ax=axes[i][1])
        # axes[i][1].axis('off')

        axes[0][0].set_title("Ground Truth")
        axes[0][1].set_title("Generated")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'train', f"train_{epoch}_{step}.png")
    plt.savefig(save_path)
    plt.close()
    # plt.show()
    print(f"Saved visualization to: {save_path}")