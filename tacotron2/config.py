# Audio
sample_rate = 22050
n_mel_channels = 80
filter_length = 1024
hop_length = 256
win_length = 1024
n_fft = 1024
max_wav_value = 1.0
n_mel_channels = 80

# Encoder
# vocab_size = 256
prenet_dropout = 0.5
prenet_hidden_dim = 256

encoder_embedding_dim = 512
encoder_kernel_size = 5

# Location Attention
attention_dim = 128
attention_rnn_dim = 1024
attention_n_filters = 32
attention_kernel_size = 31

# Decoder
decoder_rnn_dim = 1024
location_channels = 32
p_decoder_dropout = 0.1
p_attention_dropout = 0.1
max_decoder_steps = 1000
postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_dropout = 0.5
gate_threshold = 0.5

# Tacotron2 Parameters
tacotron2_lr = 1e-3
mask_padding = True

# Model Parameters
batch_size = 16
epochs = 20
save_every = 50
sample_every = 25
n_samples = 4