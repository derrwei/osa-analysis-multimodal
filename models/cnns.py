import torch
import torch.nn as nn
import torchaudio

import math

def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class CNN_OSA(nn.Module):
    """
    CNN backbone with 3 convolutional layers and fully connected layer.
    Architecture from paper:
    - 3 conv layers: 16, 32, 64 filters with 3x3 kernels
    - 4x3 max pooling after each conv layer
    - Batch normalization after each conv layer
    - ReLU activation, 0.3 dropout
    - Fully connected layer with 512 units
    
    Input: (B,1,1500,64)
    Output: (B,512) after flattening and FC layer
    """
    def __init__(self, dropout=0.3, norm='batch', num_classes=2, data_batch_norm=False):
        super().__init__()

        self.data_batch_norm = data_batch_norm
        if self.data_batch_norm:
            self.input_bn = nn.BatchNorm2d(64)
        
        def norm2d(norm_type='batch_frequency', n_channels=64):
            if norm_type == 'batch':
                return nn.BatchNorm2d(n_channels)
            elif norm_type == 'batch_frequency':
                return nn.BatchNorm2d(n_channels)
            elif norm_type == 'layer':
                return nn.LayerNorm(n_channels)
            elif norm_type == 'group':
                return nn.GroupNorm(num_groups=16, num_channels=n_channels)
            else:
                raise ValueError("Unknown norm type: {}".format(norm_type))

        # First convolutional layer: 16 filters, 3x3 kernel
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=0), # mono channel audio
            norm2d(n_channels=16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 3)),  # 4x3 max pooling
            # nn.Dropout(dropout),
        )
        
        # Second convolutional layer: 32 filters, 3x3 kernel
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=0),
            norm2d(n_channels=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 3)),  # 4x3 max pooling
            # nn.Dropout(dropout),
        )
        
        # Third convolutional layer: 64 filters, 3x3 kernel
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=0),
            norm2d(n_channels=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 3)),  # 4x3 max pooling
            # nn.Dropout(dropout),
        )
        
        # Calculate the flattened size after conv layers
        # This depends on input size (1500, 64) and pooling operations
        # We'll compute this dynamically in forward pass
        self._calculate_fc_input_size()
        # self._init_layers()
        
        # Fully connected layer with 512 units
        self.fc = nn.Sequential(
            nn.Linear(self._fc_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.fc_osa = nn.Sequential(
            nn.Linear(512, num_classes)
        )
    
    def _init_layers(self):
        """Initialize all layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init_layer(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                init_bn(m)

    def _calculate_fc_input_size(self):
        """Calculate the input size for the FC layer by doing a forward pass"""
        # Create a dummy input to calculate the size
        dummy_input = torch.zeros(1, 1, 1500, 64)  # (B, C, T, F)
        x = self.block1(dummy_input)
        x = self.block2(x)
        x = self.block3(x)
        self._fc_input_size = x.view(x.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape : (B, 1500, 64)
        # print(x.shape)
        x = x.float()
        (_, time, freq) = x.shape
        # x = x.view(-1, 1, time, freq)
        x = x.unsqueeze(1)  # (B, 1, T, F)
        if self.data_batch_norm:
            x = x.transpose(1, 3)  # (B, F, T, 1)
            x = self.input_bn(x)
            x = x.transpose(1, 3)  # (B, 1, T, F)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Flatten for FC layer
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        
        # Apply fully connected layer
        x = self.fc(x)
        x = self.fc_osa(x)
        
        return x  # (B, num_classes)


class CNN_OSA_Large(nn.Module):
    """
    CNN backbone with 3 convolutional layers and fully connected layer.
    Architecture from paper:
    - 3 conv layers: 16, 32, 64 filters with 3x3 kernels
    - 4x3 max pooling after each conv layer
    - Batch normalization after each conv layer
    - ReLU activation, 0.3 dropout
    - Fully connected layer with 512 units
    
    Input: (B,1,1500,64)
    Output: (B,512) after flattening and FC layer
    """
    def __init__(self, dropout=0.3, norm='batch', in_ch=1):
        super().__init__()
        
        def norm2d(c: int):
            if norm == 'batch':
                return nn.BatchNorm2d(c)
            elif norm in ('group', 'layer_channel'):
                return nn.GroupNorm(1, c)
            else:
                raise ValueError(f"Unknown norm: {norm}")

        # First convolutional layer: 16 filters, 3x3 kernel
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=(3, 3), padding=0),
            norm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 3)),  # 4x3 max pooling
            nn.Dropout(dropout),
        )
        
        # Second convolutional layer: 32 filters, 3x3 kernel
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=0),
            norm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 3)),  # 4x3 max pooling
            nn.Dropout(dropout),
        )
        
        # Third convolutional layer: 64 filters, 3x3 kernel
        self.cnn_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=0),
            norm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 3)),  # 4x3 max pooling
            nn.Dropout(dropout),
        )

        self.cnn_feature_extractor = nn.Sequential(
            self.cnn_block1,
            self.cnn_block2,
            self.cnn_block3,
        )
        ### -> flatten -> 512 embedding
        
        # Calculate the flattened size after conv layers
        # This depends on input size (1500, 64) and pooling operations
        # We'll compute this dynamically in forward pass
        self._calculate_fc_input_size()
        

        # ========= CNN LSTMs =========
        self.cnnlstm_block1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            norm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),   # T 1500->750, F 64->32
            nn.Dropout(dropout),
        )
        self.cnnlstm_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            norm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),   # 750->375, 32->16
            nn.Dropout(dropout),
        )
        self.cnnlstm_block3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            norm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),   # 375->187, 16->8
            nn.Dropout(dropout),
        )
        self.cnnlstm_block4 = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1),
            norm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,2)),   # 187->187, 8->4
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            # dropout=dropout
        )

        self.cnn_lstm_feature_extractor = nn.Sequential(
            self.cnnlstm_block1,
            self.cnnlstm_block2,
            self.cnnlstm_block3,
            self.cnnlstm_block4,
            # self.lstm,
        )

        # Fully connected layer with 512 units
        self.fc = nn.Sequential(
            nn.Linear(self._fc_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(512 + 2 * 128, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )
    
    def _calculate_fc_input_size(self):
        """Calculate the input size for the FC layer by doing a forward pass"""
        # Create a dummy input to calculate the size
        dummy_input = torch.zeros(1, 1, 1500, 64)
        x = self.cnn_feature_extractor(dummy_input)
        self._fc_input_size = x.view(x.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, T, F)
        cnn_features = self.cnn_feature_extractor(x)    
        # print(cnn_features.shape)
        # flatten
        cnn_features = cnn_features.flatten(start_dim=1)  # (B, feat_dim)
        cnn_embeddings = self.fc(cnn_features)  # (B, 512)

        # CNN LSTM features
        cnnlstm_x = self.cnn_lstm_feature_extractor(x)  # (B, T', 2*hidden_size)
        cnnlstm_x = cnnlstm_x.mean(-1)
        cnnlstm_x = cnnlstm_x.permute(0,2,1)
        cnnlstm_x, _ = self.lstm(cnnlstm_x)
        # Take the last time step
        cnnlstm_embeddings = cnnlstm_x[:, -1, :]
        # concate 
        fused = torch.cat([cnn_embeddings, cnnlstm_embeddings], dim=-1)
        fused = self.fusion(fused)
        return fused 

if __name__ == "__main__":
    # Test the CNN_OSA model
    model = CNN_OSA(dropout=0.3, norm='batch', num_classes=2)
    # print(model)
    # dummy_input = torch.randn(4, 1500, 64)  # Batch size of 4
    dummy_input = torch.rand(16000*3)
    print(dummy_input.shape)
    mel = torchaudio.transforms.MelSpectrogram()
    dummy_input = mel(dummy_input)
    print(dummy_input.shape)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (4, 2)