import torch
import torch.nn as nn

class HubertClassificationHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(HubertClassificationHead, self).__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        # We take the mean over the sequence length dimension
        x = x.float()
        x = self.dropout(x)
        x = x.mean(dim=1) if x.dim() == 3 else x  # (batch_size, embedding_dim)
        logits = self.classifier(x)
        return logits
    
if __name__ == "__main__":
    # read one sample Hubert feature tensor
    import numpy as np
    sample_feature = np.load('/export/catch2/users/xiaolei/projects/BRAHMS/data/osa-brahms/features/hubert_base_samples/040/2020-01-28/sample_2.npy')
    sample_tensor = torch.from_numpy(sample_feature).unsqueeze(0)  # add batch dimension
    print('Sample tensor shape:', sample_tensor.shape)  # should be (1, seq_len, embedding_dim) 
    # create classification head
    embedding_dim = sample_tensor.shape[2]
    num_classes = 2  # e.g., OSA vs non-OSA
    classification_head = HubertClassificationHead(embedding_dim, num_classes)
    # forward pass
    logits = classification_head(sample_tensor)
    print('Logits shape:', logits.shape)  # should be (1, num_classes