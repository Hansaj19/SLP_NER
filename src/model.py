"""
PyTorch Recurrent Neural Network (RNN / LSTM / GRU) for Named Entity Recognition.
"""
import torch
import torch.nn as nn
from src.data_utils import PAD_IDX

class RNNSequenceTagger(nn.Module):
    """
    Sequence Tagging Model using Recurrent Neural Networks in PyTorch.
    
    Supports:
      - 'rnn'  : SimpleRNN (torch.nn.RNN)
      - 'lstm' : Long Short-Term Memory (torch.nn.LSTM)
      - 'gru'  : Gated Recurrent Unit (torch.nn.GRU)
    """
    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        cell_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.2
    ):
        super(RNNSequenceTagger, self).__init__()
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # 1. Embedding Layer: Maps word indices to dense continuous vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX
        )

        # 2. Regularization Dropout
        self.dropout = nn.Dropout(p=dropout)

        # 3. Recurrent Layer
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif self.cell_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                nonlinearity="tanh",
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unsupported cell_type '{cell_type}'. Choose 'lstm', 'gru', or 'rnn'.")

        # 4. Emission / Output Classification Layer: Maps hidden state to BIO tag scores
        self.fc = nn.Linear(
            hidden_dim * self.num_directions,
            tagset_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence labeling.
        Args:
          x: LongTensor of shape (batch_size, seq_len)
        Returns:
          logits: FloatTensor of shape (batch_size, seq_len, tagset_size)
        """
        # Step 1: Embedding lookup -> (batch_size, seq_len, embedding_dim)
        embeds = self.dropout(self.embedding(x))

        # Step 2: Pass through recurrent network
        # Output shape: (batch_size, seq_len, hidden_dim * num_directions)
        if self.cell_type == "lstm":
            rnn_out, (hn, cn) = self.rnn(embeds)
        else:
            rnn_out, hn = self.rnn(embeds)

        rnn_out = self.dropout(rnn_out)

        # Step 3: Project hidden representations to class scores (logits)
        # Logits shape: (batch_size, seq_len, tagset_size)
        logits = self.fc(rnn_out)

        return logits
