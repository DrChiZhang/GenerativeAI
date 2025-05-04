import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Acts as a base for this and many
    other models using a sequence-to-sequence approach.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        # Initialize the encoder and decoder components
        self.encoder = encoder
        self.decoder = decoder
        
        # Initialize embedding layers for source and target sequences
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        
        # Initialize the generator for producing final output predictions
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Defines the forward pass of the model.
        
        Parameters:
        - src: Source sequence input
        - tgt: Target sequence input
        - src_mask: Mask for the source sequence to ignore padding in attention
        - tgt_mask: Mask for the target sequence to prevent future token attention
        
        Returns:
        - Decoded output from the target input and encoded source context
        """
        # Encode the source sequence and then decode using the encoded context
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        Encodes the source sequence.
        
        Parameters:
        - src: Source sequence input
        - src_mask: Mask for the source sequence
        
        Returns:
        - Contextual (hidden) representations from the encoder
        """
        # Embed the source sequence and pass it through the encoder
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Decodes the target sequence using the encoded source context.
        
        Parameters:
        - memory: Encoded context from the source sequence
        - src_mask: Mask for the source sequence
        - tgt: Target sequence input
        - tgt_mask: Mask for the target sequence
        
        Returns:
        - Transformed target sequence predictions
        """
        # Embed the target sequence, then pass it through the decoder 
        # with the encoder's context
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    """
    Produce N identical layers.
    
    This function creates a list of N deep-copied instances of a given module.
    Each instance is a separate object with its own parameters.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

"""
The following classes are used to implement the Encoder and Decoder layers.
"""
class LayerNorm(nn.Module):
    """
    Construct a layer normalization module.
    
    Layer normalization normalizes the input over the features dimension.
    This helps stabilize the learning and improve the convergence rate.
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # Learnable parameters for scaling and shifting the normalized output
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        
        # Small constant for numerical stability during division
        self.eps = eps

    def forward(self, x):
        # Compute mean and standard deviation along the last dimension
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        # Apply layer normalization with learnable parameters
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer normalization.
    
    This pattern is used in Transformer models to stabilize the training
    of deep networks.
    
    Note: In this implementation, the norm is applied before the sublayer
    application for simplicity.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        # Initialize layer normalization
        self.norm = LayerNorm(size)
        # Initialize dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        
        Parameters:
        - x: Input to the sublayer
        - sublayer: Function representing the sublayer to be applied
        
        Returns:
        - Output after applying residual connection, norm, and dropout
        """
        # Apply normalization, sublayer, dropout, and add residual connection
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """
    Defines a single encoder layer composed of self-attention and
    feed-forward network.
    
    Each layer in the encoder is designed as per the Transformer architecture.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        # Self-attention mechanism for this encoder layer
        self.self_attn = self_attn
        # Feed-forward network for processing the attention outputs
        self.feed_forward = feed_forward
        # Two sublayers, one for each operation, wrapped with residual connections
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Apply self-attention and feed-forward network in sequence.
        
        Parameters:
        - x: Input to the encoder layer
        - mask: Mask to be applied in the self-attention mechanism
        
        Returns:
        - Transformed output after the encoder layer operations
        """
        # Apply self-attention followed by a feed-forward network
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    """
    Core encoder is a stack of N identical layers.
    
    The encoder processes input sequences and generates contexual representations.
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # Create N identical encoder layers
        self.layers = clones(layer, N)
        # Final layer normalization after passing through all encoder layers
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.
        
        Parameters:
        - x: Input sequence
        - mask: Mask indicating allowed and disallowed positions in attention
        
        Returns:
        - Final normalized output after processing through all encoder layers
        """
        # Sequentially apply each encoder layer to the input
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

def subsequent_mask(size):
    """
    Create a mask to hide future positions in a sequence.
    Used in decoder self-attention to prevent positions from attending to subsequent positions (look-ahead masking).
    """
    attn_shape = (1, size, size)  # (batch_size=1, seq_len, seq_len); broadcastable for any batch size
    # Upper-triangular matrix with ones above main diagonal; 0 elsewhere.
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # Return mask where allowed positions are True (1), masked positions are False (0)
    return subsequent_mask == 0

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder block.
    Contains:
      - Masked self-attention (only sees previous tokens)
      - Source attention (attends to encoder outputs)
      - Feed-forward network (position-wise)
    Each sublayer is wrapped in a residual connection with layer normalization and dropout.
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Args:
            size (int): Hidden dimension size.
            self_attn (module): Multi-head masked self-attention.
            src_attn (module): Multi-head source attention.
            feed_forward (module): Position-wise feed-forward network.
            dropout (float): Dropout probability for sublayers.
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn    # Masked multi-head self-attention
        self.src_attn = src_attn      # Source-target multi-head attention (to encoder output/memory)
        self.feed_forward = feed_forward
        # Each sublayer is wrapped with normalization, residual, and dropout (typically)
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Pass the input through the three sublayers of the decoder.

        Args:
            x: Input tensor (target sequence embeddings so far).
            memory: Encoder outputs ("memory" the decoder can attend to).
            src_mask: Mask for encoder output (usually for padding).
            tgt_mask: Mask for current sequence (usually subsequent_mask or padding).
        Returns:
            Tensor after passing through all operations and residual connections.
        """
        m = memory
        # 1. Masked self-attention: attend to previous tokens (apply tgt_mask)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2. Source attention: attend to encoder output (apply src_mask)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 3. Feed-forward position-wise processing
        return self.sublayer[2](x, self.feed_forward)
    
class Decoder(nn.Module):
    """
    Stacked Transformer decoder.
    Consists of N DecoderLayer blocks, followed by a final layer normalization.
    """

    def __init__(self, layer, N):
        """
        Args:
            layer: A DecoderLayer instance or class to clone.
            N: Number of layers to stack.
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)           # Stack N identical layers
        self.norm = LayerNorm(layer.size)        # Layer normalization at the end

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Pass the input through each decoder layer in sequence.
        Returns normalized output after final block.

        Args:
            x: Input tensor (target sequence embeddings so far).
            memory: Encoder outputs ("memory" the decoder can attend to).
            src_mask: Mask for encoder output (usually for padding).
            tgt_mask: Mask for current sequence (usually subsequent_mask or padding).
        Returns:
            Final output after all decoder layers and normalization.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
"""
Attention.
"""
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'.

    This function forms the core of the Transformer's attention mechanism.
    For each 'query' element, attention scores are computed against all 'keys'.
    These scores are turned into probabilities (softmax), then used to mix the corresponding 'values'.

    Args:
        query: Tensor of shape (..., seq_len_q, d_k)
        key: Tensor of shape (..., seq_len_k, d_k)
        value: Tensor of shape (..., seq_len_v, d_v); usually seq_len_k == seq_len_v
        mask: Optional mask tensor (broadcastable to attention shape), positions with 0 are masked out.
        dropout: Optional dropout module applied to the attention weights.

    Returns:
        output: Weighted sum over 'value', shape (..., seq_len_q, d_v)
        p_attn: Attention weights (useful for visualization/debugging)
    """
    d_k = query.size(-1)  # Key/query dimension for scaling to stabilize gradients
    # Calculate raw attention scores by dot product between query and key
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # Fill masked positions with large negative value so their softmax is near zero
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Turn scores into probabilities (sum to 1 over keys dimension)
    p_attn = scores.softmax(dim=-1)

    # Optional dropout over attention weights (regularization)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # Return the weighted sum over values (output) and the attention weights
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Multi-Headed Attention module (as per Transformer architecture).
    It enables the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        Args:
            h: number of attention heads
            d_model: model dimension (split across heads)
            dropout: dropout probability (applied to attention weights)
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # Ensure dimension can be evenly split among heads
        
        self.d_k = d_model // h  # Dimensionality per head
        self.h = h               # Number of heads
        # We need four linear layers:
        # - For projecting the input into Q, K, V vectors for each head (3 layers)
        # - For final mapping the concatenated output back to d_model (1 layer)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # Placeholder to store attention weights (for visualization/debug)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements multi-headed attention as described in "Attention is All You Need".
        
        Args:
            query, key, value: Input tensors of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor (broadcastable for attention computation)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if mask is not None:
            # Expand mask so it is applied to all heads; insert a dimension for heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Linear projection: Input (batch, seq_len, d_model)
        # Project input tensor to Q, K, V for each head, then reshape to (batch, h, seq_len, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Scaled dot-product attention: independently on each head
        # Output x: (batch, h, seq_len, d_k), attn: (batch, h, seq_len, seq_len)
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) Concatenate all heads' outputs, then apply final linear layer
        # x: (batch, h, seq_len, d_k) --> (batch, seq_len, h * d_k = d_model)
        x = (
            x.transpose(1, 2)
             .contiguous()
             .view(nbatches, -1, self.h * self.d_k)
        )

        # Clean up memory (optional, for large batches)
        del query
        del key
        del value

        # Final linear projection to return to model dimension
        return self.linears[-1](x)
    
"""
Embedding.
"""
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Neural Network (FFN) as used in Transformer blocks.
    Applies two linear layers with a ReLU in between, independently to each position.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Dimensionality of input and output (model size)
            d_ff: Dimensionality of intermediate hidden layer
            dropout: Dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)     # First linear transformation: d_model → d_ff
        self.w_2 = nn.Linear(d_ff, d_model)     # Second linear transformation: d_ff → d_model
        self.dropout = nn.Dropout(dropout)      # Dropout for regularization

    def forward(self, x):
        # Apply first linear transformation, then ReLU, then dropout, finally project back
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    """
    Turns token indices into continuous word embeddings and scales by sqrt(model size).
    """
    def __init__(self, d_model, vocab):
        """
        Args:
            d_model: Embedding dimensionality (model size)
            vocab: Vocabulary size (number of unique tokens)
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)  # Lookup table for embeddings
        self.d_model = d_model                   # Save model dimension for scaling

    def forward(self, x):
        # Convert input indices to embeddings, then scale by sqrt(d_model) (helps with variance)
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding, 
    adding position information to input embeddings for the Transformer.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        """
        Args:
            d_model: Embedding dimensionality (model size)
            dropout: Dropout probability
            max_len: Maximum possible sequence length for which to precompute encodings
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a (max_len, d_model) matrix for positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)
        # Compute the div_term (different frequencies for each dimension)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # Apply sine to even dimensions, cosine to odd (per original Transformer paper)
        pe[:, 0::2] = torch.sin(position * div_term)      # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)      # Odd indices
        pe = pe.unsqueeze(0)                              # Shape: (1, max_len, d_model) for easy addition
        # Register 'pe' as a buffer so it's saved in state_dict but NOT a model parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Add positional encoding to input, with the same sequence length
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        # Apply dropout and return
        return self.dropout(x)

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    """
    Constructs a full Transformer model as described in the "Attention is All You Need" paper,
    assembling all the necessary encoder, decoder, embedding, positional encoding, and generator modules.

    Args:
        src_vocab: size of the source vocabulary (input language)
        tgt_vocab: size of the target vocabulary (output language)
        N: number of layers for both encoder and decoder stacks (default 6)
        d_model: dimension of the embeddings and all hidden states (default 512)
        d_ff: inner-layer dimension of position-wise feed-forward networks (default 2048)
        h: number of attention heads (default 8)
        dropout: dropout probability for all dropout layers (default 0.1)

    Returns:
        A fully constructed Transformer model ready for training.
    """
    # c is an alias for copy.deepcopy, needed to ensure independent modules for each use (especially attention layers)
    c = copy.deepcopy

    # Build one instance each of multi-head attention, position-wise feed-forward, and positional encoding
    attn = MultiHeadedAttention(h, d_model)                     # Multi-head self-attention module
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)        # Feed-forward network module
    position = PositionalEncoding(d_model, dropout)             # Sinusoidal positional encoding module

    # Build the full Encoder-Decoder model:
    model = EncoderDecoder(
        # Encoder: Stack of N encoder layers, each with attention and feed-forward modules
        Encoder(
            EncoderLayer(d_model, c(attn), c(ff), dropout), 
            N),
        # Decoder: Stack of N decoder layers, each with self-attn, src-attn, and feed-forward modules
        Decoder(
            DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), 
            N),
        # Source embeddings followed by positional encoding (combined in a Sequential block)
        nn.Sequential(
            Embeddings(d_model, src_vocab), 
            c(position)),
        # Target embeddings followed by positional encoding (combined in a Sequential block)
        nn.Sequential(
            Embeddings(d_model, tgt_vocab), 
            c(position)),
        # Generator: final linear projection to predict target vocabulary distribution
        Generator(d_model, tgt_vocab),
    )

    # Proper initialization of parameters is essential—use Xavier (Glorot uniform) initialization.
    # This is done for all parameters with more than 1 dimension (i.e., weight matrices, not bias vectors)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model