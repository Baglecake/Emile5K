"""
Transformer Modules for Émile-2 Simulation
------------------------------------------
Implements neural network components for processing quantum-influenced data,
including positional encoding, attention mechanisms, and transformer architectures.
"""
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Tuple, Dict, Union, List, Any
import traceback
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emile4.transformer_modules")

# Import from other modules
from utilities import MOMENTUM_DECAY, MINIMUM_COHERENCE_FLOOR
from data_classes import TransformerOutput

# ---------------------------
# Positional Encoding Modules
# ---------------------------
class PositionalEncoding(nn.Module):
    """
    Standard positional encoding with quantum-aware modulation.

    Adds positional information to input embeddings and optionally
    modulates them based on phase and coherence.
    """
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create standard positional encoding
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position*div_term)
        pe[:, 0, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)

        # Quantum modulation parameters
        self.phase_modulation = nn.Parameter(torch.zeros(1, 1, d_model))
        self.coherence_scale = nn.Parameter(torch.ones(1, 1, d_model))

        logger.debug(f"Initialized PositionalEncoding with d_model={d_model}")

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor]=None,
                coherence: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Apply positional encoding with optional quantum modulation.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            phase: Optional phase tensor for quantum modulation
            coherence: Optional coherence tensor for quantum modulation

        Returns:
            Positionally encoded tensor with same shape as input
        """
        try:
            # Validate input
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input (batch, seq, features), got {x.dim()}D")

            # Get sequence length
            batch, seq_len, d_model = x.size()

            # Apply standard positional encoding
            x = x + self.pe[:seq_len]

            # Apply quantum modulation if phase is provided
            if phase is not None:
                # Process phase tensor
                if isinstance(phase, torch.Tensor):
                    # Convert to appropriate dimensions if needed
                    if phase.dim() == 1:
                        phase = phase.unsqueeze(-1).unsqueeze(0)  # [1, batch, 1]
                    elif phase.dim() == 2:
                        phase = phase.unsqueeze(0)  # [1, batch, seq]

                    # Create phase modulation factor
                    phase_factor = torch.sin(phase)

                    # Create coherence scaling if provided
                    if coherence is None:
                        coherence = torch.ones_like(phase) * MINIMUM_COHERENCE_FLOOR
                    else:
                        # Ensure coherence has same dimensions as phase
                        if coherence.dim() != phase.dim():
                            coherence = coherence.view_as(phase)

                    # Apply coherence scaling
                    coherence_scaling = torch.sigmoid(self.coherence_scale) * coherence

                    # Apply quantum modulation
                    quantum_modulation = self.phase_modulation * phase_factor
                    x = x * (1.0 + quantum_modulation * coherence_scaling)
                else:
                    logger.warning("Phase provided is not a tensor, skipping quantum modulation")

            # Apply dropout
            x = self.dropout(x)

            return x

        except Exception as e:
            logger.error(f"Error in PositionalEncoding forward: {e}")
            # Return original input on error as fallback
            return x

class EnhancedPositionalEncoding(nn.Module):
    """
    Enhanced positional encoding with identity embedding and stability-modulated dropout.

    Adds positional information to input embeddings, identity embeddings,
    and applies quantum-aware modulation with stability tracking.
    """
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create standard positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position*div_term)
        pe[:, 0, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)

        # Enhanced parameters
        self.phase_modulation = nn.Parameter(torch.zeros(1, 1, d_model))
        self.coherence_scale = nn.Parameter(torch.ones(1, 1, d_model))
        self.identity_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        # Stability tracking
        self.stability_factor = 1.0
        self.phase_momentum = 0.0
        self.coherence_history = deque(maxlen=100)

        logger.debug(f"Initialized EnhancedPositionalEncoding with d_model={d_model}")

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor]=None,
                coherence: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Apply enhanced positional encoding with stability tracking.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            phase: Optional phase tensor for quantum modulation
            coherence: Optional coherence tensor for quantum modulation

        Returns:
            Enhanced encoded tensor with same shape as input
        """
        try:
            # Validate input
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input (batch, seq, features), got {x.dim()}D")

            # Get dimensions
            batch_size, seq_len, d_model = x.shape

            # Apply standard positional encoding
            encoded = x + self.pe[:seq_len]

            # Apply quantum modulation if phase is provided
            if phase is not None:
                # Process phase tensor
                if isinstance(phase, torch.Tensor):
                    # Convert to appropriate dimensions if needed
                    if phase.dim() == 1:
                        phase = phase.unsqueeze(-1).unsqueeze(0)  # [1, batch, 1]
                    elif phase.dim() == 2:
                        phase = phase.unsqueeze(0)  # [1, batch, seq]

                    # Create phase modulation factor
                    phase_factor = torch.sin(phase)

                    # Update phase momentum for stability tracking
                    current_phase = phase_factor.mean().item()
                    self.phase_momentum = MOMENTUM_DECAY * self.phase_momentum + (1 - MOMENTUM_DECAY) * current_phase

                    # Create quantum modulation with momentum influence
                    quantum_modulation = self.phase_modulation * phase_factor * (1.0 + 0.1 * self.phase_momentum)

                    # Add coherence scaling
                    coherence_scaling = torch.sigmoid(self.coherence_scale)
                    if coherence is None:
                        coherence = torch.ones_like(phase) * MINIMUM_COHERENCE_FLOOR
                    else:
                        # Ensure coherence has same dimensions as phase
                        if coherence.dim() != phase.dim():
                            coherence = coherence.view_as(phase)

                    # Apply coherence to scaling
                    coherence_scaling = coherence_scaling * coherence

                    # Apply quantum modulation
                    encoded = encoded * (1.0 + quantum_modulation * coherence_scaling)

                    # Update stability factor
                    self.stability_factor = 0.95*self.stability_factor + 0.05*(1.0 - abs(self.phase_momentum))
                else:
                    logger.warning("Phase provided is not a tensor, skipping quantum modulation")

            # Apply identity embedding (skip connection)
            identity_factor = torch.sigmoid(self.identity_embedding)
            encoded = encoded + identity_factor * x

            # Apply stability-modulated dropout
            effective_dropout = self.dropout.p * (2.0 - self.stability_factor)
            effective_dropout = max(0.0, min(0.5, effective_dropout))  # Clamp to reasonable range
            encoded = F.dropout(encoded, p=effective_dropout, training=self.training)

            return encoded

        except Exception as e:
            logger.error(f"Error in EnhancedPositionalEncoding forward: {e}")
            # Return original input on error as fallback
            return x

    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics for monitoring."""
        return {
            'stability_factor': float(self.stability_factor),
            'phase_momentum': float(self.phase_momentum),
            'effective_dropout': float(self.dropout.p * (2.0 - self.stability_factor))
        }

# ---------------------------
# Enhanced Multi-Head Attention Module
# ---------------------------
class EnhancedMultiheadAttention(nn.Module):
    """
    Multi-head attention with quantum-aware modulation and stability adjustments.

    Implements attention mechanism with added quantum influence, shape handling,
    and stability tracking.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Quantum modulation parameters
        self.phase_coupling = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.coherence_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.attention_scale = nn.Parameter(torch.ones(1))
        self.stability_factor = nn.Parameter(torch.ones(1))

        # Dimension adaptation for handling different input sizes
        self.dim_adapt_in = nn.Linear(d_model, d_model)
        self.dim_adapt_out = nn.Linear(d_model, d_model)

        # Dropout and scaling
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Attention pattern tracking
        self.attention_patterns = deque(maxlen=100)
        self.stability_threshold = 0.1
        self.dimension_mismatch_counter = 0
        self.max_dimension_retries = 3

        logger.debug(f"Initialized EnhancedMultiheadAttention with d_model={d_model}, num_heads={num_heads}")

    def _validate_input_shapes(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate and reshape input tensors to correct dimensions.

        Args:
            q, k, v: Input tensors for query, key, value

        Returns:
            Reshaped tensors with correct dimensions
        """
        try:
            # Define shape fixing function
            def fix_shape(x: torch.Tensor) -> torch.Tensor:
                # Handle extra dimensions by squeezing
                while x.dim() > 3:
                    x = x.squeeze(1)

                # Handle fewer dimensions by unsqueezing
                if x.dim() == 1:
                    x = x.unsqueeze(0).unsqueeze(0)
                elif x.dim() == 2:
                    x = x.unsqueeze(1)

                return x

            # Fix shapes for all inputs
            q = fix_shape(q)
            k = fix_shape(k)
            v = fix_shape(v)

            # Handle dimension mismatches with dimension adaptation layers
            if q.size(-1) != self.d_model:
                logger.debug(f"Adapting query from {q.size(-1)} to {self.d_model} dimensions")
                q = self.dim_adapt_in(q)

            if k.size(-1) != self.d_model:
                logger.debug(f"Adapting key from {k.size(-1)} to {self.d_model} dimensions")
                k = self.dim_adapt_in(k)

            if v.size(-1) != self.d_model:
                logger.debug(f"Adapting value from {v.size(-1)} to {self.d_model} dimensions")
                v = self.dim_adapt_in(v)

            return q, k, v

        except Exception as e:
            logger.error(f"Error in shape validation: {e}")
            # Return original inputs on error
            return q, k, v

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape tensor for multi-head attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Reshaped tensor [batch, num_heads, seq_len, head_dim]
        """
        try:
            # Extract dimensions
            batch_size = x.size(0)
            seq_len = x.size(1)

            # Adapt dimensions if needed
            if x.size(-1) != self.d_model:
                x = self.dim_adapt_in(x)

            # Reshape: [batch, seq, d_model] -> [batch, seq, num_heads, head_dim]
            x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

            # Transpose: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
            x = x.transpose(1, 2)

            return x

        except Exception as e:
            logger.error(f"Error in tensor reshaping: {e}")
            # Return safe default on error
            batch_size = x.size(0) if x.dim() > 0 else 1
            return torch.zeros((batch_size, self.num_heads, 1, self.head_dim), device=x.device)

    def _track_attention_pattern(self, attn_weights: torch.Tensor) -> None:
        """
        Track attention patterns for stability monitoring.

        Args:
            attn_weights: Attention weight tensor
        """
        try:
            with torch.no_grad():
                # Extract mean attention pattern across batch
                pattern = attn_weights.mean(dim=0).detach().cpu()

                # Store in history
                self.attention_patterns.append(pattern)

                # Check for pattern stability if we have enough history
                if len(self.attention_patterns) > 1:
                    current = self.attention_patterns[-1]
                    previous = self.attention_patterns[-2]

                    # Calculate pattern difference
                    pattern_diff = torch.norm(current - previous)

                    # Update stability factor based on pattern difference
                    if pattern_diff > self.stability_threshold:
                        # Decrease stability on high differences
                        self.stability_factor.data *= 0.95
                    else:
                        # Increase stability on low differences (with max cap)
                        self.stability_factor.data = torch.min(
                            self.stability_factor.data * 1.05,
                            torch.tensor(1.0, device=self.stability_factor.device)
                        )

        except Exception as e:
            logger.error(f"Error tracking attention pattern: {e}")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           phase: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for enhanced multi-head attention.

        Args:
            q, k, v: Query, key, value tensors
            phase: Optional phase tensor for quantum modulation

        Returns:
            Tuple of (output tensor, attention weights)
        """
        try:
            # Validate and fix input shapes
            q, k, v = self._validate_input_shapes(q, k, v)

            # Get dimensions
            batch_size = q.size(0)
            seq_len = q.size(1)

            # Apply projection layers first (before reshaping)
            q = self.q_proj(q)  # [batch, seq, d_model]
            k = self.k_proj(k)  # [batch, seq, d_model]
            v = self.v_proj(v)  # [batch, seq, d_model]

            # Reshape for multi-head attention
            q = self._shape(q)  # [batch, num_heads, seq, head_dim]
            k = self._shape(k)  # [batch, num_heads, seq, head_dim]
            v = self._shape(v)  # [batch, num_heads, seq, head_dim]

            # Calculate attention scores: [batch, num_heads, seq, seq]
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale * self.attention_scale

            # Apply phase modulation if provided
            if phase is not None:
                # Ensure phase is a tensor on the correct device
                if isinstance(phase, torch.Tensor):
                    phase = phase.to(q.device)

                    # Reshape phase to match attention dimensions
                    if phase.dim() == 1:
                        phase = phase.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, 1]

                    # Create phase factor
                    phase_factor = torch.sin(phase)

                    # Apply phase coupling to attention scores
                    scores = scores * (1.0 + self.phase_coupling * phase_factor)
                else:
                    logger.warning("Phase is not a tensor, skipping modulation")

            # Apply softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Track attention patterns for stability monitoring
            self._track_attention_pattern(attn_weights)

            # Apply attention weights to values
            output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq, head_dim]

            # Reshape output: [batch, num_heads, seq, head_dim] -> [batch, seq, d_model]
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch_size, seq_len, self.d_model)

            # Apply output projection with stability factor
            output = self.o_proj(output) * self.stability_factor

            return output, attn_weights

        except Exception as e:
            logger.error(f"Error in EnhancedMultiheadAttention forward: {e}")

            # Create safe default outputs on error
            device = q.device if hasattr(q, 'device') else torch.device('cpu')
            batch_size = q.size(0) if hasattr(q, 'size') else 1
            seq_len = q.size(1) if q.dim() > 1 else 1

            default_output = torch.zeros((batch_size, seq_len, self.d_model), device=device)
            default_weights = torch.zeros((batch_size, self.num_heads, seq_len, seq_len), device=device)

            return default_output, default_weights

    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics for monitoring."""
        return {
            'stability_factor': float(self.stability_factor.item()),
            'attention_scale': float(self.attention_scale.item()),
            'dimension_mismatches': self.dimension_mismatch_counter
        }

# ---------------------------
# Four-Dimensional Transformer Adapter
# ---------------------------
class FourDimTransformerAdapter(nn.Module):
    """
    Adapter for handling 4D inputs in a transformer.

    If the input has shape [batch, extra, seq, embed], this adapter supports two strategies:
    - "merge": Merge the extra dimension with the sequence dimension.
    - "separate": Process the extra dimension via a convolutional branch and then fuse.
    """
    def __init__(self, base_transformer: nn.Module, merge_strategy: str = "merge"):
        super().__init__()
        self.base_transformer = base_transformer
        self.merge_strategy = merge_strategy

        # Validate strategy
        if self.merge_strategy not in ["merge", "separate"]:
            raise ValueError("merge_strategy must be either 'merge' or 'separate'")

        # Set up separate processing branch if needed
        if self.merge_strategy == "separate":
            # Find input dimension from base transformer
            embed_dim = self._get_embed_dim_from_transformer(base_transformer)

            # Convolutional branch to process the extra dimension
            self.extra_conv = nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=(3,1), padding=(1,0)
            )

            # Fusion layer to combine processed features
            self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)

        logger.info(f"Initialized FourDimTransformerAdapter with {merge_strategy} strategy")

    def _get_embed_dim_from_transformer(self, transformer: nn.Module) -> int:
        """
        Extract the embedding dimension from the base transformer.
        """
        embed_dim = None

        # Try various attribute paths that might contain the dimension
        if hasattr(transformer, "input_layer") and hasattr(transformer.input_layer, "in_features"):
            embed_dim = transformer.input_layer.in_features
        elif hasattr(transformer, "d_model"):
            embed_dim = transformer.d_model
        elif hasattr(transformer, "embedding_dim"):
            embed_dim = transformer.embedding_dim
        elif hasattr(transformer, "emb_dim"):
            embed_dim = transformer.emb_dim

        # If still not found, check first layer for Linear or Embedding
        if embed_dim is None:
            for module in transformer.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    if hasattr(module, "in_features"):
                        embed_dim = module.in_features
                        break
                    elif hasattr(module, "embedding_dim"):
                        embed_dim = module.embedding_dim
                        break

        # If still not found, raise error
        if embed_dim is None:
            raise ValueError(
                "Could not determine embedding dimension from base transformer. "
                "Please specify embed_dim manually or use a transformer with "
                "detectable embedding dimensions."
            )

        return embed_dim

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor] = None) -> TransformerOutput:
        """
        Forward pass of the adapter.

        Handles emerging 4D inputs and adapts them according to the strategy.

        Args:
            x: Input tensor, potentially 4D [batch, extra, seq, embed]
            phase: Optional phase tensor for quantum modulation

        Returns:
            TransformerOutput from base transformer
        """
        try:
            # Ensure x is a tensor
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"Expected input to be a torch.Tensor, but got {type(x)} instead."
                )

            # Log input shape for debugging
            original_shape = x.shape
            dim_message = f"Input shape: {original_shape}"

            # Handle 4D input: [batch, extra, seq, embed]
            if x.dim() == 4:
                batch, extra, seq, embed = x.size()
                dim_message += f" → 4D input detected"

                if self.merge_strategy == "merge":
                    # Merge extra with sequence dimension: [batch, extra*seq, embed]
                    x = x.reshape(batch, extra * seq, embed)
                    dim_message += f" → Merged to: {x.shape}"

                elif self.merge_strategy == "separate":
                    # Process extra dimension separately with convolutional branch
                    # First reshape: [batch*extra, 1, seq, embed]
                    x_reshaped = x.reshape(batch * extra, 1, seq, embed)

                    # Apply convolution
                    x_conv = self.extra_conv(x_reshaped)  # [batch*extra, 1, seq, embed]
                    x_conv = F.relu(x_conv)

                    # Restore original dimensions: [batch, extra, seq, embed]
                    x_conv = x_conv.reshape(batch, extra, seq, embed)

                    # Extract primary slice: [batch, seq, embed]
                    x_primary = x[:, 0, :, :]

                    # Average over extra dimension: [batch, seq, embed]
                    x_extra = x_conv.mean(dim=1)

                    # Concatenate along embedding: [batch, seq, 2*embed]
                    x_cat = torch.cat([x_primary, x_extra], dim=-1)

                    # Fuse with linear layer: [batch, seq, embed]
                    x = self.fusion_layer(x_cat)
                    dim_message += f" → Processed separately to: {x.shape}"

            # Log dimension info
            logger.debug(dim_message)

            # Pass to base transformer
            return self.base_transformer(x, phase)

        except Exception as e:
            logger.error(f"Error in FourDimTransformerAdapter forward: {e}")
            traceback.print_exc()
            # Create an emergency default output
            batch_size = x.size(0) if hasattr(x, 'size') else 1
            device = x.device if hasattr(x, 'device') else torch.device('cpu')

            # Return minimal valid TransformerOutput
            return TransformerOutput(
                prediction=torch.zeros((batch_size, 1), device=device),
                phase_prediction=torch.zeros((batch_size, 1), device=device),
                value_estimate=torch.zeros((batch_size, 1), device=device),
                attention_weights={},
                entropy=torch.tensor(0.0, device=device),
                coherence_estimate=torch.tensor(MINIMUM_COHERENCE_FLOOR, device=device)
            )

# ---------------------------
# Transformer Layer and Recursive Transformer
# ---------------------------
class TransformerLayer(nn.Module):
    """
    A single transformer layer with multi-head attention and feed-forward network.

    Implements a standard transformer layer with residual connections,
    layer normalization, and optional quantum modulation.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attention = EnhancedMultiheadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        logger.debug(f"Initialized TransformerLayer with d_model={d_model}, nhead={nhead}")

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for transformer layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            phase: Optional phase tensor for quantum modulation

        Returns:
            Tuple of (output tensor, attention weights)
        """
        try:
            # Apply attention with residual connection and normalization
            attn_out, attn_weights = self.attention(x, x, x, phase)
            x = self.norm1(x + self.dropout(attn_out))

            # Apply feed-forward with residual connection and normalization
            ff_out = self.ff(x)
            x = self.norm2(x + self.dropout(ff_out))

            return x, attn_weights

        except Exception as e:
            logger.error(f"Error in TransformerLayer forward: {e}")

            # Create safe default outputs
            batch_size, seq_len, d_model = x.size()
            device = x.device

            default_output = torch.zeros_like(x)
            default_weights = torch.zeros(
                batch_size, self.attention.num_heads, seq_len, seq_len, device=device
            )

            return default_output, default_weights

class RecursiveDistinctionTransformer(nn.Module):
    """
    A transformer for processing and predicting distinction levels.

    Implements a transformer architecture with quantum awareness,
    accepting 3D inputs and adapting to emergent dimensions.
    """
    def __init__(self,
                 input_size: int = 20,
                 d_model: int = 20,
                 nhead: int = 4,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        # Validate dimensions and parameters
        self.input_size = input_size
        self.d_model = d_model

        # Input projection layer
        self.input_layer = nn.Linear(input_size, d_model)

        # Positional encoding - use enhanced version
        self.positional_encoding = EnhancedPositionalEncoding(d_model, dropout=dropout)

        # Create transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])

        # Output projection layers
        self.output_layer = nn.Linear(d_model, output_size)
        self.phase_output = nn.Linear(d_model, 1)
        self.value_output = nn.Linear(d_model, 1)
        self.coherence_output = nn.Linear(d_model, 1)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"Initialized RecursiveDistinctionTransformer with input_size={input_size}, "
            f"d_model={d_model}, nhead={nhead}, num_layers={num_layers}"
        )

    def _init_weights(self):
        """Initialize transformer weights for stable training."""
        # Xavier initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Validate and adapt input tensor to correct shape.

        Args:
            x: Input tensor of arbitrary shape

        Returns:
            Properly shaped tensor [batch, seq, input_size]
        """
        device = x.device if hasattr(x, 'device') else torch.device('cpu')

        # Handle emergent 4D+ inputs
        if x.dim() > 3:
            batch_size = x.size(0)
            embed_dim = x.size(-1)
            new_seq_len = int(np.prod(x.shape[1:-1]))

            logger.debug(
                f"Emergent shape detected: {x.shape}. "
                f"Reshaping to [{batch_size}, {new_seq_len}, {embed_dim}]"
            )

            # Merge all dimensions between batch and embedding
            x = x.view(batch_size, new_seq_len, embed_dim)

        # Handle 1D tensor (single vector)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)

        # Handle 2D tensor (batch of vectors)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # Check final shape
        batch_size, seq_len, input_size = x.size()

        # Handle input size mismatch
        if input_size != self.input_size:
            logger.warning(
                f"Input size {input_size} does not match expected {self.input_size}. "
                f"Reshaping tensor."
            )

            if input_size > self.input_size:
                # Trim if larger
                x = x[:, :, :self.input_size]
            else:
                # Pad if smaller
                padding = torch.zeros(
                    batch_size, seq_len, self.input_size - input_size,
                    device=device
                )
                x = torch.cat([x, padding], dim=2)

        return x

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor]=None) -> TransformerOutput:
        """
        Forward pass of the recursive distinction transformer.

        Args:
            x: Input tensor, may be variably shaped
            phase: Optional phase tensor for quantum modulation

        Returns:
            TransformerOutput containing predictions and attention
        """
        try:
            # Set device
            device = x.device if hasattr(x, 'device') else torch.device('cpu')

            # Validate and reshape input
            x = self._validate_input(x)
            batch_size, seq_len, input_size = x.size()

            # Apply input projection
            x = self.input_layer(x)

            # Apply positional encoding with phase modulation
            x = self.positional_encoding(x, phase)

            # Track attention weights and layer outputs
            attention_weights = {}
            layer_outputs = []

            # Process through transformer layers
            for i, layer in enumerate(self.layers):
                try:
                    # Apply transformer layer
                    layer_output, layer_attn = layer(x, phase)

                    # Store results
                    x = layer_output
                    attention_weights[f'layer_{i}'] = layer_attn
                    layer_outputs.append(layer_output)

                except Exception as layer_error:
                    logger.error(f"Error in layer {i}: {layer_error}")
                    # Continue with current x if a layer fails
                    attention_weights[f'layer_{i}'] = torch.zeros(
                        batch_size, self.layers[0].attention.num_heads,
                        seq_len, seq_len, device=device
                    )

            try:
                # Generate outputs
                # Main prediction - applies sigmoid for 0-1 range
                prediction = torch.sigmoid(self.output_layer(x))

                # Phase prediction
                phase_pred = self.phase_output(x)

                # Value estimate
                value_est = self.value_output(x)

                # Calculate coherence estimates from each layer
                coherence_estimates = []
                for output in layer_outputs:
                    coherence_est = torch.sigmoid(self.coherence_output(output))
                    coherence_estimates.append(coherence_est)

                # Combine coherence estimates with weighted average
                if coherence_estimates:
                    # Calculate softmax weights based on mean coherence values
                    coherence_weights = torch.softmax(
                        torch.stack([est.mean() for est in coherence_estimates]),
                        dim=0
                    )

                    # Apply weights to coherence estimates
                    coherence_est = (
                        torch.stack(coherence_estimates) * coherence_weights.view(-1, 1, 1)
                    ).sum(0)
                else:
                    # Default coherence if no estimates available
                    coherence_est = torch.tensor(
                        MINIMUM_COHERENCE_FLOOR,
                        device=device
                    ).expand_as(prediction)

                # Calculate attention entropy
                entropies = []
                for weights in attention_weights.values():
                    # Add small epsilon to prevent log(0)
                    entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()
                    entropies.append(entropy)

                # Average entropy across layers
                attn_entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=device)

                # Construct the final output
                return TransformerOutput(
                    prediction=prediction.squeeze(-1),
                    phase_prediction=phase_pred.squeeze(-1),
                    value_estimate=value_est.squeeze(-1),
                    attention_weights=attention_weights,
                    entropy=attn_entropy,
                    coherence_estimate=coherence_est
                )

            except Exception as output_error:
                logger.error(f"Error generating transformer outputs: {output_error}")
                # Return default output on error
                return TransformerOutput(
                    prediction=torch.zeros((batch_size, seq_len), device=device),
                    phase_prediction=torch.zeros((batch_size, seq_len), device=device),
                    value_estimate=torch.zeros((batch_size, seq_len), device=device),
                    attention_weights={},
                    entropy=torch.tensor(0.0, device=device),
                    coherence_estimate=torch.tensor(MINIMUM_COHERENCE_FLOOR, device=device)
                )

        except Exception as e:
            logger.error(f"Error in RecursiveDistinctionTransformer forward: {e}")

            # Determine batch size for default output
            batch_size = x.size(0) if hasattr(x, 'size') else 1

            # Return default output with proper device
            device = x.device if hasattr(x, 'device') else torch.device('cpu')

            return TransformerOutput(
                prediction=torch.zeros((batch_size, 1), device=device),
                phase_prediction=torch.zeros((batch_size, 1), device=device),
                value_estimate=torch.zeros((batch_size, 1), device=device),
                attention_weights={},
                entropy=torch.tensor(0.0, device=device),
                coherence_estimate=torch.tensor(MINIMUM_COHERENCE_FLOOR, device=device)
            )

    def get_attention_patterns(self) -> Dict[str, torch.Tensor]:
        """Get attention patterns from each layer for visualization."""
        patterns = {}

        for i, layer in enumerate(self.layers):
            if hasattr(layer.attention, 'attention_patterns') and layer.attention.attention_patterns:
                # Get most recent pattern
                pattern = layer.attention.attention_patterns[-1]
                patterns[f'layer_{i}'] = pattern

        return patterns

