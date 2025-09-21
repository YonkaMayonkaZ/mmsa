"""
Advanced multimodal fusion model with self-attention and hierarchical fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for self-attention"""

    # CORRECTED: Increased max_len to a more reasonable default
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Changed shape for easier batch compatibility
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class IntraModalSelfAttention(nn.Module):
    """Self-attention within a single modality to capture temporal dependencies"""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    # CORRECTED: Added key_padding_mask argument
    def forward(self, x, key_padding_mask=None):
        # Add positional encoding
        x = self.pos_encoding(x)

        # Self-attention with residual connection and padding mask
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    # CORRECTED: Added key_padding_mask argument
    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_weights = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights

class TensorFusion(nn.Module):
    """Tensor Fusion for multimodal fusion using pairwise interactions"""
    def __init__(self, input_dims, output_dim, dropout=0.1):
        super(TensorFusion, self).__init__()
        
        hidden_dim = input_dims[0]
        self.fusion_tv = nn.Bilinear(hidden_dim, hidden_dim, output_dim)
        self.fusion_ta = nn.Bilinear(hidden_dim, hidden_dim, output_dim)
        self.fusion_va = nn.Bilinear(hidden_dim, hidden_dim, output_dim)
        self.final_fusion = nn.Linear(3 * output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, modality_features):
        text_p, visual_p, audio_p = modality_features
        fused_tv = self.fusion_tv(text_p, visual_p)
        fused_ta = self.fusion_ta(text_p, audio_p)
        fused_va = self.fusion_va(visual_p, audio_p)
        concat_fused = torch.cat([fused_tv, fused_ta, fused_va], dim=-1)
        final_fused = self.final_fusion(concat_fused)
        return self.norm(self.dropout(final_fused))


class AttentionFusionModel(nn.Module):
    """Advanced multimodal fusion model with self-attention and hierarchical fusion"""

    def __init__(self, config):
        super().__init__()
        text_dim = config['data']['dims']['text']
        visual_dim = config['data']['dims']['visual']
        audio_dim = config['data']['dims']['audio']
        hidden_dim = config['model']['hidden_dim']

        self.text_norm = nn.LayerNorm(text_dim)
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.audio_norm = nn.LayerNorm(audio_dim)

        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.visual_encoder = nn.Linear(visual_dim, hidden_dim)
        self.audio_encoder = nn.Linear(audio_dim, hidden_dim)

        self.text_self_attn = IntraModalSelfAttention(hidden_dim, config['model']['num_heads'], config['model']['dropout'])
        self.visual_self_attn = IntraModalSelfAttention(hidden_dim, config['model']['num_heads'], config['model']['dropout'])
        self.audio_self_attn = IntraModalSelfAttention(hidden_dim, config['model']['num_heads'], config['model']['dropout'])

        self.text_visual_attn = CrossModalAttention(hidden_dim, config['model']['num_heads'], config['model']['dropout'])
        self.visual_text_attn = CrossModalAttention(hidden_dim, config['model']['num_heads'], config['model']['dropout'])
        self.text_audio_attn = CrossModalAttention(hidden_dim, config['model']['num_heads'], config['model']['dropout'])
        self.audio_text_attn = CrossModalAttention(hidden_dim, config['model']['num_heads'], config['model']['dropout'])
        self.visual_audio_attn = CrossModalAttention(hidden_dim, config['model']['num_heads'], config['model']['dropout'])
        self.audio_visual_attn = CrossModalAttention(hidden_dim, config['model']['num_heads'], config['model']['dropout'])

        self.tensor_fusion = TensorFusion([hidden_dim, hidden_dim, hidden_dim], config['model']['fusion_dim'], config['model']['dropout'])

        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(config['model']['fusion_dim'], config['model']['fusion_dim']),
            nn.LayerNorm(config['model']['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(config['model']['fusion_dim'], config['model']['num_classes'])
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    # CORRECTED: attention pooling now accepts a mask
    def attention_pooling(self, x, mask=None):
        attn_weights = self.temporal_attn(x)
        if mask is not None:
            # Set attention weights for padding tokens to a very low value before softmax
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled

    # CORRECTED: Forward pass now accepts and uses attention masks
    def forward(self, text, visual, audio, attention_mask=None):
        text = self.text_norm(text)
        visual = self.visual_norm(visual)
        audio = self.audio_norm(audio)

        text_feat = self.text_encoder(text)
        visual_feat = self.visual_encoder(visual)
        audio_feat = self.audio_encoder(audio)
        
        # Pass the mask to the intra-modal attention layers
        text_self = self.text_self_attn(text_feat, key_padding_mask=attention_mask)
        visual_self = self.visual_self_attn(visual_feat, key_padding_mask=attention_mask)
        audio_self = self.audio_self_attn(audio_feat, key_padding_mask=attention_mask)

        # Pass the mask to the cross-modal attention layers
        text_from_visual, _ = self.text_visual_attn(text_self, visual_self, visual_self, key_padding_mask=attention_mask)
        visual_from_text, _ = self.visual_text_attn(visual_self, text_self, text_self, key_padding_mask=attention_mask)
        text_from_audio, _ = self.text_audio_attn(text_self, audio_self, audio_self, key_padding_mask=attention_mask)
        audio_from_text, _ = self.audio_text_attn(audio_self, text_self, text_self, key_padding_mask=attention_mask)
        visual_from_audio, _ = self.visual_audio_attn(visual_self, audio_self, audio_self, key_padding_mask=attention_mask)
        audio_from_visual, _ = self.audio_visual_attn(audio_self, visual_self, visual_self, key_padding_mask=attention_mask)
        
        text_enhanced = text_self + 0.3 * (text_from_visual + text_from_audio)
        visual_enhanced = visual_self + 0.3 * (visual_from_text + visual_from_audio)
        audio_enhanced = audio_self + 0.3 * (audio_from_text + audio_from_visual)

        # Pass mask to temporal pooling
        text_pooled = self.attention_pooling(text_enhanced, attention_mask)
        visual_pooled = self.attention_pooling(visual_enhanced, attention_mask)
        audio_pooled = self.attention_pooling(audio_enhanced, attention_mask)
        
        fused_repr = self.tensor_fusion([text_pooled, visual_pooled, audio_pooled])
        output = self.classifier(fused_repr)
        return output
