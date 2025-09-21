"""
Advanced multimodal fusion model with self-attention and hierarchical fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for self-attention"""
    
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

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
    
    def forward(self, x):
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
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
    
    def forward(self, query, key, value):
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights

class GatedFusion(nn.Module):
    """Gated fusion mechanism to dynamically weight modality contributions"""
    
    def __init__(self, input_dims, output_dim, dropout=0.1):
        super().__init__()
        total_dim = sum(input_dims)
        
        # Gate network to compute attention weights for each modality
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, len(input_dims)),
            nn.Softmax(dim=-1)
        )
        
        # Projection networks for each modality
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of tensors [text_feat, visual_feat, audio_feat]
        """
        # Concatenate all features for gate computation
        concat_features = torch.cat(modality_features, dim=-1)
        
        # Compute gating weights
        gate_weights = self.gate_network(concat_features)  # [B, num_modalities]
        
        # Project each modality to common dimension and apply gates
        projected_features = []
        for i, (feat, proj) in enumerate(zip(modality_features, self.projections)):
            projected = proj(feat)  # [B, output_dim]
            weighted = projected * gate_weights[:, i:i+1]  # Broadcasting
            projected_features.append(weighted)
        
        # Sum weighted projections
        fused = sum(projected_features)
        return self.output_norm(fused), gate_weights

class HierarchicalFusion(nn.Module):
    """Hierarchical fusion: pairwise fusion followed by final integration"""
    
    def __init__(self, hidden_dim, fusion_dim, dropout=0.1):
        super().__init__()
        
        # Pairwise fusion networks
        self.text_visual_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim)
        )
        
        self.text_audio_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim)
        )
        
        self.visual_audio_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim)
        )
        
        # Final fusion of pairwise interactions
        self.final_fusion = GatedFusion(
            [hidden_dim, hidden_dim, hidden_dim], 
            hidden_dim, 
            dropout
        )
    
    def forward(self, text_feat, visual_feat, audio_feat):
        # Pairwise fusion
        tv_fused = self.text_visual_fusion(torch.cat([text_feat, visual_feat], dim=-1))
        ta_fused = self.text_audio_fusion(torch.cat([text_feat, audio_feat], dim=-1))
        va_fused = self.visual_audio_fusion(torch.cat([visual_feat, audio_feat], dim=-1))
        
        # Final gated fusion of pairwise results
        final_fused, gate_weights = self.final_fusion([tv_fused, ta_fused, va_fused])
        
        return final_fused, gate_weights

class AttentionFusionModel(nn.Module):
    """Advanced multimodal fusion model with self-attention and hierarchical fusion"""
    
    def __init__(self, config):
        super().__init__()
        
        # Feature dimensions
        text_dim = config['data']['dims']['text']
        visual_dim = config['data']['dims']['visual']
        audio_dim = config['data']['dims']['audio']
        hidden_dim = config['model']['hidden_dim']
        
        # Input normalization
        self.text_norm = nn.LayerNorm(text_dim)
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.audio_norm = nn.LayerNorm(audio_dim)
        
        # Unimodal encoders
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Intra-modal self-attention layers
        self.text_self_attn = IntraModalSelfAttention(
            hidden_dim, 
            config['model']['num_heads'], 
            config['model']['dropout']
        )
        self.visual_self_attn = IntraModalSelfAttention(
            hidden_dim, 
            config['model']['num_heads'], 
            config['model']['dropout']
        )
        self.audio_self_attn = IntraModalSelfAttention(
            hidden_dim, 
            config['model']['num_heads'], 
            config['model']['dropout']
        )
        
        # Cross-modal attention layers
        self.text_visual_attn = CrossModalAttention(
            hidden_dim, 
            config['model']['num_heads'], 
            config['model']['dropout']
        )
        self.visual_text_attn = CrossModalAttention(
            hidden_dim, 
            config['model']['num_heads'], 
            config['model']['dropout']
        )
        
        self.text_audio_attn = CrossModalAttention(
            hidden_dim, 
            config['model']['num_heads'], 
            config['model']['dropout']
        )
        self.audio_text_attn = CrossModalAttention(
            hidden_dim, 
            config['model']['num_heads'], 
            config['model']['dropout']
        )
        
        self.visual_audio_attn = CrossModalAttention(
            hidden_dim, 
            config['model']['num_heads'], 
            config['model']['dropout']
        )
        self.audio_visual_attn = CrossModalAttention(
            hidden_dim, 
            config['model']['num_heads'], 
            config['model']['dropout']
        )
        
        # Hierarchical fusion
        self.hierarchical_fusion = HierarchicalFusion(
            hidden_dim, 
            config['model']['fusion_dim'], 
            config['model']['dropout']
        )
        
        # Temporal attention pooling
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, config['model']['fusion_dim']),
            nn.LayerNorm(config['model']['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(config['model']['fusion_dim'], config['model']['num_classes'])
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for numerical stability"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def attention_pooling(self, x):
        """Attention-based temporal pooling"""
        # Compute attention weights for each timestep
        attn_weights = self.temporal_attn(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # [B, T, 1]
        
        # Weighted sum over temporal dimension
        pooled = torch.sum(x * attn_weights, dim=1)  # [B, H]
        return pooled
    
    def forward(self, text, visual, audio):
        # Input normalization
        text = self.text_norm(text)
        visual = self.visual_norm(visual)
        audio = self.audio_norm(audio)
        
        # Unimodal encoding
        text_feat = self.text_encoder(text)
        visual_feat = self.visual_encoder(visual)
        audio_feat = self.audio_encoder(audio)
        
        # Intra-modal self-attention (capture temporal dependencies within each modality)
        text_self = self.text_self_attn(text_feat)
        visual_self = self.visual_self_attn(visual_feat)
        audio_self = self.audio_self_attn(audio_feat)
        
        # Cross-modal attention (bidirectional)
        text_from_visual, _ = self.text_visual_attn(text_self, visual_self, visual_self)
        visual_from_text, _ = self.visual_text_attn(visual_self, text_self, text_self)
        
        text_from_audio, _ = self.text_audio_attn(text_self, audio_self, audio_self)
        audio_from_text, _ = self.audio_text_attn(audio_self, text_self, text_self)
        
        visual_from_audio, _ = self.visual_audio_attn(visual_self, audio_self, audio_self)
        audio_from_visual, _ = self.audio_visual_attn(audio_self, visual_self, visual_self)
        
        # Enhanced representations with cross-modal information
        text_enhanced = text_self + 0.3 * (text_from_visual + text_from_audio)
        visual_enhanced = visual_self + 0.3 * (visual_from_text + visual_from_audio)
        audio_enhanced = audio_self + 0.3 * (audio_from_text + audio_from_visual)
        
        # Attention-based temporal pooling
        text_pooled = self.attention_pooling(text_enhanced)
        visual_pooled = self.attention_pooling(visual_enhanced)
        audio_pooled = self.attention_pooling(audio_enhanced)
        
        # Hierarchical fusion
        fused_repr, gate_weights = self.hierarchical_fusion(
            text_pooled, visual_pooled, audio_pooled
        )
        
        # Final classification
        output = self.classifier(fused_repr)
        
        return output