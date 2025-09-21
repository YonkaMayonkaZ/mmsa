"""
Attention-based multimodal fusion model - NUMERICALLY STABLE VERSION
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
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

class AttentionFusionModel(nn.Module):
    """Main multimodal fusion model with numerical stability improvements"""
    
    def __init__(self, config):
        super().__init__()
        
        # Feature dimensions
        text_dim = config['data']['dims']['text']
        visual_dim = config['data']['dims']['visual']
        audio_dim = config['data']['dims']['audio']
        hidden_dim = config['model']['hidden_dim']
        
        # Layer normalization for input features
        self.text_norm = nn.LayerNorm(text_dim)
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.audio_norm = nn.LayerNorm(audio_dim)
        
        # Encoders with proper initialization
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
        
        # Cross-modal attention
        self.text_visual_attn = CrossModalAttention(
            hidden_dim, 
            config['model']['num_heads'],
            config['model']['dropout']
        )
        
        self.text_audio_attn = CrossModalAttention(
            hidden_dim,
            config['model']['num_heads'],
            config['model']['dropout']
        )
        
        self.visual_audio_attn = CrossModalAttention(
            hidden_dim,
            config['model']['num_heads'],
            config['model']['dropout']
        )
        
        # Fusion with proper normalization
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, config['model']['fusion_dim']),
            nn.LayerNorm(config['model']['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(config['model']['fusion_dim'], config['model']['num_classes'])
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for numerical stability"""
        if isinstance(module, nn.Linear):
            # Xavier initialization
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, text, visual, audio):
        # Normalize inputs to prevent exploding values
        text = self.text_norm(text)
        visual = self.visual_norm(visual) 
        audio = self.audio_norm(audio)
        
        # Encode each modality
        text_feat = self.text_encoder(text)
        visual_feat = self.visual_encoder(visual)
        audio_feat = self.audio_encoder(audio)
        
        # Cross-modal attention with residual connections
        text_v, _ = self.text_visual_attn(text_feat, visual_feat, visual_feat)
        text_a, _ = self.text_audio_attn(text_feat, audio_feat, audio_feat)
        visual_a, _ = self.visual_audio_attn(visual_feat, audio_feat, audio_feat)
        
        # Combine with scaled residual connections
        text_enhanced = text_feat + 0.5 * (text_v + text_a)
        visual_enhanced = visual_feat + 0.5 * visual_a
        audio_enhanced = audio_feat
        
        # Global pooling with clipping to prevent overflow
        text_global = torch.mean(text_enhanced, dim=1).clamp(-10, 10)
        visual_global = torch.mean(visual_enhanced, dim=1).clamp(-10, 10)
        audio_global = torch.mean(audio_enhanced, dim=1).clamp(-10, 10)
        
        # Concatenate all features
        combined = torch.cat([text_global, visual_global, audio_global], dim=-1)
        
        # Final classification
        output = self.fusion(combined)
        
        return output