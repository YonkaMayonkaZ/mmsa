"""
Attention-based multimodal fusion model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
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
    """Main multimodal fusion model"""
    
    def __init__(self, config):
        super().__init__()
        
        # Feature dimensions
        text_dim = config['data']['dims']['text']
        visual_dim = config['data']['dims']['visual']
        audio_dim = config['data']['dims']['audio']
        hidden_dim = config['model']['hidden_dim']
        
        # Encoders
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(hidden_dim, hidden_dim)
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
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, config['model']['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(config['model']['fusion_dim'], config['model']['num_classes'])
        )
    
    def forward(self, text, visual, audio):
        # Encode
        text_feat = self.text_encoder(text)
        visual_feat = self.visual_encoder(visual)
        audio_feat = self.audio_encoder(audio)
        
        # Cross-modal attention
        text_v, _ = self.text_visual_attn(text_feat, visual_feat, visual_feat)
        text_a, _ = self.text_audio_attn(text_feat, audio_feat, audio_feat)
        visual_a, _ = self.visual_audio_attn(visual_feat, audio_feat, audio_feat)
        
        # Combine
        text_enhanced = text_feat + text_v + text_a
        visual_enhanced = visual_feat + visual_a
        
        # Global pooling
        text_global = torch.mean(text_enhanced, dim=1)
        visual_global = torch.mean(visual_enhanced, dim=1)
        audio_global = torch.mean(audio_feat, dim=1)
        
        # Concatenate
        combined = torch.cat([text_global, visual_global, audio_global], dim=-1)
        
        # Classify
        output = self.fusion(combined)
        
        return output