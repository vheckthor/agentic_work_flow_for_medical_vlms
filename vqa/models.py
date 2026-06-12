"""Four fine-tuned multimodal classifier variants for SLAKE closed-set VQA.
Implemented by me
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel,
    CLIPModel,
    ViTModel,
)


class ViTBERTClassifier(nn.Module):
    """Variant 1: ViT image encoder + BERT question encoder, concat -> linear."""

    def __init__(
        self,
        num_classes: int,
        vit_name: str = "google/vit-base-patch16-224",
        bert_name: str = "bert-base-multilingual-cased",
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_name)
        self.bert = BertModel.from_pretrained(bert_name)
        d = self.vit.config.hidden_size + self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d // 2, num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        v = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        t = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.head(torch.cat([v, t], dim=-1))


class ResNetBERTClassifier(nn.Module):
    """Variant 2: ResNet-50 CNN + BERT (different vision backbone)."""

    def __init__(
        self,
        num_classes: int,
        bert_name: str = "bert-base-multilingual-cased",
    ):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.bert = BertModel.from_pretrained(bert_name)
        d_img = 2048
        d = d_img + self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d // 2, num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.cnn(pixel_values)
        x = torch.flatten(x, 1)
        t = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.head(torch.cat([x, t], dim=-1))


class CLIPFinetunedClassifier(nn.Module):
    """Variant 3: Fine-tune OpenAI CLIP dual encoders + MLP head over concatenated embeddings."""

    def __init__(
        self,
        num_classes: int,
        clip_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        h = self.clip.config.projection_dim
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h, num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        z = torch.cat([out.image_embeds, out.text_embeds], dim=-1)
        return self.head(z)


class ViTBERTCrossAttentionClassifier(nn.Module):
    """Variant 4: ViT patch tokens + BERT [CLS] with lightweight cross-attention fusion."""

    def __init__(
        self,
        num_classes: int,
        vit_name: str = "google/vit-base-patch16-224",
        bert_name: str = "bert-base-multilingual-cased",
        num_heads: int = 8,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_name)
        self.bert = BertModel.from_pretrained(bert_name)
        self.d_v = self.vit.config.hidden_size
        self.d_t = self.bert.config.hidden_size
        self.proj_t = nn.Linear(self.d_t, self.d_v)
        self.attn = nn.MultiheadAttention(self.d_v, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.d_v)
        self.pool = nn.Linear(self.d_v, 1)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.d_v, self.d_v // 2),
            nn.GELU(),
            nn.Linear(self.d_v // 2, num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vseq = self.vit(pixel_values=pixel_values).last_hidden_state
        cls_txt = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0:1, :]
        q = self.proj_t(cls_txt)
        fused, _ = self.attn(query=q, key=vseq, value=vseq)
        fused = self.norm(fused + q)
        w = torch.softmax(self.pool(fused), dim=1)
        pooled = (w * fused).sum(dim=1)
        return self.head(pooled)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal loss for imbalanced answer distribution (used as learning-strategy variant)."""
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()
