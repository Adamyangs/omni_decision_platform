
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(
            self,
            attention_dropout=0.0
    ):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(
            self,
            q,
            k ,
            v,
            scale=None,
            attn_mask=None
    ):
        attention = th.bmm(q, k.transpose(1, 2))

        if scale:
            attention *= scale
        if attn_mask:
            attention = attn_mask.masked_fill_(attn_mask, -np.inf)

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = th.bmm(attention, v)

        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            in_dim,
            model_dim=512,
            n_heads=8,
            dropout=0.0
    ):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // n_heads
        self.n_heads= n_heads

        self.residual_fc = nn.Linear(in_dim, model_dim)

        self.linear_k = nn.Linear(in_dim, self.dim_per_head * n_heads)
        self.linear_v = nn.Linear(in_dim, self.dim_per_head * n_heads)
        self.linear_q = nn.Linear(in_dim, self.dim_per_head * n_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(
            self,
            k,
            v,
            q,
            attn_mask=None
    ):
        dim_per_head = self.dim_per_head
        n_heads = self.n_heads
        batch_size = k.size(0)

        residual = self.residual_fc(q)

        k = self.linear_k(k).view(batch_size * n_heads, -1, dim_per_head)
        v = self.linear_v(v).view(batch_size * n_heads, -1, dim_per_head)
        q = self.linear_q(q).view(batch_size * n_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)

        scale = (k.size(-1)) ** (-0.5)
        context, attention = self.dot_product_attention(q, k, v, scale, attn_mask)

        context = context.view(batch_size, -1, dim_per_head * n_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)

        return output, attention

class MultiHeadAttentionV2(nn.Module):
    def __init__(
            self,
            k_in_dim,
            v_in_dim,
            q_in_dim,
            model_dim=512,
            n_heads=8,
            dropout=0.0
    ):
        super(MultiHeadAttentionV2, self).__init__()

        self.dim_per_head = model_dim // n_heads
        self.n_heads= n_heads

        self.residual_fc = nn.Linear(q_in_dim, model_dim)

        self.linear_k = nn.Linear(k_in_dim, self.dim_per_head * n_heads)
        self.linear_v = nn.Linear(v_in_dim, self.dim_per_head * n_heads)
        self.linear_q = nn.Linear(q_in_dim, self.dim_per_head * n_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(
            self,
            k,
            v,
            q,
            attn_mask=None
    ):
        dim_per_head = self.dim_per_head
        n_heads = self.n_heads
        batch_size = k.size(0)

        residual = self.residual_fc(q)

        k = self.linear_k(k).view(batch_size * n_heads, -1, dim_per_head)
        v = self.linear_v(v).view(batch_size * n_heads, -1, dim_per_head)
        q = self.linear_q(q).view(batch_size * n_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)

        scale = (k.size(-1)) ** (-0.5)
        context, attention = self.dot_product_attention(q, k, v, scale, attn_mask)

        context = context.view(batch_size, -1, dim_per_head * n_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)

        return output, attention

class MultiHeadAttentionV3(nn.Module):
    def __init__(
            self,
            model_dim=512,
            n_heads=8,
            dropout=0.0
    ):
        super(MultiHeadAttentionV3, self).__init__()

        self.dim_per_head = model_dim // n_heads
        self.n_heads= n_heads

        self.dot_product_attention = ScaledDotProductAttention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(
            self,
            k,
            v,
            q,
            attn_mask=None
    ):
        dim_per_head = self.dim_per_head
        n_heads = self.n_heads
        batch_size = k.size(0)

        k = k.view(batch_size * n_heads, -1, dim_per_head)
        v = v.view(batch_size * n_heads, -1, dim_per_head)
        q = q.view(batch_size * n_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)

        scale = (k.size(-1)) ** (-0.5)
        context, attention = self.dot_product_attention(q, k, v, scale, attn_mask)

        context = context.view(batch_size, -1, dim_per_head * n_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(q.view(batch_size, -1, dim_per_head * n_heads) + output)

        return output, attention



class PositionalWiseFeedForward(nn.Module):
    def __init__(
            self,
            model_dim=512,
            ffn_dim=2048,
            dropout=0.0
    ):
        super(PositionalWiseFeedForward, self).__init__()

        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(
            self,
            x
    ):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        output = self.layer_norm(x + output)

        return output


class EncoderLayer(nn.Module):
    def __init__(
            self,
            in_dim,
            model_dim=512,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.0
    ):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(in_dim, model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(
            self,
            inputs,
            attn_mask=None
    ):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)

        return output, attention

class EncoderLayerv2(nn.Module):
    def __init__(
            self,
            k_in_dim,
            q_in_dim,
            v_in_dim,
            model_dim=512,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.0
    ):
        super(EncoderLayerv2, self).__init__()

        self.attention = MultiHeadAttentionV2(k_in_dim=k_in_dim, 
                                              q_in_dim=q_in_dim, 
                                              v_in_dim=v_in_dim, 
                                              model_dim=model_dim, 
                                              n_heads=num_heads, 
                                              dropout=dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(
            self,
            k,
            v,
            q,
            attn_mask=None
    ):
        context, attention = self.attention(k, v, q, attn_mask)
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    def __init__(
            self,
            num_layers=6,
            model_dim=512,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )

    def forward(
            self,
            inputs
    ):
        output = inputs

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, None)
            attentions.append(attention)

        return output, attentions

class DynamicEncoderLayer(nn.Module):
    def __init__(self, in_dim, model_dim, num_heads, ffn_dim, dropout=0.0):
        super(DynamicEncoderLayer, self).__init__()

        self.attention = DynamicMultiHeadAttention(in_dim, model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, k, v, q, r, attn_mask=None):
        context, attention = self.attention(k, q, v, r, attn_mask)
        output = self.feed_forward(context)

        return output, attention

class DynamicMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, model_dim, n_heads, dropout=0.0):
        super(DynamicMultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // n_heads
        self.n_heads = n_heads

        self.dot_product_attention = ScaledDotProductAttention(dropout)

        self.fc_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, k, v, q, r, attn_mask=None):
        bz = k.size(0)

        k = k.view(bz * self.n_heads, -1, self.dim_per_head)
        v = v.view(bz * self.n_heads, -1, self.dim_per_head)
        q = q.view(bz * self.n_heads, -1, self.dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.n_heads, 1, 1)

        scale = (k.size(-1)) ** (0.5)
        context, attention = self.dot_product_attention(q, k, v, scale, attn_mask)

        context = context.view(bz, -1, self.dim_per_head * self.n_heads)
        output = self.fc_final(context)
        output = self.dropout(output)
        output = self.layer_norm(r + output)

        return output, attention

    
class ScaleDynamicMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, model_dim, n_scale, dropout=0.0):
        super(ScaleDynamicMultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // n_scale
        self.n_scale = n_scale

        self.dot_product_attention = ScaledDotProductAttention(dropout)

        self.fc_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, k, v, q, r, attn_mask=None):
        bz = k.size(0)

        k = k.view(bz, -1, self.n_scale, self.dim_per_head)
        v = v.view(bz, -1, self.n_scale, self.dim_per_head)
        q = q.view(bz, -1, self.n_scale, self.dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.n_heads, 1, 1)

        scale = (k.size(-1)) ** (0.5)

        context, attention = self.dot_product_attention(q, k, v, scale, attn_mask)
        context_tot = [context]
        for i in range(1, self.n_scale):
            context, attention = self.dot_product_attention(context, context, v, scale, attn_mask)
            context_tot.append(context)
        
        context_tot = th.stack(context_tot, dim=-2).view(bz, -1, self.dim_per_head * self.n_scale)

        output = self.fc_final(context_tot)
        output = self.dropout(output)
        output = self.layer_norm(r + output)

        return output, None
        


class ScaleDynamicEncoderLayer(nn.Module):
    def __init__(self, in_dim, model_dim, num_heads, ffn_dim, dropout=0.0):
        super(ScaleDynamicEncoderLayer, self).__init__()

        self.attention = ScaleDynamicMultiHeadAttention(in_dim, model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, k, v, q, r, attn_mask=None):
        context, attention = self.attention(k, q, v, r, attn_mask)
        output = self.feed_forward(context)

        return output, attention   


class ScaleMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, model_dim, n_head, n_scale=2, dropout=0.0):
        super(ScaleMultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // n_head
        self.n_heads = n_head
        self.n_scale = n_scale

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.r = nn.Linear(model_dim, model_dim * n_scale)

        self.fc_final = nn.Linear(model_dim * n_scale, model_dim * n_scale)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim * n_scale)
    
    def forward(self, k, v, q, r, attn_mask=None):
        bz = k.size(0)

        k = k.view(bz * self.n_heads, -1, self.dim_per_head)
        v = v.view(bz * self.n_heads, -1, self.dim_per_head)
        q = q.view(bz * self.n_heads, -1, self.dim_per_head)

        r = self.r(r)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.n_heads, 1, 1)

        scale = (k.size(-1)) ** (0.5)

        context, attention = self.dot_product_attention(q, k, v, scale, attn_mask)
        
        # n_scale: 2
        new_context, _ = self.dot_product_attention(context, k, v, scale, attn_mask)
        context_tot = th.stack([context, new_context], dim=-2)

        context = context_tot.view(bz, -1, self.dim_per_head * self.n_heads * self.n_scale)
        output = self.fc_final(context)
        output = self.dropout(output)
        output = self.layer_norm(r + output)

        return output, attention

class ScaleEncoderLayer(nn.Module):
    def __init__(self, in_dim, model_dim, num_scale, ffn_dim, dropout=0.0):
        super(ScaleEncoderLayer, self).__init__()

        self.attention = ScaleMultiHeadAttention(in_dim, model_dim, num_scale, dropout=dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim * 2, ffn_dim, dropout)

    def forward(self, k, v, q, r, attn_mask=None):
        context, attention = self.attention(k, q, v, r, attn_mask)
        output = self.feed_forward(context)

        return output, attention  


class HeterogeneousEncoderLayer(nn.Module):
    def __init__(self, in_dim, model_dim, num_heads, dropout=0.0):
        super(HeterogeneousEncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(in_dim, model_dim, num_heads, dropout)
        slef.heter_attention = MultiHeadAttention(in_dim, model_dim, num_heads, dropout)
       

    def forward(self, inputs, attn_mask=None):
        _, heter_attention = self.heter_attention(inputs, inputs, inputs, attn_mask)

        onses = th.ones_like(attn_mask, device=inputs.device)
        attn_mask = th.where(attn_mask==0., onses, heter_attention)

        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        return context, attention