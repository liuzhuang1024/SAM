import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.params = params
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']
        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

    def forward(self, cnn_features, cnn_features_trans, hidden, alpha_sum, image_mask=None):
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0,2,3,1))
        alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha + cnn_features_trans.permute(0,2,3,1))
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = torch.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        alpha_sum = alpha[:,None,:,:] + alpha_sum
        context_vector = (alpha[:,None,:,:] * cnn_features).sum(-1).sum(-1)
        return context_vector, alpha, alpha_sum



class AttentionWithLoc(nn.Module):

    def __init__(self, params):
        super(AttentionWithLoc, self).__init__()

        self.params = params
        self.channel = params['encoder']['out_channel']
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']

        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        self.encoder_feature_conv = nn.Conv2d(self.channel, self.attention_dim, kernel_size=1)

        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)


    def forward(self, cnn_features, cnn_features_trans, hidden, alpha_sum, image_mask=None, counting_map_trans=None):
        b,c,h,w = cnn_features.shape
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0,2,3,1))

        alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha + cnn_features_trans.permute(0,2,3,1) + counting_map_trans)
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()

        energy = energy.masked_fill(~image_mask.bool().permute(0,2,3,1),float('-inf'))
        energy = energy.squeeze(-1).contiguous().view(b, h*w)
        alpha = F.softmax(energy, dim=-1)
        alpha = alpha.contiguous().view(b, 1, h, w)

        alpha_sum = alpha + alpha_sum
        context_vector = (alpha.contiguous().view(b, 1, h, w) * cnn_features).sum(-1).sum(-1)

        # energy_exp = torch.exp(energy.squeeze(-1))
        # if image_mask is not None:
        #     energy_exp = energy_exp * image_mask.squeeze(1)
        # alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        # alpha_sum = alpha[:,None,:,:] + alpha_sum
        # context_vector = (alpha[:,None,:,:] * cnn_features).sum(-1).sum(-1)

        return context_vector, alpha.squeeze(1), alpha_sum


class Attention_new(nn.Module):
    def __init__(self, params):
        super(Attention_new, self).__init__()
        self.params = params
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']
        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

    def forward(self, cnn_features, cnn_features_trans, hidden, alpha_sum, alpha_trans, image_mask=None):
        b, c, h, w = cnn_features.shape
        query = self.hidden_weight(hidden)
        alpha_score = torch.tanh(query[:, None, None, :] + alpha_trans + cnn_features_trans.permute(0,2,3,1))
        energy = self.alpha_convert(alpha_score)

        energy = energy.masked_fill(~image_mask.bool().permute(0,2,3,1),float('-inf'))
        energy = energy.squeeze(-1).contiguous().view(b, h*w)
        alpha = F.softmax(energy, dim=-1)
        alpha = alpha.contiguous().view(b, 1, h, w)

        alpha_sum = alpha + alpha_sum
        context_vector = (alpha.contiguous().view(b, 1, h, w) * cnn_features).sum(-1).sum(-1)
        #
        # energy_exp = torch.exp(energy.squeeze(-1))
        # if image_mask is not None:
        #     energy_exp = energy_exp * image_mask.squeeze(1)
        # alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        # alpha_sum = alpha[:,None,:,:] + alpha_sum
        # context_vector = (alpha[:,None,:,:] * cnn_features).sum(-1).sum(-1)

        return context_vector, alpha.squeeze(1), alpha_sum