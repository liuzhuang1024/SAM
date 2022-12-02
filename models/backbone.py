import torch
import torch.nn as nn
import pickle as pkl
import torch.nn.functional as F

import models
from models.densenet import DenseNet

from einops.layers.torch import Rearrange
from traceback import print_exc

class Model(nn.Module):
    def __init__(self, params={}):
        super(Model, self).__init__()
        self.params = params

        with open(params['matrix_path'], 'rb') as f:
            matrix = pkl.load(f)
        self.matrix = torch.Tensor(matrix).to(device=params['device'])

        assert params['sim_loss']['type'] in ['l1', 'l2']
        self.sim_loss_type = params['sim_loss']['type']

        self.use_label_mask = params['use_label_mask']
        self.encoder = DenseNet(params=self.params)

        self.decoder = getattr(models, params['decoder']['net'])(params=self.params)
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()

        self.ignore_symbols = set(self.params['words'].encode(['{', '}', '^', '_']) + [1] + [0])

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

        if self.params['context_loss'] or self.params['word_state_loss']:
            self.cma_context = nn.Sequential(
                nn.Linear(params['encoder']['out_channel'], params['decoder']['input_size']),
                Rearrange("b l h->b h l"),
                nn.BatchNorm1d(params['decoder']['input_size']),
                Rearrange("b h l->b l h"),
                nn.ReLU()
            )
            self.cma_word = nn.Sequential(
                nn.Linear(params['decoder']['input_size'], params['decoder']['input_size']),
                Rearrange("b l h->b h l"),
                nn.BatchNorm1d(params['decoder']['input_size']),
                Rearrange("b h l->b l h"),
                nn.ReLU()
            )

    def forward(self, images, images_mask, labels, labels_mask, matrix=None, counting_labels=None, is_train=True):
        cnn_features = self.encoder(images)

        word_probs, word_alphas, embedding = self.decoder(cnn_features, labels, images_mask, labels_mask, counting_labels=counting_labels, is_train=is_train)

        context_loss, word_state_loss, word_sim_loss, counting_loss = 0, 0, 0, 0
        embedding, word_context_vec_list, word_out_state_list, _, counting_loss = embedding
        if self.params['context_loss'] or self.params['word_state_loss'] and is_train:
            if 'context_loss' in self.params and self.params['context_loss']:
                word_context_vec_list = torch.stack(word_context_vec_list, 1)
                context_embedding = self.cma_context(word_context_vec_list)
                context_loss = self.cal_cam_loss_v2(context_embedding, labels, matrix)
            if 'word_state_loss' in self.params and self.params['word_state_loss']:
                word_out_state_list = torch.stack(word_out_state_list, 1)
                word_state_embedding = self.cma_word(word_out_state_list)
                word_state_loss = self.cal_cam_loss_v2(word_state_embedding, labels, matrix)
                
        word_loss = self.cross(word_probs.contiguous().view(-1, word_probs.shape[-1]), labels.view(-1))
        word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (labels_mask.sum() + 1e-10) if self.use_label_mask else word_loss

        if 'sim_loss' in self.params and self.params['sim_loss']['use_flag']:
            word_sim_loss = self.cal_word_similarity(embedding)

        return word_probs, (word_average_loss, word_sim_loss, context_loss, word_state_loss, counting_loss)


    def cal_cam_loss_v2(self, word_embedding, labels, matrix):
        (B, L, H), device = word_embedding.shape, word_embedding.device
        
        W = torch.matmul(word_embedding, word_embedding.transpose(-1, -2)) # B L L
        denom = torch.matmul(word_embedding.unsqueeze(-2), word_embedding.unsqueeze(-1)).squeeze(-1) ** (0.5)
        # B L 1 H @ B L H 1 -> B L 1 1
        cosine = W / (denom @ denom.transpose(-1, -2))
        sim_mask = matrix != 0
        if self.sim_loss_type == 'l1':
            loss = abs((cosine - matrix) * sim_mask)
        else:
            loss = (cosine - matrix) ** 2 * sim_mask
        return loss.sum() / B / (labels != 0).sum()
    
    def cal_word_similarity(self, word_embedding):

        num = word_embedding @ word_embedding.transpose(1,0)

        denom = torch.matmul(word_embedding.unsqueeze(1), word_embedding.unsqueeze(2)).squeeze(1) ** (0.5)

        cosine = num / (denom @ denom.transpose(1, 0))

        sim_mask = self.matrix != 0

        if self.sim_loss_type == 'l1':
            loss = abs((cosine - self.matrix) * sim_mask)
        else:
            loss = (cosine - self.matrix) ** 2 * sim_mask

        loss = loss.sum() / sim_mask.sum()

        return loss





