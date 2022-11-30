import torch
import torch.nn as nn
import pickle as pkl
import torch.nn.functional as F

import models
from models.densenet import DenseNet


class Model(nn.Module):
    def __init__(self, params=None):
        super(Model, self).__init__()
        self.params = params

        with open(params['matrix_path'], 'rb') as f:
            matrix = pkl.load(f)
        self.matrix = torch.Tensor(matrix).to(device=params['device'])

        assert params['sim_loss']['type'] in ['l1', 'l2']
        self.sim_loss_type = params['sim_loss']['type']

        self.use_label_mask = params['use_label_mask']
        self.encoder = DenseNet(params=self.params)
        # self.in_channel = params['counting_decoder']['in_channel']
        # self.out_channel = params['counting_decoder']['out_channel']

        # self.output_counting_feature = params['output_counting_feature'] if 'output_counting_feature' in params else False
        # self.channel_attn_feature = params['output_channel_attn_feature'] if 'output_channel_attn_feature' in params else False

        self.decoder = getattr(models, params['decoder']['net'])(params=self.params)
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()

        self.ignore_symbols = set(self.params['words'].encode(['{', '}', '^', '_']) + [1] + [0])

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

    def forward(self, images, images_mask, labels, labels_mask, is_train=True):
        cnn_features = self.encoder(images)

        word_probs, word_alphas, embedding = self.decoder(cnn_features, labels, images_mask, labels_mask, is_train=is_train)

        word_loss = self.cross(word_probs.contiguous().view(-1, word_probs.shape[-1]), labels.view(-1))
        word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (labels_mask.sum() + 1e-10) if self.use_label_mask else word_loss

        word_sim_loss = self.cal_word_similarity(embedding)

        return word_probs, (word_average_loss, word_sim_loss)


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





