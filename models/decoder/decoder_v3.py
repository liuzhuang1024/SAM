import torch
import torch.nn as nn
from models.decoder.attention import Attention
import models

class Decoder_v1(nn.Module):
    def __init__(self, params):
        super(Decoder_v1, self).__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channel']
        self.attention_dim = params['attention']['attention_dim']
        self.dropout_prob = params['dropout']
        self.device = params['device']
        self.word_num = params['word_num']
        # self.counting_num = params['counting_decoder']['out_channel']

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

        # init hidden state
        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)
        # word embedding
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        # word gru
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.word_output_gru = nn.GRUCell(self.out_channel, self.hidden_size)
        # attention
        self.word_attention = Attention(params)
        self.encoder_feature_conv = nn.Conv2d(self.out_channel, self.attention_dim,
                                              kernel_size=params['attention']['word_conv_kernel'],
                                              padding=params['attention']['word_conv_kernel']//2)

        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Linear(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        # self.counting_context_weight = nn.Linear(self.counting_num, self.hidden_size)
        self.word_convert = nn.Linear(self.hidden_size, self.word_num)
        
        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])

        if "counting_decoder" in self.params and self.params['counting_decoder']['use_flag']:
            self.counting_context_weight = nn.Linear(self.params['counting_decoder']['out_channel'], self.params['decoder']['hidden_size'])
            self.counting_decoder1 = getattr(models, "counting_decoder")(self.params['counting_decoder']['in_channel'], self.params['counting_decoder']['out_channel'], 3)
            self.counting_decoder2 = getattr(models, "counting_decoder")(self.params['counting_decoder']['in_channel'], self.params['counting_decoder']['out_channel'], 5)

            self.counting_loss = nn.SmoothL1Loss(reduction='mean')
            
    def counting(self, cnn_features, images_mask, counting_labels):
        counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]

        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        counting_preds = (counting_preds1 + counting_preds2) / 2
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) \
                        + self.counting_loss(counting_preds, counting_labels)
        
        return counting_loss, counting_preds
    
    def forward(self, cnn_features, labels, images_mask, label_mask, counting_labels=None, is_train=True):
        counting_loss, counting_context_weighted = 0., 0.
        if "counting_decoder" in self.params and self.params['counting_decoder']['use_flag']:
            counting_loss, counting_preds = self.counting(cnn_features, images_mask, counting_labels) 
            counting_context_weighted = self.counting_context_weight(counting_preds)

        batch_size, num_steps = labels.shape
        height, width = cnn_features.shape[2:]
        word_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(device=self.device)
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]

        word_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
        word_alphas = torch.zeros((batch_size, num_steps, height, width)).to(device=self.device)
        hidden = self.init_hidden(cnn_features, images_mask)

        cnn_features_trans = self.encoder_feature_conv(cnn_features)

        word_context_vec_list, label_list, word_out_state_list = [], [], []
        if is_train:
            for i in range(num_steps):
                word_embedding = self.embedding(labels[:, i-1]) if i else self.embedding(torch.ones([batch_size]).long().to(self.device))
                hidden = self.word_input_gru(word_embedding, hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                                   word_alpha_sum, images_mask)
                hidden = self.word_output_gru(word_context_vec, hidden)
                                                                                   
                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

                word_prob = self.word_convert(word_out_state)
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha
                
                word_context_vec_list.append(word_context_vec)
                label_list.append(labels[:, i])
                word_out_state_list.append(word_out_state)

        else:
            word_embedding = self.embedding(torch.ones([batch_size]).long().to(device=self.device))
            for i in range(num_steps):
                hidden = self.word_input_gru(word_embedding, hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                                   word_alpha_sum, images_mask)
                hidden = self.word_output_gru(word_context_vec, hidden)

                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

                word_prob = self.word_convert(word_out_state)
                _, word = word_prob.max(1)
                word_embedding = self.embedding(word)
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha
                
                # word and context collect
                word_context_vec_list.append(word_context_vec)
                label_list.append(labels[:, i])
                word_out_state_list.append(word_out_state)
            self.word_context_vec_list = word_context_vec_list
            self.word_out_state_list = word_out_state_list
            
        return word_probs, word_alphas, (self.embedding.weight, word_context_vec_list, word_out_state_list, label_list, counting_loss)

    def init_hidden(self, features, feature_mask):
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)
        return torch.tanh(average)


