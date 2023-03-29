from transformers import BertPreTrainedModel,BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy


class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config,soft_label=True,loss_type='ce'):
        super(BertSpanForNer, self).__init__(config)
        self.loss_type = loss_type
        self.soft_label=soft_label
        self.num_labels = config.num_labels
        if self.loss_type == 'lsr':
            self.loss_fct = LabelSmoothingCrossEntropy()
        elif self.loss_type == 'focal':
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.start_fc = nn.Linear(config.hidden_size, self.num_labels)
        if soft_label:
            hidden_size=config.hidden_size + self.num_labels
        else:
            hidden_size=config.hidden_size + 1

        self.end_fc =nn.Sequential( nn.Linear(hidden_size, hidden_size),
                                    nn.Tanh(),
                                    nn.LayerNorm(hidden_size),
                                    nn.Linear(hidden_size, self.num_labels))

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,end_positions=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()

        end_logits = self.end_fc(torch.cat([sequence_output, label_logits], dim=-1))

        loss=None
        if start_positions is not None and end_positions is not None:

            start_logits_temp = start_logits.contiguous().view(-1, self.num_labels)
            end_logits_temp = end_logits.contiguous().view(-1, self.num_labels)
            active_loss = attention_mask.contiguous().view(-1) == 1
            active_start_logits = start_logits_temp[active_loss]
            active_end_logits = end_logits_temp[active_loss]

            active_start_labels = start_positions.contiguous().view(-1)[active_loss]
            active_end_labels = end_positions.contiguous().view(-1)[active_loss]

            start_loss = self.loss_fct(active_start_logits, active_start_labels)
            end_loss = self.loss_fct(active_end_logits, active_end_labels)
            loss = (start_loss + end_loss) / 2


        return {'loss':loss,'start_logits':start_logits,'end_logits':end_logits}
