from transformers import BertPreTrainedModel,BertModel
import torch.nn as nn
from models.layers.crf import CRF


class BertBilstmCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertBilstmCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            batch_first=True,
            num_layers=config.bilstm_layers,
            dropout=config.hidden_dropout_prob,
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,label_ids=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        bilstm_output, _ = self.bilstm(sequence_output)
        sequence_output = self.dropout(bilstm_output)
        logits = self.classifier(sequence_output)

        loss=None
        if label_ids is not None:
            loss = self.crf(emissions = logits, tags=label_ids, mask=attention_mask)
            outputs =(-1*loss,)+outputs


        return {'loss':loss,'logits':logits}