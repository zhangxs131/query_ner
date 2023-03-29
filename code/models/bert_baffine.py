from transformers import BertPreTrainedModel,BertModel
import torch.nn as nn
import torch
from models.layers.baffine import Biaffine


class BertBiaffineForNer(BertPreTrainedModel):

    def __init__(
        self,
        config,
        encoder_trained=True,
        biaffine_size=128,
        lstm_dropout=0.4,
        select_bert_layer=-1
    ):
        super(BertBiaffineForNer, self).__init__(config)

        self.num_labels = config.num_labels
        self.select_bert_layer = select_bert_layer

        self.bert = BertModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.lstm = torch.nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )

        self.start_encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2*config.hidden_size,
                out_features=biaffine_size),
            torch.nn.ReLU()
        )

        self.end_encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2*config.hidden_size,
                out_features=biaffine_size),
            torch.nn.ReLU()
        )

        self.baffine = Biaffine(biaffine_size, self.num_labels)

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.start_encoder[0].weight)
        nn.init.xavier_uniform_(self.end_encoder[0].weight)

    def _calculate_loss(
            self, span_scores: torch.Tensor, span_labels: torch.LongTensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        span_labels : (batch_size, seq_len, seq_len)
        span_scores : (batch_size, seq_len, seq_len, num_classes)
        """

        label_mask = torch.triu(mask.unsqueeze(-1).expand_as(span_labels).clone())

        loss = nn.functional.cross_entropy(
            span_scores.reshape(-1, self.num_labels),
            span_labels.reshape(-1),
        )
        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        label_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True
        )

        sequence_output = outputs.hidden_states[self.select_bert_layer]

        # lstm编码
        sequence_output, _ = self.lstm(sequence_output)

        start_logits = self.start_encoder(sequence_output)
        end_logits = self.end_encoder(sequence_output)

        span_logits = self.baffine(start_logits, end_logits)
        span_logits = span_logits.contiguous()



        outputs=(span_logits,) + outputs[2:]

        loss=None
        if label_ids != None:

            mask = attention_mask

            loss = self._calculate_loss(span_logits, label_ids,mask)
            outputs = (loss,) + outputs


        return {'loss':loss,'logits':span_logits}