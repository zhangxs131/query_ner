from transformers import BertPreTrainedModel,BertModel
import torch.nn as nn
import torch
from models.layers.globalpointer import GlobalPointer
from losses.gp_loss import GlobalPointerCrossEntropy



class BertGPForNer(BertPreTrainedModel):
    """
    GlobalPointer + Bert 的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://www.kexue.fm/archives/8373
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        head_size=64
    ):
        super(BertGPForNer, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.global_pointer = GlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size
        )

        self.loss_func=GlobalPointerCrossEntropy()

        self.init_weights()

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
        )

        sequence_output = outputs[0]

        logits = self.global_pointer(sequence_output, mask=attention_mask)


        outputs=(logits,)+ outputs[2:]

        loss=None
        if label_ids != None:

            loss = self.loss_func(logits,label_ids)
            outputs = (loss,) + outputs

        return {'loss':loss,'logits':logits}