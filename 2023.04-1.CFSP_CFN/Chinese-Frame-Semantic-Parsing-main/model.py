from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput
import allennlp.modules.span_extractors.max_pooling_span_extractor as max_pooling_span_extractor
from allennlp.nn.util import get_mask_from_sequence_lengths, masked_log_softmax

@dataclass
class FrameSRLModelOutput(ModelOutput):
    """

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, FE_num, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, FE_num, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertForFrameSRL(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        # self.bert = BertModel(config)
        # self.start_pointer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # self.end_pointer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.ffn = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.mlp = nn.Sequential(self.ffn, self.activation, self.classifier)
        self.span_extractor = max_pooling_span_extractor.MaxPoolingSpanExtractor(config.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        # self.loss_fct_nll = nn.NLLLoss(ignore_index=-1)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        span_token_idx: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        loss = None


        # span_token_idx (B, span_num, 2) -> span_rep (B, span_num, H) allennlp maxpoolingspanextractor
        # logits (B, num_labels, span_num)

        span_rep = self.span_extractor(sequence_output, span_token_idx)
        logits = self.mlp(span_rep).permute(0, 2, 1)

        if labels is not None:
            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return FrameSRLModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        



