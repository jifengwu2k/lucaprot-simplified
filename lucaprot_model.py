from torch.nn import BCEWithLogitsLoss
import torch
from transformers import BertConfig

from modeling_bert import BertPreTrainedModel, BertModel


class GlobalMaskValueAttentionPooling1D(torch.nn.Module):
    def __init__(
        self,
        embed_size,
        units=None,
        use_additive_bias=False,
        use_attention_bias=False,
    ):  # type: (int, object, bool, bool) -> None
        super(GlobalMaskValueAttentionPooling1D, self).__init__()
        self.embed_size = embed_size
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.units = units if units else embed_size

        self.U = torch.nn.Parameter(torch.Tensor(self.embed_size, self.units))
        self.V = torch.nn.Parameter(torch.Tensor(self.embed_size, self.units))
        if self.use_additive_bias:
            self.b1 = torch.nn.Parameter(torch.Tensor(self.units))
            torch.nn.init.trunc_normal_(self.b1, std=0.01)
        if self.use_attention_bias:
            self.b2 = torch.nn.Parameter(torch.Tensor(self.embed_size))
            torch.nn.init.trunc_normal_(self.b2, std=0.01)

        self.W = torch.nn.Parameter(torch.Tensor(self.units, self.embed_size))

        torch.nn.init.trunc_normal_(self.U, std=0.01)
        torch.nn.init.trunc_normal_(self.V, std=0.01)
        torch.nn.init.trunc_normal_(self.W, std=0.01)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.U)
        k = torch.matmul(x, self.V)
        if self.use_additive_bias:
            h = torch.tanh(q + k + self.b1)
        else:
            h = torch.tanh(q + k)

        if self.use_attention_bias:
            e = torch.matmul(h, self.W) + self.b2
        else:
            e = torch.matmul(h, self.W)
        if mask is not None:
            attention_probs = torch.nn.Softmax(dim=1)(
                e + torch.unsqueeze((1.0 - mask) * -10000, dim=-1)
            )
        else:
            attention_probs = torch.nn.Softmax(dim=1)(e)
        x = torch.sum(attention_probs * x, dim=1)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.embed_size)
            + " -> "
            + str(self.embed_size)
            + ")"
        )


def create_activation_function(activation_function_name):  # type: (str) -> torch.nn.Module
    if activation_function_name == "tanh":
        return torch.nn.Tanh()
    elif activation_function_name == "relu":
        return torch.nn.ReLU()
    elif activation_function_name == "leakyrelu":
        return torch.nn.LeakyReLU()
    elif activation_function_name == "gelu":
        return torch.nn.GELU()
    else:
        return torch.nn.Tanh()


class SequenceAndStructureFusionNetwork(BertPreTrainedModel):
    def __init__(self, config):  # type: (BertConfig) -> None
        super().__init__(config)

        self.seq_encoder = BertModel(config)
        self.seq_pooler = GlobalMaskValueAttentionPooling1D(
            embed_size=config.hidden_size
        )

        self.seq_linear = torch.nn.ModuleList()

        input_size = config.hidden_size
        for output_size in config.seq_fc_size:
            seq_linear_module = torch.nn.Linear(input_size, output_size)

            self.seq_linear.append(seq_linear_module)
            self.seq_linear.append(create_activation_function(config.activate_func))

            input_size = output_size

        self.embedding_pooler = GlobalMaskValueAttentionPooling1D(
            embed_size=config.embedding_input_size
        )

        self.embedding_linear = torch.nn.ModuleList()

        input_size = config.embedding_input_size
        for output_size in config.embedding_fc_size:
            embedding_linear_module = torch.nn.Linear(input_size, output_size)

            self.embedding_linear.append(embedding_linear_module)
            self.embedding_linear.append(
                create_activation_function(config.activate_func)
            )

            input_size = output_size

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(
            config.seq_fc_size[-1] + config.embedding_fc_size[-1], 1
        )
        self.output = torch.nn.Sigmoid()
        self.loss_fct = BCEWithLogitsLoss(
            pos_weight=torch.tensor([config.pos_weight], dtype=torch.long)
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        struct_input_ids=None,
        struct_contact_map=None,
        embedding_info=None,
        embedding_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        seq_outputs = self.seq_encoder(
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
        seq_pooled_output = self.seq_pooler(seq_outputs[0])
        seq_pooled_output = self.dropout(seq_pooled_output)
        for seq_linear_module in self.seq_linear:
            seq_pooled_output = seq_linear_module(seq_pooled_output)

        embedding_pooled_output = self.embedding_pooler(
            embedding_info, mask=embedding_attention_mask
        )
        embedding_pooled_output = self.dropout(embedding_pooled_output)
        for embedding_linear_module in self.embedding_linear:
            embedding_pooled_output = embedding_linear_module(embedding_pooled_output)

        pooled_output = torch.cat([seq_pooled_output, embedding_pooled_output], dim=-1)
        logits = self.classifier(pooled_output)
        output = self.output(logits)

        return logits, output
