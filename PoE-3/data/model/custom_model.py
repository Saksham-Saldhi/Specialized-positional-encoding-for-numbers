# model/custom_model.py
import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

class CustomPositionalEncodingGPTNeoX(GPTNeoXForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.custom_position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is None:
            position_ids = self.create_custom_position_ids(input_ids)
        return super().forward(input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)

    def create_custom_position_ids(self, input_ids):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        custom_position_ids = position_ids.unsqueeze(0).expand_as(input_ids).clone()

        num_start = False
        for i, token in enumerate(input_ids[0]):
            if token == tokenizer.convert_tokens_to_ids("[NUM]"):
                num_start = True
                digits_before_decimal = []
                digits_after_decimal = []
                decimal_found = False
            elif num_start and tokenizer.convert_ids_to_tokens(token).isdigit():
                if not decimal_found:
                    digits_before_decimal.append(i)
                else:
                    digits_after_decimal.append(i)
            elif num_start and tokenizer.convert_ids_to_tokens(token) == '.':
                decimal_found = True
            elif num_start and token == tokenizer.convert_tokens_to_ids("[/NUM]"):
                num_len = len(digits_before_decimal)
                for idx, pos in enumerate(digits_before_decimal):
                    custom_position_ids[0][pos] = num_len - idx
                for idx, pos in enumerate(digits_after_decimal, start=1):
                    custom_position_ids[0][pos] = -idx
                num_start = False

        return custom_position_ids

def create_model(tokenizer_len):
    config = GPTNeoXConfig.from_pretrained("EleutherAI/pythia-70m")
    model = CustomPositionalEncodingGPTNeoX(config)
    model.resize_token_embeddings(tokenizer_len)
    return model

