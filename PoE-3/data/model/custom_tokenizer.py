# model/custom_tokenizer.py
import re
from transformers import PreTrainedTokenizerFast

class CustomGPTNeoXTokenizer(PreTrainedTokenizerFast):
    def __init__(self, tokenizer_object, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_object = tokenizer_object
        self.special_tokens_map = {'additional_special_tokens': ['[NUM]', '[/NUM]']}
        self.add_special_tokens(self.special_tokens_map)

    def tokenize(self, text, **kwargs):
        text = re.sub(r'(\d+(\.\d+)?)', r'[NUM] \1 [/NUM]', text)
        tokens = []
        i = 0
        while i < len(text):
            if text[i:i+5] == "[NUM]":
                tokens.append("[NUM]")
                i += 5
            elif text[i:i+6] == "[/NUM]":
                tokens.append("[/NUM]")
                i += 6
            elif text[i].isdigit() or text[i] == '.':
                while i < len(text) and (text[i].isdigit() or text[i] == '.'):
                    tokens.append(text[i])
                    i += 1
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.tokenizer_object.convert_tokens_to_ids(token) for token in tokens]

