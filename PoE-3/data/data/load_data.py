# data/load_data.py
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class CustomDropDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        context = self.dataset[idx]['passage']
        question = self.dataset[idx]['question']
        text = f"Context: {context} Question: {question}"
        encoded = self.tokenizer.encode(text, return_tensors="pt")
        input_ids = encoded[0]
        return input_ids

def load_drop_dataset(tokenizer, split="train[:10%]"):
    dataset = load_dataset("drop", split=split)
    return CustomDropDataset(dataset, tokenizer)

def get_dataloader(dataset, batch_size=2, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

