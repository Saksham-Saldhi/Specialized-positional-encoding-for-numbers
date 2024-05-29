!pip install transformers datasets torch accelerate
!pip install --upgrade torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from torch.utils.data import Dataset, DataLoader
!pip install accelerate -U
# !pip install transformers[torch]
import torch
# !pip show accelerate
# torch.cuda.empty_cache()
# Load the DROP dataset
train_dataset = load_dataset("ucinlp/drop", split="train[:10%]") #Initially only load 10% of the data, when everything works, please remove [:10%]
validation_dataset = load_dataset("ucinlp/drop", split="validation[:10%]")

#------------------
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
#------------------

# Load the Pythia model and tokenizer
# model = GPTNeoXForCausalLM.from_pretrained(
#     "EleutherAI/pythia-14m",
#     revision="step3000",
#     cache_dir="./pythia-14m/step3000",
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     "EleutherAI/pythia-14m",
#     revision="step3000",
#     cache_dir="./pythia-14m/step3000",
# )
tokenizer.add_special_tokens({'pad_token': '[EOS]'})

def format_data(example):
    context = example['passage']
    question = example['question']
    answer = ' , '.join(example['answers_spans']['spans'])  # Adjusted to the correct path for answers in DROP dataset
    input_text = f"Context: {context} Question: {question} Answer:"
    return {'input_text': input_text, 'target_text': answer}

# Format the datasets
formatted_train_dataset = train_dataset.map(format_data, remove_columns=train_dataset.column_names)
formatted_validation_dataset = validation_dataset.map(format_data, remove_columns=validation_dataset.column_names)

# Tokenize the datasets
def tokenize_data(example):
    input_encodings = tokenizer(example['input_text'], truncation=True, padding="max_length", max_length=512)
    target_encodings = tokenizer(example['target_text'], truncation=True, padding="max_length", max_length=128)

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

tokenized_train_dataset = formatted_train_dataset.map(tokenize_data, batched=True)
tokenized_validation_dataset = formatted_validation_dataset.map(tokenize_data, batched=True)

# Ensure there are no empty entries
for idx, entry in enumerate(tokenized_train_dataset):
    if not entry['input_ids'] or not entry['labels']:
        print(f"Empty entry found at index {idx}")

# Ensure the dataset length is correct
print(f"Total samples in tokenized train dataset: {len(tokenized_train_dataset)}")
print(f"Total samples in tokenized validation dataset: {len(tokenized_validation_dataset)}")

class DropDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        labels = torch.tensor(item['labels'])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

train_dataset = DropDataset(tokenized_train_dataset)
validation_dataset = DropDataset(tokenized_validation_dataset)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

def evaluate_model(model, tokenizer, dataset):
    correct = 0
    total = 0

    for example in dataset:
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0)
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        answer = tokenizer.decode(torch.tensor(example['labels']), skip_special_tokens=True).strip()

        if prediction.lower() == answer.lower():
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy

accuracy = evaluate_model(model, tokenizer, tokenized_validation_dataset)
print(f"Model accuracy on DROP dataset: {accuracy}")
