# model/train.py
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

def train_model(model, dataloader, num_epochs=3, learning_rate=5e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch.squeeze(1).to(model.device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    return model

