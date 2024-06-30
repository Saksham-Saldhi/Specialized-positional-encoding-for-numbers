# evaluation/evaluate.py
from datasets import load_metric
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizer

def evaluate_model(model_name, tokenizer_name, validation_dataset):
    model = GPTNeoXForCausalLM.from_pretrained(model_name)
    tokenizer = GPTNeoXTokenizer.from_pretrained(tokenizer_name)

    validation_data = CustomDropDataset(validation_dataset, tokenizer)
    validation_dataloader = DataLoader(validation_data, batch_size=2, shuffle=False)

    model.eval()
    predictions = []
    references = []

    for batch in validation_dataloader:
        input_ids = batch.squeeze(1).to(model.device)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=256)
        for output in outputs:
            pred_text = tokenizer.decode(output, skip_special_tokens=True)
            predictions.append(pred_text)

    for entry in validation_dataset:
        references.append(entry['answer'])

    metric = load_metric("accuracy")
    metric.add_batch(predictions=predictions, references=references)
    final_score = metric.compute()

    return final_score

