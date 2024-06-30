# main.py
from data.load_data.py import load_drop_dataset, get_dataloader
from model.custom_tokenizer import CustomGPTNeoXTokenizer
from model.custom_model import create_model
from model.train import train_model
from evaluation.evaluate import evaluate_model

def main():
    # Initialize tokenizer
    original_tokenizer = GPTNeoXTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer = CustomGPTNeoXTokenizer(tokenizer_object=original_tokenizer)

    # Load and prepare the dataset
    train_dataset = load_drop_dataset(tokenizer, split="train[:10%]")
    train_dataloader = get_dataloader(train_dataset, batch_size=2, shuffle=True)

    # Create model
    model = create_model(len(tokenizer))

    # Train the model
    model = train_model(model, train_dataloader, num_epochs=3, learning_rate=5e-5)

    # Save the model
    model.save_pretrained("custom-gpt-neox-drop")
    tokenizer.save_pretrained("custom-gpt-neox-drop")

    # Evaluate the model
    validation_dataset = load_drop_dataset(tokenizer, split="validation[:10%]")
    final_score = evaluate_model("custom-gpt-neox-drop", "custom-gpt-neox-drop", validation_dataset)
    print(f"Evaluation Score: {final_score}")

if __name__ == "__main__":
    main()

