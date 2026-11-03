import numpy as np
from pathlib import Path
from prepare_data import prepare_data
from transformer import TransformerLM


CHECKPOINT = Path("checkpoints/transformer_weights.npz")
BATCH_SIZE = 8


def calculate_loss(model, inputs, targets):
    losses = []

    for start in range(0, len(inputs), BATCH_SIZE):
        batch_x = inputs[start:start + BATCH_SIZE]
        batch_y = targets[start:start + BATCH_SIZE]

        loss, _ = model.loss_and_gradients(batch_x, batch_y)

        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss.")

        losses.append((float(loss), len(batch_x)))

    total_examples = sum(count for _, count in losses)
    weighted_loss = sum(loss * count for loss, count in losses)

    return weighted_loss / total_examples


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    train_x, train_y, val_x, val_y, tokenizer = prepare_data()

    model = TransformerLM(
        len(tokenizer.token_to_id),
        sequence_length=8,
    )
    model.load(CHECKPOINT)

    print("Artificial Brain evaluation")
    print(f"Checkpoint epoch: {model.epoch}")
    print(f"Optimizer steps: {model.step}")
    print(f"Vocabulary: {model.vocab_size}")

    training_loss = calculate_loss(model, train_x, train_y)
    validation_loss = calculate_loss(model, val_x, val_y)

    print(f"Training loss:   {training_loss:.4f}")
    print(f"Validation loss: {validation_loss:.4f}")

    if validation_loss > training_loss * 1.5:
        print("Observation: validation loss is much higher.")
        print("More training may cause overfitting.")
    elif validation_loss < training_loss:
        print("Observation: validation loss is lower than training loss.")
    else:
        print("Observation: compare both losses after future training.")

    print("Evaluation complete. Checkpoint was not modified.")


if __name__ == "__main__":
    main()