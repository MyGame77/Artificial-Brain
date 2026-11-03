from pathlib import Path
import numpy as np

from tokenizer import Tokenizer


def load_dataset(sequence_length=8):
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    base_dir = Path(__file__).resolve().parent
    text_path = base_dir / "data" / "train.txt"
    vocab_path = base_dir / "checkpoints" / "vocabulary.json"

    if not text_path.exists():
        raise FileNotFoundError(f"Missing training text: {text_path}")

    if not vocab_path.exists():
        raise FileNotFoundError(
            "Vocabulary missing. Run python tokenizer.py first."
        )

    text = text_path.read_text(encoding="utf-8")
    tokenizer = Tokenizer()
    tokenizer.load(vocab_path)

    token_ids = tokenizer.encode(text)

    if len(token_ids) <= sequence_length:
        raise ValueError(
            f"Need more than {sequence_length} tokens; "
            f"found {len(token_ids)}."
        )

    tokens = np.asarray(token_ids, dtype=np.int64)

    inputs = np.stack([
        tokens[start:start + sequence_length]
        for start in range(len(tokens) - sequence_length)
    ])

    targets = np.stack([
        tokens[start + 1:start + sequence_length + 1]
        for start in range(len(tokens) - sequence_length)
    ])

    return inputs, targets, tokenizer


def main():
    inputs, targets, tokenizer = load_dataset()

    print("Dataset loaded successfully.")
    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")

    print("\nExample training pair:")
    print("Input: ", tokenizer.decode(inputs[0]))
    print("Target:", tokenizer.decode(targets[0]))

    print("\nDataset arrays are ready for model training.")


if __name__ == "__main__":
    main()