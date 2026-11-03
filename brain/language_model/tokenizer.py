import json
import re
from pathlib import Path


class Tokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    @staticmethod
    def split(text):
        return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    def fit(self, text):
        tokens = sorted(set(self.split(text)))

        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        vocabulary = special_tokens + [
            token for token in tokens if token not in special_tokens
        ]

        self.token_to_id = {
            token: index for index, token in enumerate(vocabulary)
        }
        self.id_to_token = {
            index: token for token, index in self.token_to_id.items()
        }

    def encode(self, text):
        unknown_id = self.token_to_id["<unk>"]
        return [
            self.token_to_id.get(token, unknown_id)
            for token in self.split(text)
        ]

    def decode(self, token_ids):
        tokens = [
            self.id_to_token.get(int(token_id), "<unk>")
            for token_id in token_ids
        ]

        text = " ".join(
            token for token in tokens
            if token not in {"<pad>", "<bos>", "<eos>"}
        )

        return re.sub(r"\s+([.,!?;:])", r"\1", text)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.token_to_id, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path):
        vocabulary = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        self.token_to_id = vocabulary
        self.id_to_token = {
            index: token for token, index in vocabulary.items()
        }


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "train.txt"
    vocab_path = base_dir / "checkpoints" / "vocabulary.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Training text not found: {data_path}")

    text = data_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Training text is empty.")

    tokenizer = Tokenizer()
    tokenizer.fit(text)
    tokenizer.save(vocab_path)

    sample = "artificial brain learns language"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("Tokenizer initialized successfully.")
    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")
    print(f"Sample text: {sample}")
    print(f"Token IDs: {encoded}")
    print(f"Decoded text: {decoded}")
    print(f"Vocabulary saved to: {vocab_path}")


if __name__ == "__main__":
    main()