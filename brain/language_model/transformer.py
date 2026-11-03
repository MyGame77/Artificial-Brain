import os
import tempfile
from pathlib import Path

import numpy as np


class TransformerLM:
    def __init__(
        self,
        vocab_size,
        d_model=32,
        ff_dim=64,
        sequence_length=8,
        learning_rate=0.001,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        rng = np.random.default_rng()

        def weight(shape, fan_in):
            return (
                rng.standard_normal(shape).astype(np.float32)
                * np.sqrt(2.0 / fan_in)
            )

        self.params = {
            "E": weight((vocab_size, d_model), d_model),
            "P": weight((sequence_length, d_model), d_model),
            "Wq": weight((d_model, d_model), d_model),
            "Wk": weight((d_model, d_model), d_model),
            "Wv": weight((d_model, d_model), d_model),
            "Wo": weight((d_model, d_model), d_model),
            "W1": weight((d_model, ff_dim), d_model),
            "b1": np.zeros(ff_dim, dtype=np.float32),
            "W2": weight((ff_dim, d_model), ff_dim),
            "b2": np.zeros(d_model, dtype=np.float32),
            "Wout": weight((d_model, vocab_size), d_model),
            "bout": np.zeros(vocab_size, dtype=np.float32),
        }

        self.m = {
            name: np.zeros_like(value)
            for name, value in self.params.items()
        }
        self.v = {
            name: np.zeros_like(value)
            for name, value in self.params.items()
        }

        self.step = 0
        self.epoch = 0

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = dict(self.params)
        state["optimizer_step"] = np.array(self.step, dtype=np.int64)
        state["saved_epoch"] = np.array(self.epoch, dtype=np.int64)
        state["vocab_size"] = np.array(self.vocab_size, dtype=np.int64)
        state["d_model"] = np.array(self.d_model, dtype=np.int64)
        state["ff_dim"] = np.array(self.ff_dim, dtype=np.int64)
        state["sequence_length"] = np.array(
            self.sequence_length, dtype=np.int64
        )

        for name in self.params:
            state[f"adam_m_{name}"] = self.m[name]
            state[f"adam_v_{name}"] = self.v[name]

        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )

        try:
            with os.fdopen(fd, "wb") as file:
                np.savez(file, **state)
                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_path, path)

            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def load(self, path):
        with np.load(path, allow_pickle=False) as state:
            for key, expected in (
                ("vocab_size", self.vocab_size),
                ("d_model", self.d_model),
                ("ff_dim", self.ff_dim),
                ("sequence_length", self.sequence_length),
            ):
                if key in state and int(state[key]) != expected:
                    raise ValueError(
                        f"Checkpoint {key} mismatch: "
                        f"expected {expected}, found {int(state[key])}"
                    )

            loaded_params = {}

            for name, current in self.params.items():
                if name not in state:
                    raise ValueError(
                        f"Checkpoint is missing parameter: {name}"
                    )

                value = state[name]

                if value.shape != current.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: "
                        f"expected {current.shape}, found {value.shape}"
                    )

                if not np.all(np.isfinite(value)):
                    raise ValueError(
                        f"Checkpoint parameter {name} contains "
                        "non-finite values"
                    )

                loaded_params[name] = value.astype(
                    np.float32, copy=True
                )

            loaded_m = {}
            loaded_v = {}

            for name, current in self.params.items():
                m_key = f"adam_m_{name}"
                v_key = f"adam_v_{name}"

                has_m = m_key in state
                has_v = v_key in state

                if has_m != has_v:
                    raise ValueError(
                        f"Incomplete Adam state for parameter {name}"
                    )

                if has_m:
                    m_value = state[m_key]
                    v_value = state[v_key]

                    if (
                        m_value.shape != current.shape
                        or v_value.shape != current.shape
                    ):
                        raise ValueError(
                            f"Adam state shape mismatch for {name}"
                        )

                    if not (
                        np.all(np.isfinite(m_value))
                        and np.all(np.isfinite(v_value))
                    ):
                        raise ValueError(
                            f"Non-finite Adam state for {name}"
                        )

                    loaded_m[name] = m_value.astype(
                        np.float32, copy=True
                    )
                    loaded_v[name] = v_value.astype(
                        np.float32, copy=True
                    )
                else:
                    loaded_m[name] = np.zeros_like(current)
                    loaded_v[name] = np.zeros_like(current)

            loaded_step = int(state["optimizer_step"])
            loaded_epoch = (
                int(state["saved_epoch"])
                if "saved_epoch" in state
                else 0
            )

            if loaded_step < 0 or loaded_epoch < 0:
                raise ValueError(
                    "Checkpoint has invalid training counters"
                )

        self.params = loaded_params
        self.m = loaded_m
        self.v = loaded_v
        self.step = loaded_step
        self.epoch = loaded_epoch

    def forward(self, tokens):
        tokens = np.asarray(tokens, dtype=np.int64)

        if tokens.ndim != 2:
            raise ValueError("Input tokens must have shape (batch, time)")

        batch_size, time_steps = tokens.shape

        if time_steps < 1 or time_steps > self.sequence_length:
            raise ValueError("Input length is outside the supported range")

        if np.any(tokens < 0) or np.any(tokens >= self.vocab_size):
            raise ValueError("Input contains an invalid token ID")

        p = self.params
        hidden = p["E"][tokens] + p["P"][:time_steps][None, :, :]

        query = hidden @ p["Wq"]
        key = hidden @ p["Wk"]
        value = hidden @ p["Wv"]

        scores = query @ key.transpose(0, 2, 1)
        scores /= np.sqrt(self.d_model)

        causal_mask = np.triu(
            np.ones((time_steps, time_steps), dtype=bool),
            k=1,
        )
        scores = np.where(causal_mask[None, :, :], -1e9, scores)

        scores -= np.max(scores, axis=-1, keepdims=True)
        attention = np.exp(scores)
        attention /= np.sum(attention, axis=-1, keepdims=True)

        attended = attention @ value
        attention_output = attended @ p["Wo"]
        residual = hidden + attention_output

        ff_pre = residual @ p["W1"] + p["b1"]
        ff_activated = np.maximum(ff_pre, 0)
        ff_output = ff_activated @ p["W2"] + p["b2"]

        final_hidden = residual + ff_output
        logits = final_hidden @ p["Wout"] + p["bout"]

        cache = {
            "tokens": tokens,
            "hidden": hidden,
            "query": query,
            "key": key,
            "value": value,
            "attention": attention,
            "attended": attended,
            "residual": residual,
            "ff_pre": ff_pre,
            "ff_activated": ff_activated,
            "final_hidden": final_hidden,
        }

        return logits, cache

    def loss_and_gradients(self, inputs, targets):
        inputs = np.asarray(inputs, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)

        if inputs.shape != targets.shape:
            raise ValueError("Inputs and targets must have matching shapes")

        logits, cache = self.forward(inputs)
        batch_size, time_steps, vocab_size = logits.shape

        if targets.shape != (batch_size, time_steps):
            raise ValueError("Targets have an incompatible shape")

        if np.any(targets < 0) or np.any(targets >= vocab_size):
            raise ValueError("Targets contain an invalid token ID")

        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probabilities = exp_logits / np.sum(
            exp_logits, axis=-1, keepdims=True
        )

        rows = np.arange(batch_size)[:, None]
        times = np.arange(time_steps)[None, :]
        chosen = probabilities[rows, times, targets]

        loss = -np.mean(np.log(np.maximum(chosen, 1e-12)))

        dlogits = probabilities.copy()
        dlogits[rows, times, targets] -= 1.0
        dlogits /= batch_size * time_steps

        p = self.params
        gradients = {
            name: np.zeros_like(value)
            for name, value in p.items()
        }

        gradients["Wout"] = (
            cache["final_hidden"].reshape(-1, self.d_model).T
            @ dlogits.reshape(-1, vocab_size)
        )
        gradients["bout"] = np.sum(dlogits, axis=(0, 1))

        d_final = dlogits @ p["Wout"].T

        gradients["W2"] = (
            cache["ff_activated"].reshape(-1, self.ff_dim).T
            @ d_final.reshape(-1, self.d_model)
        )
        gradients["b2"] = np.sum(d_final, axis=(0, 1))

        d_ff_activated = d_final @ p["W2"].T
        d_ff_pre = d_ff_activated * (cache["ff_pre"] > 0)

        gradients["W1"] = (
            cache["residual"].reshape(-1, self.d_model).T
            @ d_ff_pre.reshape(-1, self.ff_dim)
        )
        gradients["b1"] = np.sum(d_ff_pre, axis=(0, 1))

        # The residual connection carries both gradient paths.
        d_residual = d_final + d_ff_pre @ p["W1"].T

        # FIX: attention receives the complete residual gradient.
        gradients["Wo"] = (
            cache["attended"].reshape(-1, self.d_model).T
            @ d_residual.reshape(-1, self.d_model)
        )
        d_attended = d_residual @ p["Wo"].T

        d_attention = (
            d_attended @ cache["value"].transpose(0, 2, 1)
        )
        d_value = (
            cache["attention"].transpose(0, 2, 1) @ d_attended
        )

        attention = cache["attention"]
        d_scores = attention * (
            d_attention
            - np.sum(
                d_attention * attention,
                axis=-1,
                keepdims=True,
            )
        )

        causal_mask = np.triu(
            np.ones((time_steps, time_steps), dtype=bool),
            k=1,
        )
        d_scores = np.where(causal_mask[None, :, :], 0.0, d_scores)

        scale = np.sqrt(self.d_model)
        d_query = d_scores @ cache["key"] / scale
        d_key = d_scores.transpose(0, 2, 1) @ cache["query"] / scale

        hidden_flat = cache["hidden"].reshape(-1, self.d_model)

        gradients["Wq"] = (
            hidden_flat.T @ d_query.reshape(-1, self.d_model)
        )
        gradients["Wk"] = (
            hidden_flat.T @ d_key.reshape(-1, self.d_model)
        )
        gradients["Wv"] = (
            hidden_flat.T @ d_value.reshape(-1, self.d_model)
        )

        d_hidden = (
            d_residual
            + d_query @ p["Wq"].T
            + d_key @ p["Wk"].T
            + d_value @ p["Wv"].T
        )

        gradients["P"][:time_steps] = np.sum(d_hidden, axis=0)

        for batch_index in range(batch_size):
            np.add.at(
                gradients["E"],
                cache["tokens"][batch_index],
                d_hidden[batch_index],
            )

        return float(loss), gradients

    def train_batch(self, inputs, targets):
        loss, gradients = self.loss_and_gradients(inputs, targets)

        if not np.isfinite(loss):
            raise FloatingPointError("Training loss is not finite")

        global_norm = np.sqrt(
            sum(
                float(np.sum(gradient.astype(np.float64) ** 2))
                for gradient in gradients.values()
            )
        )

        if not np.isfinite(global_norm):
            raise FloatingPointError("Gradient norm is not finite")

        clip_scale = min(1.0, 1.0 / (global_norm + 1e-12))
        self.step += 1

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        for name, parameter in self.params.items():
            gradient = gradients[name] * clip_scale

            self.m[name] = (
                beta1 * self.m[name] + (1 - beta1) * gradient
            )
            self.v[name] = (
                beta2 * self.v[name] + (1 - beta2) * gradient**2
            )

            m_hat = self.m[name] / (1 - beta1**self.step)
            v_hat = self.v[name] / (1 - beta2**self.step)

            parameter -= self.learning_rate * m_hat / (
                np.sqrt(v_hat) + epsilon
            )

        return loss

    def generate(
        self,
        prompt_ids,
        max_new_tokens=20,
        temperature=1.0,
        rng=None,
    ):
        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero")

        rng = rng or np.random.default_rng()
        tokens = list(map(int, prompt_ids))

        if not tokens:
            raise ValueError("Prompt must contain at least one token")

        for _ in range(max_new_tokens):
            context = tokens[-self.sequence_length:]
            inputs = np.asarray([context], dtype=np.int64)
            logits, _ = self.forward(inputs)

            next_logits = logits[0, -1].astype(np.float64)
            next_logits /= temperature
            next_logits -= np.max(next_logits)

            probabilities = np.exp(next_logits)
            probabilities /= probabilities.sum()

            tokens.append(
                int(rng.choice(self.vocab_size, p=probabilities))
            )

        return tokens


def main():
    from dataset import load_dataset

    sequence_length = 8
    checkpoint_path = Path("checkpoints/transformer_weights.npz")

    inputs, targets, tokenizer = load_dataset(sequence_length)
    model = TransformerLM(
        len(tokenizer.token_to_id),
        sequence_length=sequence_length,
    )

    if checkpoint_path.exists():
        model.load(checkpoint_path)
        print(f"Restored checkpoint at epoch {model.epoch}")
        print(f"Restored optimizer step: {model.step}")
    else:
        print("No checkpoint found. Initializing from scratch.")

    total_epochs = 200
    start_epoch = model.epoch

    for epoch_index in range(start_epoch, start_epoch + total_epochs):
        order = np.random.permutation(len(inputs))
        batch_losses = []

        for start in range(0, len(order), 8):
            indices = order[start:start + 8]
            batch_losses.append(
                model.train_batch(inputs[indices], targets[indices])
            )

        model.epoch = epoch_index + 1
        mean_loss = float(np.mean(batch_losses))

        if model.epoch % 20 == 0 or model.epoch == start_epoch + 1:
            print(
                f"Epoch {model.epoch} | loss: {mean_loss:.4f} "
                f"| optimizer step: {model.step}"
            )
            model.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    model.save(checkpoint_path)
    print("Training complete.")


if __name__ == "__main__":
    main()