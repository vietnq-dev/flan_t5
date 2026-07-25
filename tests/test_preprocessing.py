from datasets import Dataset

from src.data.preprocessing import preprocess_boolq_example, preprocess_squad_example, tokenize_dataset


class DummyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts: list[str],
        max_length: int,
        padding: str,
        truncation: bool,
    ) -> dict[str, list[list[int]]]:
        del padding, truncation
        input_ids = []
        attention_mask = []
        for text in texts:
            ids = [ord(char) for char in text[:max_length]]
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
            input_ids.append(ids)
            attention_mask.append([1 if token_id != self.pad_token_id else 0 for token_id in ids])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in ids if token_id != self.pad_token_id)


def test_preprocess_boolq_example_formats_source_and_target() -> None:
    example = {
        "question": "is the sky blue",
        "passage": "The sky often appears blue during the day.",
        "answer": True,
    }

    result = preprocess_boolq_example(example)

    assert result["input_text"] == (
        "question: is the sky blue\npassage: The sky often appears blue during the day."
    )
    assert result["target_text"] == "yes"


def test_tokenize_dataset_handles_boolq_columns() -> None:
    dataset = Dataset.from_dict(
        {
            "question": ["is water dry"],
            "passage": ["Water is wet."],
            "answer": [False],
        }
    )
    tokenizer = DummyTokenizer()

    tokenized = tokenize_dataset(
        dataset,
        tokenizer,
        max_source_length=32,
        max_target_length=8,
        num_proc=1,
    )

    decoded_label = tokenizer.decode(
        [token_id for token_id in tokenized[0]["labels"] if token_id != -100],
        skip_special_tokens=True,
    )
    assert decoded_label == "no"


def test_preprocess_squad_example_uses_context_and_first_answer() -> None:
    example = {
        "question": "Where was Tesla born?",
        "context": "Nikola Tesla was born in Smiljan.",
        "answers": {"text": ["Smiljan"], "answer_start": [27]},
    }

    result = preprocess_squad_example(example)

    assert result["input_text"] == (
        "question: Where was Tesla born?\ncontext: Nikola Tesla was born in Smiljan."
    )
    assert result["target_text"] == "Smiljan"


def test_tokenize_dataset_handles_squad_columns() -> None:
    dataset = Dataset.from_dict(
        {
            "id": ["1"],
            "title": ["Tesla"],
            "question": ["Where was Tesla born?"],
            "context": ["Nikola Tesla was born in Smiljan."],
            "answers": [{"text": ["Smiljan"], "answer_start": [27]}],
        }
    )
    tokenizer = DummyTokenizer()

    tokenized = tokenize_dataset(
        dataset,
        tokenizer,
        max_source_length=64,
        max_target_length=16,
        num_proc=1,
    )

    decoded_label = tokenizer.decode(
        [token_id for token_id in tokenized[0]["labels"] if token_id != -100],
        skip_special_tokens=True,
    )
    assert decoded_label == "Smiljan"
