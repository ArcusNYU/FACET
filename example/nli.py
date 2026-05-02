import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class SinglePersonCaptionNLI:
    def __init__(
        self,
        model_dir: str,
        onnx_filename: str = "model.onnx",
        provider: str = "cuda",
        max_length: int = 256,
        positive_threshold: float = 0.65,
        negative_threshold: float = 0.40,
        margin_threshold: float = 0.25,
    ):
        self.model_dir = model_dir
        self.onnx_path = f"{model_dir.rstrip('/')}/{onnx_filename}"
        self.max_length = max_length

        # 认定结果为True的条件

        # 正向分数至少要这么高
        self.positive_threshold = positive_threshold

        # 反向分数不能超过这个值
        self.negative_threshold = negative_threshold

        # 正向分数要比反向分数至少高这么多
        self.margin_threshold = margin_threshold

        self.label_mapping = ["contradiction", "entailment", "neutral"]
        self.contradiction_id = 0
        self.entailment_id = 1
        self.neutral_id = 2

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if provider == "cuda":
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = [
                "CPUExecutionProvider",
            ]

        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=providers,
        )

        self.input_names = [x.name for x in self.session.get_inputs()]

        self.positive_hypotheses = [
            "The caption describes one person.",
            "The caption is about a single person.",
            "The caption focuses on one individual.",
        ]

        self.negative_hypotheses = [
            "The caption describes multiple people.",
            "The caption says there is a second person.",
            "The caption says the person is accompanied by someone else.",
        ]

    @staticmethod
    def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        logits = logits - np.max(logits, axis=axis, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def predict_proba(self, caption: str, hypothesis: str) -> dict:
        encoded = self.tokenizer(
            caption,
            hypothesis,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        ort_inputs = {
            name: encoded[name]
            for name in self.input_names
            if name in encoded
        }

        logits = self.session.run(None, ort_inputs)[0]
        probs = self._softmax(logits, axis=-1)[0]

        return {
            "contradiction": float(probs[self.contradiction_id]),
            "entailment": float(probs[self.entailment_id]),
            "neutral": float(probs[self.neutral_id]),
        }

    def predict_hypotheses(self, caption: str, hypotheses: list[str]) -> list[dict]:
        """
        一次性跑多个 hypothesis，比逐条跑更快。
        """
        captions = [caption] * len(hypotheses)

        encoded = self.tokenizer(
            captions,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        ort_inputs = {
            name: encoded[name]
            for name in self.input_names
            if name in encoded
        }

        logits = self.session.run(None, ort_inputs)[0]
        probs = self._softmax(logits, axis=-1)

        results = []
        for p in probs:
            results.append({
                "contradiction": float(p[self.contradiction_id]),
                "entailment": float(p[self.entailment_id]),
                "neutral": float(p[self.neutral_id]),
            })

        return results

    def is_single_person_video_caption(
        self,
        caption: str,
        return_debug: bool = False,
    ):
        pos_probs = self.predict_hypotheses(
            caption,
            self.positive_hypotheses,
        )

        neg_probs = self.predict_hypotheses(
            caption,
            self.negative_hypotheses,
        )

        pos_entailments = [p["entailment"] for p in pos_probs]
        neg_entailments = [p["entailment"] for p in neg_probs]

        positive_score = max(pos_entailments)
        negative_score = max(neg_entailments)

        result = (
            positive_score >= self.positive_threshold
            and negative_score <= self.negative_threshold
            and positive_score - negative_score >= self.margin_threshold
        )

        if return_debug:
            return bool(result), {
                "positive_score": positive_score,
                "negative_score": negative_score,
                "positive_hypotheses": list(zip(self.positive_hypotheses, pos_entailments)),
                "negative_hypotheses": list(zip(self.negative_hypotheses, neg_entailments)),
            }

        return bool(result)


if __name__ == "__main__":
    clf = SinglePersonCaptionNLI(
        model_dir="./weights/NLI",
        onnx_filename="model.onnx",
        provider="cuda",
        max_length=256,

        # 保守一点：宁愿 false negative，不要 false positive
        positive_threshold=0.65,
        negative_threshold=0.40,
        margin_threshold=0.25,
    )

    examples = [
        "The video shows two men, both with dark hair, seated in a vehicle at night, engaged in a conversation or observation as they drive through a dimly lit, indistinct environment. The man on the left wears a light-colored uniform with insignia, while the man on the right wears a light-colored shirt, both with relaxed yet attentive postures.",
    ]

    for text in examples:
        result, debug = clf.is_single_person_video_caption(
            text,
            return_debug=True,
        )

        print(text)
        print("single_person:", result)
        print("positive_score:", debug["positive_score"])
        print("negative_score:", debug["negative_score"])
        print("negative details:")
        for h, s in debug["negative_hypotheses"]:
            print(f"  {s:.4f} | {h}")
        print()