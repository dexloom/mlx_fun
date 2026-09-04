"""Q&A-driven expert relevance probing for a domain.

Where ``domain-scan`` compares routing over raw corpora, this module asks the
model a curated set of questions and watches which experts it uses while it
*answers* them. Two passes:

1. **Trace** — run each question (teacher-forced against a reference answer, or
   with the model generating its own) under the observer hooks and record, per
   question, the normalized expert frequency and routed weight over the answer
   positions. Domain questions minus general questions gives a differential
   score, exactly as ``safety.compute_differential_scores`` does, but weighted
   per question instead of per token so one long answer cannot dominate.

2. **Knockout** — mask a candidate expert out of the *real* router and measure
   how much the answer log-likelihood degrades. The mask is applied to the gate
   parameter that enters the top-k selection score, so grouped selection,
   ``norm_topk_prob``, ``routed_scaling_factor`` and any latent projections all
   run untouched: no surrogate router, no hooks during scoring.

The report is a superset of ``DomainReport``, so ``prune --domain-map``,
``amplify``, ``serve --domain-map`` and the steering API consume it unchanged.
"""

import json
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .domain import DomainReport
from .saliency import SaliencyAccumulator

DOMAIN = "domain"
GENERAL = "general"
_LABELS = (DOMAIN, GENERAL)


# ---------------------------------------------------------------------------
# Question sets
# ---------------------------------------------------------------------------

@dataclass
class ProbeQuestion:
    """One probe item: a question and (for teacher mode) its reference answer."""

    question: str
    answer: Optional[str] = None
    system: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ProbeExample:
    """A chat-templated question plus the answer tokens scored against it."""

    tokens: List[int]
    prompt_len: int
    question_index: int

    @property
    def n_answer(self) -> int:
        return len(self.tokens) - self.prompt_len


def load_probe_set(path: str, max_questions: int = 0) -> List[ProbeQuestion]:
    """Load a probe question set from JSONL.

    Each line is an object with a required ``question`` and optional ``answer``
    (required for teacher mode), ``system`` and ``tags``.

    Args:
        path: Path to the JSONL file.
        max_questions: If > 0 and the file holds more, randomly subsample this
            many (seed with ``random.seed`` for reproducibility).

    Returns:
        List of ProbeQuestion.

    Raises:
        ValueError: On a malformed line or a missing/empty question.
    """
    questions: List[ProbeQuestion] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")

            question = obj.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"{path}:{line_no}: 'question' must be a non-empty string"
                )

            answer = obj.get("answer")
            if answer is not None and not isinstance(answer, str):
                raise ValueError(f"{path}:{line_no}: 'answer' must be a string")

            system = obj.get("system")
            if system is not None and not isinstance(system, str):
                raise ValueError(f"{path}:{line_no}: 'system' must be a string")

            tags = obj.get("tags", [])
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                raise ValueError(f"{path}:{line_no}: 'tags' must be a list of strings")

            questions.append(
                ProbeQuestion(
                    question=question, answer=answer, system=system, tags=list(tags),
                )
            )

    if max_questions > 0 and len(questions) > max_questions:
        questions = random.sample(questions, max_questions)
    return questions


def _has_chat_template(tokenizer) -> bool:
    """Whether the tokenizer can apply a chat template."""
    flag = getattr(tokenizer, "has_chat_template", None)
    if flag is not None:
        return bool(flag)
    return getattr(tokenizer, "chat_template", None) is not None


def _template_ids_to_list(prompt) -> List[int]:
    """Normalize an ``apply_chat_template(tokenize=True)`` result to flat ids.

    Tokenizer backends disagree on the return shape: mlx-lm's
    ``TokenizerWrapper`` yields a flat ``list[int]``, but transformers' newer
    ``TokenizersBackend`` (GLM-5.3-Flash, Qwen4-Exp and other recent VLMs)
    returns a ``BatchEncoding`` keyed by ``input_ids``, and some return a raw
    ``Encoding``. Left unhandled, ``list(prompt)`` yields ``Encoding`` objects
    or dict keys instead of token ids. Everything collapses to a single flat
    ``list[int]`` here; an already-flat list passes through unchanged.
    """
    # dict / BatchEncoding: take the ids field.
    if isinstance(prompt, dict) or (hasattr(prompt, "keys") and "input_ids" in prompt):
        prompt = prompt["input_ids"]
    # A bare transformers Encoding.
    if hasattr(prompt, "ids") and not isinstance(prompt, (list, tuple)):
        prompt = prompt.ids
    seq = list(prompt)
    # A list wrapping a single Encoding, or a batched [[...]] of one row.
    if len(seq) == 1 and hasattr(seq[0], "ids"):
        seq = list(seq[0].ids)
    elif seq and isinstance(seq[0], (list, tuple)):
        seq = list(seq[0])
    return [int(t) for t in seq]


def build_probe_tokens(
    tokenizer,
    q: ProbeQuestion,
    answer_mode: str = "teacher",
    max_answer_tokens: int = 128,
    chat_template_args: Optional[dict] = None,
    system: Optional[str] = None,
) -> Tuple[List[int], int]:
    """Build the token sequence for one probe question.

    The prompt ends at the assistant generation header; in teacher mode the
    reference answer is appended so a single forward pass scores it.

    Args:
        tokenizer: The model tokenizer (mlx-lm TokenizerWrapper or similar).
        q: The question.
        answer_mode: "teacher" (append the reference answer) or "generate".
        max_answer_tokens: Truncate the reference answer to this many tokens.
        chat_template_args: Extra kwargs for apply_chat_template, e.g.
            ``{"enable_thinking": False}``.
        system: Default system prompt; ``q.system`` overrides it.

    Returns:
        (tokens, prompt_len). In generate mode tokens == the prompt.

    Raises:
        ValueError: If teacher mode gets a question with no usable answer.
    """
    sys_prompt = q.system if q.system is not None else system
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": q.question})

    if _has_chat_template(tokenizer):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **(chat_template_args or {}),
        )
        prompt = _template_ids_to_list(prompt)
    else:
        text = q.question + "\n"
        if sys_prompt:
            text = f"{sys_prompt}\n\n{text}"
        prompt = [int(t) for t in tokenizer.encode(text)]

    if answer_mode == "generate":
        return prompt, len(prompt)
    if answer_mode != "teacher":
        raise ValueError(f"Unknown answer_mode '{answer_mode}'. Use: teacher, generate")

    if not q.answer or not q.answer.strip():
        raise ValueError("teacher mode requires a non-empty 'answer'")
    answer_ids = list(tokenizer.encode(q.answer, add_special_tokens=False))
    if max_answer_tokens > 0:
        answer_ids = answer_ids[:max_answer_tokens]
    if not answer_ids:
        raise ValueError("reference answer encoded to zero tokens")
    return prompt + answer_ids, len(prompt)


# ---------------------------------------------------------------------------
# Capture slicing and answer likelihood
# ---------------------------------------------------------------------------

def slice_answer_captures(
    block_captures: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    prompt_len: int,
    n_answer: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice one block's captures down to the answer-producing positions.

    Logits at position ``t`` predict token ``t+1``, so the positions that
    produce the ``n_answer`` scored predictions are ``[prompt_len-1 :
    prompt_len-1+n_answer]``. Attributing routing there — rather than to the
    positions that *consume* answer tokens — keeps the observational score and
    the knockout delta describing the same predictions, and includes the
    routing that picks the first answer token.

    A forward pass may arrive as several captures (mlx-lm prefills in chunks and
    then decodes one token at a time), so they are concatenated along the
    sequence axis first.

    Args:
        block_captures: The ``(inds, scores, norms)`` tuples for one MoE block,
            each shaped (batch, seq, top_k).
        prompt_len: Number of prompt tokens.
        n_answer: Number of answer tokens scored.

    Returns:
        (inds, scores, norms), each (n_answer, top_k).

    Raises:
        ValueError: If the captures do not cover exactly prompt_len + n_answer
            positions — a loud failure beats silent misalignment.
    """
    if not block_captures:
        raise ValueError("no captures for this block; were hooks installed?")

    def _cat(idx: int) -> np.ndarray:
        parts = [c[idx].reshape(-1, c[idx].shape[-1]) for c in block_captures]
        return np.concatenate(parts, axis=0)

    inds, scores, norms = _cat(0), _cat(1), _cat(2)
    total = inds.shape[0]
    expected = prompt_len + n_answer
    if total != expected:
        raise ValueError(
            f"captured {total} positions, expected {expected} "
            f"(prompt_len={prompt_len}, n_answer={n_answer})"
        )
    sl = slice(prompt_len - 1, prompt_len - 1 + n_answer)
    return inds[sl], scores[sl], norms[sl]


def answer_nll(logits: mx.array, tokens: Sequence[int], prompt_len: int) -> float:
    """Mean negative log-likelihood of the answer tokens, in nats per token.

    Only the answer rows are upcast to float32 — a full-sequence log_softmax
    over a large vocabulary would be needlessly expensive.

    Args:
        logits: (1, T, V) logits from a forward pass over ``tokens``.
        tokens: The full token sequence (prompt + answer).
        prompt_len: Number of prompt tokens.

    Returns:
        Mean NLL. May be non-finite if the model produced non-finite logits.
    """
    total = len(tokens)
    n_answer = total - prompt_len
    if n_answer <= 0:
        raise ValueError("no answer tokens to score")

    sliced = logits[0, prompt_len - 1 : total - 1, :].astype(mx.float32)
    targets = mx.array(np.asarray(tokens[prompt_len:], dtype=np.int32))
    loss = mx.mean(nn.losses.cross_entropy(sliced, targets))
    mx.eval(loss)
    return float(loss.item())


# ---------------------------------------------------------------------------
# Question-weighted routing statistics
# ---------------------------------------------------------------------------

def question_vectors(
    sliced_per_layer: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_experts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-question routing vectors, normalized by answer length.

    Args:
        sliced_per_layer: Per MoE layer, the ``(inds, scores, norms)`` arrays of
            shape (n_answer, top_k) from :func:`slice_answer_captures`.
        num_experts: Number of routed experts.

    Returns:
        (freq, weight), each (num_layers, num_experts) float64. ``freq[l, e]``
        is the fraction of answer positions routing to expert ``e``; each row
        sums to top_k. ``weight[l, e]`` is its mean routed weight per position.
    """
    num_layers = len(sliced_per_layer)
    freq = np.zeros((num_layers, num_experts), dtype=np.float64)
    weight = np.zeros((num_layers, num_experts), dtype=np.float64)

    for layer_idx, (inds, scores, _norms) in enumerate(sliced_per_layer):
        n_positions = inds.shape[0]
        if n_positions == 0:
            continue
        flat_inds = inds.ravel().astype(np.intp)
        np.add.at(freq[layer_idx], flat_inds, 1.0)
        np.add.at(weight[layer_idx], flat_inds, scores.ravel().astype(np.float64))
        freq[layer_idx] /= n_positions
        weight[layer_idx] /= n_positions

    return freq, weight


class ProbeStats:
    """Question-weighted routing statistics for the domain and general sets.

    Every question contributes one normalized vector regardless of how long its
    answer is, so a single verbose answer cannot dominate the scores. Raw
    token-level saliency is accumulated separately for the ``.npz`` that
    ``prune --saliency`` consumes.
    """

    def __init__(self, num_layers: int, num_experts: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        shape = (num_layers, num_experts)
        self.freq_sum = {label: np.zeros(shape, dtype=np.float64) for label in _LABELS}
        self.weight_sum = {label: np.zeros(shape, dtype=np.float64) for label in _LABELS}
        self.coverage = {label: np.zeros(shape, dtype=np.float64) for label in _LABELS}
        self.n_questions = {label: 0 for label in _LABELS}
        self.saliency = SaliencyAccumulator(num_layers, num_experts)

    def add_question(
        self,
        label: str,
        sliced_per_layer: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        saliency_weighting: str = "question",
    ) -> None:
        """Fold one question's answer routing into the statistics.

        Args:
            label: "domain" or "general".
            sliced_per_layer: Per-layer (inds, scores, norms) over answer positions.
            saliency_weighting: "question" divides each question's saliency
                contribution by its answer length so long answers do not
                dominate the ``.npz``; "token" keeps raw counts.
        """
        if label not in _LABELS:
            raise ValueError(f"Unknown label '{label}'. Use: {', '.join(_LABELS)}")

        freq, weight = question_vectors(sliced_per_layer, self.num_experts)
        self.freq_sum[label] += freq
        self.weight_sum[label] += weight
        self.coverage[label] += (freq > 0).astype(np.float64)
        self.n_questions[label] += 1

        if label != DOMAIN:
            return

        for layer_idx, (inds, scores, norms) in enumerate(sliced_per_layer):
            n_positions = inds.shape[0]
            if n_positions == 0:
                continue
            if saliency_weighting == "question":
                # One question's worth of saliency, however long its answer.
                per_question = SaliencyAccumulator(1, self.num_experts)
                per_question.update(0, inds, scores, norms)
                self.saliency.reap_sum[layer_idx] += per_question.reap_sum[0] / n_positions
                self.saliency.reap_count[layer_idx] += per_question.reap_count[0] / n_positions
                self.saliency.ean_sum[layer_idx] += per_question.ean_sum[0] / n_positions
                self.saliency.freq[layer_idx] += per_question.freq[0] / n_positions
                self.saliency.weighted_freq_sum[layer_idx] += (
                    per_question.weighted_freq_sum[0] / n_positions
                )
            elif saliency_weighting == "token":
                self.saliency.update(layer_idx, inds, scores, norms)
            else:
                raise ValueError(
                    f"Unknown saliency_weighting '{saliency_weighting}'. "
                    f"Use: question, token"
                )

    def _mean(self, sums: Dict[str, np.ndarray], label: str) -> np.ndarray:
        n = self.n_questions[label]
        if n == 0:
            return np.zeros((self.num_layers, self.num_experts), dtype=np.float64)
        return sums[label] / n

    def mean_freq(self, label: str) -> np.ndarray:
        """Mean per-question selection frequency, (num_layers, num_experts)."""
        return self._mean(self.freq_sum, label)

    def mean_weight(self, label: str) -> np.ndarray:
        """Mean per-question routed weight, (num_layers, num_experts)."""
        return self._mean(self.weight_sum, label)

    def coverage_fraction(self, label: str) -> np.ndarray:
        """Fraction of that label's questions in which each expert was used."""
        return self._mean(self.coverage, label)


def compute_probe_scores(
    stats: ProbeStats,
    freq_weight: float = 0.5,
    weight_weight: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Differential scores between the domain and general question sets.

    Args:
        stats: Populated ProbeStats.
        freq_weight: Composite weight for the normalized frequency differential.
        weight_weight: Composite weight for the normalized routed-weight differential.

    Returns:
        (diff_freq, diff_weight, composite), each (num_layers, num_experts).
        Positive differential = domain-preferred. Composite is per-layer
        min-max normalized into [0, 1], as in ``safety.compute_differential_scores``.
    """
    from .safety import _layer_normalize

    diff_freq = stats.mean_freq(DOMAIN) - stats.mean_freq(GENERAL)
    diff_weight = stats.mean_weight(DOMAIN) - stats.mean_weight(GENERAL)
    composite = (
        freq_weight * _layer_normalize(diff_freq)
        + weight_weight * _layer_normalize(diff_weight)
    )
    return diff_freq, diff_weight, composite


def apply_coverage_filter(
    domain_experts: Dict[int, List[int]],
    coverage: np.ndarray,
    min_coverage: float,
) -> Dict[int, List[int]]:
    """Drop domain experts that fired in too few questions.

    A high composite score built from one or two questions is noise; this keeps
    only experts the domain set uses consistently.

    Args:
        domain_experts: layer_idx -> expert IDs.
        coverage: (num_layers, num_experts) fraction of questions per expert.
        min_coverage: Minimum fraction, in [0, 1]. 0 disables the filter.

    Returns:
        Filtered dict; layers left empty are dropped.
    """
    if min_coverage <= 0.0:
        return {k: list(v) for k, v in domain_experts.items()}

    filtered: Dict[int, List[int]] = {}
    for layer_idx, expert_ids in domain_experts.items():
        kept = [e for e in expert_ids if coverage[layer_idx, e] >= min_coverage]
        if kept:
            filtered[layer_idx] = kept
    return filtered


def select_knockout_candidates(
    composite: np.ndarray,
    domain_experts: Dict[int, List[int]],
    n: int,
) -> List[Tuple[int, int]]:
    """Pick the highest-scoring (layer, expert) pairs to verify by knockout.

    Args:
        composite: (num_layers, num_experts) composite scores.
        domain_experts: layer_idx -> expert IDs to choose from.
        n: How many pairs to return. Clipped to what is available; 0 returns [].

    Returns:
        Pairs sorted by descending composite score.
    """
    if n <= 0:
        return []
    pairs = [
        (layer_idx, expert_id)
        for layer_idx, expert_ids in domain_experts.items()
        for expert_id in expert_ids
    ]
    pairs.sort(key=lambda p: (-composite[p[0], p[1]], p[0], p[1]))
    return pairs[:n]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ProbeReport(DomainReport):
    """A DomainReport plus everything the probe adds.

    The base keys are written exactly as ``DomainReport.save`` writes them, so
    ``prune --domain-map``, ``amplify``, ``serve --domain-map`` and the steering
    API load this file unchanged and simply ignore the extras.
    """

    answer_mode: str = "teacher"
    scoring: str = "question_weighted"
    activation_metric: str = "routed_weight"
    saliency_weighting: str = "question"
    num_domain_questions: int = 0
    num_general_questions: int = 0
    skipped_questions: Dict[str, int] = field(default_factory=dict)
    min_coverage: float = 0.0
    domain_mean_freq: Optional[np.ndarray] = None
    general_mean_freq: Optional[np.ndarray] = None
    domain_coverage: Optional[np.ndarray] = None
    general_coverage: Optional[np.ndarray] = None
    knockout: Optional[dict] = None
    knockout_delta: Optional[np.ndarray] = None
    verified_domain_experts: Dict[int, List[int]] = field(default_factory=dict)
    prune_check: Optional[dict] = None

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "answer_mode": self.answer_mode,
            "scoring": self.scoring,
            "activation_metric": self.activation_metric,
            "saliency_weighting": self.saliency_weighting,
            "num_domain_questions": self.num_domain_questions,
            "num_general_questions": self.num_general_questions,
            "skipped_questions": dict(self.skipped_questions),
            "min_coverage": self.min_coverage,
            "domain_mean_freq": _to_list(self.domain_mean_freq),
            "general_mean_freq": _to_list(self.general_mean_freq),
            "domain_coverage": _to_list(self.domain_coverage),
            "general_coverage": _to_list(self.general_coverage),
            "knockout": self.knockout,
            "knockout_delta": _to_list(self.knockout_delta),
            "verified_domain_experts": {
                str(k): v for k, v in self.verified_domain_experts.items()
            },
            "prune_check": self.prune_check,
        })
        return data

    @classmethod
    def load(cls, path: str) -> "ProbeReport":
        """Load a probe report, tolerating a plain DomainReport JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            domain_name=data["domain_name"],
            num_layers=data["num_layers"],
            num_experts=data["num_experts"],
            threshold_percentile=data["threshold_percentile"],
            differential_freq=np.array(data["differential_freq"]),
            differential_activation=np.array(data["differential_activation"]),
            composite_score=np.array(data["composite_score"]),
            domain_experts={int(k): v for k, v in data["domain_experts"].items()},
            general_experts={int(k): v for k, v in data["general_experts"].items()},
            answer_mode=data.get("answer_mode", "teacher"),
            scoring=data.get("scoring", "question_weighted"),
            activation_metric=data.get("activation_metric", "routed_weight"),
            saliency_weighting=data.get("saliency_weighting", "question"),
            num_domain_questions=data.get("num_domain_questions", 0),
            num_general_questions=data.get("num_general_questions", 0),
            skipped_questions=data.get("skipped_questions", {}),
            min_coverage=data.get("min_coverage", 0.0),
            domain_mean_freq=_from_list(data.get("domain_mean_freq")),
            general_mean_freq=_from_list(data.get("general_mean_freq")),
            domain_coverage=_from_list(data.get("domain_coverage")),
            general_coverage=_from_list(data.get("general_coverage")),
            knockout=data.get("knockout"),
            knockout_delta=_from_list(data.get("knockout_delta")),
            verified_domain_experts={
                int(k): v for k, v in data.get("verified_domain_experts", {}).items()
            },
            prune_check=data.get("prune_check"),
        )


def _to_list(arr: Optional[np.ndarray]):
    return None if arr is None else np.asarray(arr).tolist()


def _from_list(value):
    return None if value is None else np.array(value)


# ---------------------------------------------------------------------------
# Knockout backend — mask the real router's selection score
# ---------------------------------------------------------------------------

# Per model type, where the parameter that enters the top-k selection score
# lives. Masking it leaves the model's own __call__ — grouped selection,
# norm_topk_prob, routed_scaling_factor, latent projections — entirely intact.
_SELECTION_TARGETS = {
    # inds = argpartition(-(sigmoid(gates) + e_score_correction_bias)); the bias
    # sits on the block, and the mask must land AFTER the sigmoid, so biasing
    # the gate's pre-sigmoid logits (what amplify does) would not reliably
    # deselect an expert with a large positive correction.
    "minimax": ("block", "e_score_correction_bias"),
    "minimax_m2": ("block", "e_score_correction_bias"),
    # group_expert_select adds the correction bias before group scoring and top-k.
    "glm4_moe": ("gate", "e_score_correction_bias"),
    "glm4_moe_lite": ("gate", "e_score_correction_bias"),
    "glm_moe_dsa": ("gate", "e_score_correction_bias"),
    "deepseek_v32": ("gate", "e_score_correction_bias"),
    "nemotron_h": ("gate", "e_score_correction_bias"),
    "glm5_next": ("gate", "e_score_correction_bias"),
    # Pre-softmax logits; softmax is monotonic and there is no post-softmax
    # correction, so a large negative bias removes the expert from the top-k.
    "qwen3_moe": ("gate", "bias"),
    "qwen3_next": ("gate", "bias"),
    "qwen4_exp": ("gate", "bias"),
}


def selection_bias_target(block, model_type: str):
    """Resolve the module and attribute that carry a block's selection score.

    Args:
        block: An MoE block.
        model_type: Model type string.

    Returns:
        (module, attr_name) — the parameter to add a mask to.

    Raises:
        ValueError: For a model type with no knockout support.
    """
    target = _SELECTION_TARGETS.get(model_type)
    if target is None:
        raise ValueError(
            f"Knockout not supported for model_type '{model_type}'. "
            f"Supported: {', '.join(sorted(_SELECTION_TARGETS))}"
        )
    owner, attr = target
    module = block if owner == "block" else getattr(block, "gate")
    return module, attr


@contextmanager
def expert_mask(
    moe_blocks: List,
    model_type: str,
    masks: Dict[int, Sequence[int]],
    num_experts: int,
    mask_value: float = -1e9,
):
    """Temporarily make the given experts unselectable by the real router.

    Adds ``mask_value`` to the selection-score parameter of each named expert,
    then restores the original parameter (or removes it, if it did not exist)
    when the block exits — including on an exception.

    Args:
        moe_blocks: MoE blocks indexed by accumulator layer index.
        model_type: Model type string.
        masks: layer_idx -> expert IDs to deactivate. Empty installs the same
            code path with zero bias, which is the right knockout baseline.
        num_experts: Number of routed experts.
        mask_value: Additive bias, large and negative.

    Raises:
        ValueError: For an unsupported model type or an out-of-range expert.
    """
    # (module, attr, original_or_None) — None means the attribute was absent.
    saved: List[Tuple[object, str, Optional[mx.array]]] = []
    try:
        for layer_idx, block in enumerate(moe_blocks):
            module, attr = selection_bias_target(block, model_type)
            expert_ids = masks.get(layer_idx, ())
            original = module[attr] if attr in module else None
            saved.append((module, attr, original))

            bias = np.zeros(num_experts, dtype=np.float64)
            for eid in expert_ids:
                if not 0 <= eid < num_experts:
                    raise ValueError(
                        f"layer {layer_idx}: expert {eid} out of range "
                        f"[0, {num_experts})"
                    )
                bias[eid] = mask_value

            bias_mx = mx.array(bias.astype(np.float32))
            if original is None:
                module[attr] = bias_mx
            else:
                module[attr] = original + bias_mx.astype(original.dtype)
        yield
    finally:
        for module, attr, original in saved:
            if original is None:
                if attr in module:
                    del module[attr]
            else:
                module[attr] = original


# ---------------------------------------------------------------------------
# Answer likelihood over a question set, and paired statistics
# ---------------------------------------------------------------------------

def _forward_logits(forward, tokens: Sequence[int]) -> mx.array:
    """Run one token-only forward pass and return (1, T, V) logits."""
    out = forward(mx.array(np.asarray(tokens, dtype=np.uint32)).reshape(1, -1))
    return getattr(out, "logits", out)


def per_question_nll(forward, examples: Sequence[ProbeExample]) -> np.ndarray:
    """Answer NLL for every example, as a (n,) array with nan where non-finite.

    Args:
        forward: Token-only forward callable (see ``loader.text_forward``).
        examples: The examples to score.

    Returns:
        (len(examples),) float64 array.
    """
    values = np.empty(len(examples), dtype=np.float64)
    for i, ex in enumerate(examples):
        logits = _forward_logits(forward, ex.tokens)
        value = answer_nll(logits, ex.tokens, ex.prompt_len)
        values[i] = value if np.isfinite(value) else np.nan
        del logits
    return values


def paired_delta_stats(
    baseline: np.ndarray,
    masked: np.ndarray,
    min_delta: float = 0.02,
    min_valid_fraction: float = 0.9,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired per-question delta statistics for one knockout.

    Deltas are paired question by question, so a mask that destroys the model on
    half the questions cannot look harmless by being averaged over a different
    subset than the baseline. Masked runs that go non-finite are counted as
    collapses rather than dropped.

    Args:
        baseline: (n,) baseline NLLs (nan = invalid).
        masked: (n,) NLLs under the mask (nan = collapse).
        min_delta: Minimum mean delta, in nats/token, to call an expert verified.
        min_valid_fraction: Below this fraction of usable pairs the result is
            inconclusive (or catastrophic below 0.5).
        n_boot: Bootstrap resamples for the confidence interval. 0 disables it.
        seed: Bootstrap seed.

    Returns:
        Dict with mean_delta, median_delta, ci_low, ci_high, n_total, n_valid,
        n_nonfinite, valid_fraction and status.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    masked = np.asarray(masked, dtype=np.float64)
    if baseline.shape != masked.shape:
        raise ValueError("baseline and masked must have the same shape")

    n_total = int(baseline.size)
    both_finite = np.isfinite(baseline) & np.isfinite(masked)
    n_nonfinite = int(np.sum(np.isfinite(baseline) & ~np.isfinite(masked)))
    deltas = masked[both_finite] - baseline[both_finite]
    n_valid = int(deltas.size)
    valid_fraction = n_valid / n_total if n_total else 0.0

    mean_delta = float(np.mean(deltas)) if n_valid else float("nan")
    median_delta = float(np.median(deltas)) if n_valid else float("nan")

    ci_low = ci_high = float("nan")
    if n_valid and n_boot > 0:
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, n_valid, size=(n_boot, n_valid))
        means = deltas[picks].mean(axis=1)
        ci_low = float(np.percentile(means, 2.5))
        ci_high = float(np.percentile(means, 97.5))

    if valid_fraction < 0.5:
        status = "catastrophic"
    elif valid_fraction < min_valid_fraction:
        status = "inconclusive"
    elif mean_delta >= min_delta and (np.isnan(ci_low) or ci_low > 0):
        status = "verified"
    else:
        status = "not_verified"

    return {
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_nonfinite": n_nonfinite,
        "valid_fraction": valid_fraction,
        "status": status,
    }


def _check_mask_budget(
    masks: Dict[int, Sequence[int]], num_experts: int, top_k: int,
) -> None:
    """Refuse a mask that would leave a layer with fewer experts than top_k."""
    budget = num_experts - top_k
    for layer_idx, expert_ids in masks.items():
        if len(expert_ids) > budget:
            raise ValueError(
                f"layer {layer_idx}: masking {len(expert_ids)} of {num_experts} "
                f"experts leaves fewer than top_k={top_k} selectable. "
                f"At most {budget} can be masked."
            )


@dataclass
class KnockoutResult:
    """Outcome of the per-expert knockout pass."""

    baseline_nll: float
    plain_baseline_nll: float
    per_expert: List[dict] = field(default_factory=list)
    dropped_nonfinite_baseline: int = 0
    num_questions: int = 0


def run_knockout(
    forward,
    examples: Sequence[ProbeExample],
    moe_blocks: List,
    model_type: str,
    num_experts: int,
    top_k: int,
    candidates: Sequence[Tuple[int, int]],
    composite: Optional[np.ndarray] = None,
    coverage: Optional[np.ndarray] = None,
    mask_value: float = -1e9,
    min_delta: float = 0.02,
    min_valid_fraction: float = 0.9,
    n_boot: int = 1000,
    seed: int = 0,
    progress=None,
) -> KnockoutResult:
    """Verify candidate experts by masking them out of the real router.

    The baseline is measured with an all-zero mask installed, so a candidate the
    router never selects yields a delta of exactly zero.

    Args:
        forward: Token-only forward callable.
        examples: Domain examples to score.
        moe_blocks: MoE blocks by accumulator layer index.
        model_type: Model type string.
        num_experts: Routed experts per layer.
        top_k: Experts selected per token (for the mask budget check).
        candidates: (layer, expert) pairs to verify.
        composite: Optional composite scores, recorded per candidate.
        coverage: Optional domain coverage fractions, recorded per candidate.
        mask_value: Additive selection-score mask.
        min_delta, min_valid_fraction, n_boot, seed: See :func:`paired_delta_stats`.
        progress: Optional callable(done, total) for progress reporting.

    Returns:
        KnockoutResult.
    """
    plain = per_question_nll(forward, examples)
    with expert_mask(moe_blocks, model_type, {}, num_experts, mask_value):
        baseline = per_question_nll(forward, examples)

    usable = np.isfinite(baseline)
    dropped = int(np.sum(~usable))
    kept = [ex for ex, ok in zip(examples, usable) if ok]
    baseline_kept = baseline[usable]

    result = KnockoutResult(
        baseline_nll=float(np.mean(baseline_kept)) if baseline_kept.size else float("nan"),
        plain_baseline_nll=float(np.nanmean(plain)) if plain.size else float("nan"),
        dropped_nonfinite_baseline=dropped,
        num_questions=len(kept),
    )
    if not kept:
        return result

    for done, (layer_idx, expert_id) in enumerate(candidates, start=1):
        masks = {layer_idx: [expert_id]}
        _check_mask_budget(masks, num_experts, top_k)
        with expert_mask(moe_blocks, model_type, masks, num_experts, mask_value):
            masked = per_question_nll(forward, kept)
        stats = paired_delta_stats(
            baseline_kept, masked, min_delta, min_valid_fraction, n_boot, seed,
        )
        entry = {"layer": int(layer_idx), "expert": int(expert_id)}
        if composite is not None:
            entry["composite"] = float(composite[layer_idx, expert_id])
        if coverage is not None:
            entry["domain_coverage"] = float(coverage[layer_idx, expert_id])
        entry.update(stats)
        result.per_expert.append(entry)
        if progress is not None:
            progress(done, len(candidates))

    return result


def masks_from_keep_map(
    keep_map: Dict[int, np.ndarray], num_experts: int,
) -> Dict[int, List[int]]:
    """Invert a pruner keep_map into the set of experts it removes."""
    masks = {}
    for layer_idx, keep in keep_map.items():
        removed = np.setdiff1d(np.arange(num_experts), np.asarray(keep, dtype=np.intp))
        if removed.size:
            masks[int(layer_idx)] = [int(e) for e in removed]
    return masks


def run_prune_check(
    forward,
    domain_examples: Sequence[ProbeExample],
    general_examples: Sequence[ProbeExample],
    moe_blocks: List,
    model_type: str,
    num_experts: int,
    top_k: int,
    keep_map: Dict[int, np.ndarray],
    mask_value: float = -1e9,
    min_delta: float = 0.02,
    min_valid_fraction: float = 0.9,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Measure what the exact prune set would cost on both question sets.

    The masked set is the complement of ``keep_map`` — precisely the experts
    ``mlx-fun prune`` would remove for the same arguments. Note this runs the
    original router with those experts unselectable; a real pruned checkpoint
    has a smaller expert axis, so treat this as the closest available stand-in
    rather than proof, and confirm with ``smoke-test`` on the pruned model.

    Returns:
        Dict with masked_pairs and paired stats for the domain and general sets.
    """
    masks = masks_from_keep_map(keep_map, num_experts)
    _check_mask_budget(masks, num_experts, top_k)
    masked_pairs = sum(len(v) for v in masks.values())

    out = {"masked_pairs": masked_pairs}
    for label, examples in ((DOMAIN, domain_examples), (GENERAL, general_examples)):
        if not examples:
            out[label] = None
            continue
        with expert_mask(moe_blocks, model_type, {}, num_experts, mask_value):
            baseline = per_question_nll(forward, examples)
        with expert_mask(moe_blocks, model_type, masks, num_experts, mask_value):
            masked = per_question_nll(forward, examples)
        stats = paired_delta_stats(
            baseline, masked, min_delta, min_valid_fraction, n_boot, seed,
        )
        stats["baseline_nll"] = float(np.nanmean(baseline)) if baseline.size else float("nan")
        # A credible positive delta here means the prune set HURT this set, so
        # relabel: "verified" would read as an endorsement of the prune.
        stats["interpretation"] = {
            "verified": "degraded",
            "not_verified": "unchanged",
        }.get(stats["status"], stats["status"])
        out[label] = stats
    return out


# ---------------------------------------------------------------------------
# Trace pass
# ---------------------------------------------------------------------------

def generate_answer(model, tokenizer, prompt_ids: Sequence[int], max_tokens: int):
    """Greedily generate an answer, returning its token ids and text.

    Every token yielded by ``stream_generate`` — including the final EOS or
    truncation token — was already forwarded by the model before being yielded,
    so ``len(ids)`` is exactly the number of decode positions captured.

    Args:
        model: The model (the full wrapper; mlx-lm handles the forward).
        tokenizer: The tokenizer.
        prompt_ids: Prompt token ids.
        max_tokens: Maximum tokens to generate.

    Returns:
        (ids, text).
    """
    from mlx_lm.generate import stream_generate

    ids: List[int] = []
    chunks: List[str] = []
    for response in stream_generate(
        model, tokenizer, prompt=list(prompt_ids), max_tokens=max_tokens,
    ):
        ids.append(int(response.token))
        chunks.append(response.text)
    return ids, "".join(chunks)


def trace_question_set(
    forward,
    model,
    tokenizer,
    questions: Sequence[ProbeQuestion],
    label: str,
    stats: ProbeStats,
    moe_blocks: List,
    num_experts: int,
    answer_mode: str = "teacher",
    max_answer_tokens: int = 128,
    chat_template_args: Optional[dict] = None,
    system: Optional[str] = None,
    saliency_weighting: str = "question",
    echo=None,
    progress=None,
) -> Tuple[List[ProbeExample], List[dict]]:
    """Run one question set under the observer hooks and fold in the routing.

    Callers must have installed the observer hooks; this collects and clears the
    captures after every question so nothing accumulates across questions. No
    likelihood is computed here — all scoring happens later with plain forwards,
    so no scoring pass ever runs with captures pending.

    Args:
        forward: Token-only forward callable (used in teacher mode).
        model: The model, for generation.
        tokenizer: The tokenizer.
        questions: The question set.
        label: "domain" or "general".
        stats: ProbeStats to update.
        moe_blocks: Hooked MoE blocks.
        num_experts: Routed experts per layer.
        answer_mode: "teacher" or "generate".
        max_answer_tokens: Answer cap (reference truncation or generation limit).
        chat_template_args: Extra kwargs for the chat template.
        system: Default system prompt.
        saliency_weighting: "question" or "token".
        echo: Optional callable(index, question, answer_text) for progress output.
        progress: Optional callable(done, total).

    Returns:
        (examples, skipped) — skipped entries carry an index and a reason.
    """
    from .observer import collect_captures

    examples: List[ProbeExample] = []
    skipped: List[dict] = []

    for index, q in enumerate(questions):
        try:
            tokens, prompt_len = build_probe_tokens(
                tokenizer, q, answer_mode, max_answer_tokens,
                chat_template_args, system,
            )
        except ValueError as e:
            skipped.append({"index": index, "reason": str(e)})
            continue

        answer_text = q.answer or ""
        if answer_mode == "generate":
            generated, answer_text = generate_answer(
                model, tokenizer, tokens, max_answer_tokens,
            )
            if not generated:
                collect_captures(moe_blocks)
                skipped.append({"index": index, "reason": "generated no tokens"})
                continue
            tokens = list(tokens) + generated
        else:
            _forward_logits(forward, tokens)

        n_answer = len(tokens) - prompt_len
        captures = collect_captures(moe_blocks)
        try:
            sliced = [
                slice_answer_captures(block_captures, prompt_len, n_answer)
                for block_captures in captures
            ]
        except ValueError as e:
            skipped.append({"index": index, "reason": f"capture mismatch: {e}"})
            continue

        stats.add_question(label, sliced, saliency_weighting)
        examples.append(
            ProbeExample(tokens=list(tokens), prompt_len=prompt_len, question_index=index)
        )
        if echo is not None:
            echo(index, q.question, answer_text)
        if progress is not None:
            progress(index + 1, len(questions))

    return examples, skipped
