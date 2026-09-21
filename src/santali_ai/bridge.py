"""Internal two-way Santali ↔ Gemma bridge.

This module is deliberately model-agnostic and has no public server. It defines
an auditable orchestration contract so real IndicTrans2 and Gemma adapters can
be plugged in only after offline evaluation passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

Language = Literal["sat", "eng", "hin", "ben", "unknown"]
Translate = Callable[[str, Language, Language], str]
Answer = Callable[[str, str], str]


@dataclass(frozen=True)
class BridgePolicy:
    min_translation_confidence: float = 0.80
    min_gemma_confidence: float = 0.75
    allow_public_release: bool = False


@dataclass
class BridgeResult:
    ok: bool
    user_language: Language
    internal_language: Language
    source_text: str
    translated_prompt: str = ""
    gemma_answer: str = ""
    translated_answer: str = ""
    translation_confidence: float = 0.0
    gemma_confidence: float = 0.0
    needs_review: bool = True
    reason: str = ""
    trace: list[str] = field(default_factory=list)


class InternalBridge:
    """Offline/internal orchestration only; no user-facing exposure."""

    def __init__(
        self,
        translate: Translate,
        answer: Answer,
        policy: BridgePolicy | None = None,
    ) -> None:
        self.translate = translate
        self.answer = answer
        self.policy = policy or BridgePolicy()
        if self.policy.allow_public_release:
            raise ValueError("Public release is disabled until evaluation gates pass")

    def ask_from_santali(
        self,
        text: str,
        translation_confidence: float,
        gemma_confidence: float,
    ) -> BridgeResult:
        result = BridgeResult(
            ok=False,
            user_language="sat",
            internal_language="eng",
            source_text=text,
            translation_confidence=translation_confidence,
            gemma_confidence=gemma_confidence,
            trace=["internal_only", "sat_to_eng_to_gemma_to_sat"],
        )
        if not text.strip():
            result.reason = "empty_input"
            return result
        if translation_confidence < self.policy.min_translation_confidence:
            result.reason = "translation_confidence_below_gate"
            result.trace.append("clarification_required")
            return result

        result.translated_prompt = self.translate(text, "sat", "eng")
        if gemma_confidence < self.policy.min_gemma_confidence:
            result.reason = "gemma_confidence_below_gate"
            result.trace.append("gemma_uncertain")
            return result

        result.gemma_answer = self.answer(result.translated_prompt, "Answer clearly; state uncertainty when evidence is missing.")
        result.translated_answer = self.translate(result.gemma_answer, "eng", "sat")
        result.ok = bool(result.translated_answer.strip())
        result.needs_review = True
        result.reason = "offline_candidate_requires_review"
        result.trace.append("candidate_not_public")
        return result

    def ask_from_english(
        self,
        text: str,
        translation_confidence: float,
        gemma_confidence: float,
    ) -> BridgeResult:
        result = BridgeResult(
            ok=False,
            user_language="eng",
            internal_language="eng",
            source_text=text,
            translation_confidence=translation_confidence,
            gemma_confidence=gemma_confidence,
            trace=["internal_only", "eng_to_gemma_to_sat"],
        )
        if not text.strip():
            result.reason = "empty_input"
            return result
        if gemma_confidence < self.policy.min_gemma_confidence:
            result.reason = "gemma_confidence_below_gate"
            result.trace.append("gemma_uncertain")
            return result
        result.gemma_answer = self.answer(text, "Answer clearly; do not invent facts.")
        if translation_confidence < self.policy.min_translation_confidence:
            result.reason = "answer_translation_confidence_below_gate"
            result.trace.append("translation_review_required")
            return result
        result.translated_answer = self.translate(result.gemma_answer, "eng", "sat")
        result.ok = bool(result.translated_answer.strip())
        result.needs_review = True
        result.reason = "offline_candidate_requires_review"
        result.trace.append("candidate_not_public")
        return result


def demo_bridge() -> BridgeResult:
    """Run a deterministic contract demo without loading any model."""
    def fake_translate(text: str, source: Language, target: Language) -> str:
        return f"[{source}->{target}] {text}"

    def fake_answer(prompt: str, instruction: str) -> str:
        return f"[gemma-dry-run] {prompt}"

    bridge = InternalBridge(fake_translate, fake_answer)
    return bridge.ask_from_santali("ᱥᱟᱱᱛᱟᱲᱤ", 0.99, 0.95)


if __name__ == "__main__":
    print(demo_bridge())
