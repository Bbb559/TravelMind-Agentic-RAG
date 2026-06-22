"""Reproducible evaluation helpers for the public TravelMind benchmark."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence


ROUTE_LABELS = (
    "naive_rag",
    "graphrag",
    "multimodal_rag",
    "hybrid_rag",
    "invalid_input",
    "fallback",
)


@dataclass(frozen=True)
class EvaluationBundle:
    route_cases: tuple[dict[str, Any], ...]
    workflow_cases: tuple[dict[str, Any], ...]
    paid_local_cases: tuple[dict[str, Any], ...]
    manual_annotations: tuple[dict[str, Any], ...]


def load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"jsonl_row_not_object:{path.name}:{line_number}")
        rows.append(value)
    return tuple(rows)


def load_evaluation_bundle(root: Path) -> EvaluationBundle:
    return EvaluationBundle(
        route_cases=load_jsonl(root / "route_cases.jsonl"),
        workflow_cases=load_jsonl(root / "workflow_cases.jsonl"),
        paid_local_cases=load_jsonl(root / "paid_local_cases.jsonl"),
        manual_annotations=load_jsonl(root / "manual_annotations.jsonl"),
    )


def validate_evaluation_bundle(bundle: EvaluationBundle) -> dict[str, object]:
    route_counts = Counter(
        str(case.get("expected_route"))
        for case in bundle.route_cases
    )
    all_rows = [
        *bundle.route_cases,
        *bundle.workflow_cases,
        *bundle.paid_local_cases,
        *bundle.manual_annotations,
    ]
    ids = [
        str(case.get("id") or case.get("case_id") or "")
        for case in all_rows
    ]
    queries = [
        str(case["query"]).strip()
        for case in (
            *bundle.route_cases,
            *bundle.workflow_cases,
            *bundle.paid_local_cases,
        )
    ]
    duplicate_ids = sorted(
        value
        for value, count in Counter(ids).items()
        if value and count > 1
    )
    duplicate_queries = sorted(
        value
        for value, count in Counter(queries).items()
        if value and count > 1
    )
    expected_route_counts = {
        "naive_rag": 15,
        "graphrag": 10,
        "multimodal_rag": 10,
        "hybrid_rag": 10,
        "invalid_input": 5,
        "fallback": 10,
    }
    if dict(route_counts) != expected_route_counts:
        raise ValueError("route_case_distribution_mismatch")
    if len(bundle.workflow_cases) != 40:
        raise ValueError("workflow_case_count_mismatch")
    if len(bundle.paid_local_cases) != 6:
        raise ValueError("paid_local_case_count_mismatch")
    if len(bundle.manual_annotations) != 30:
        raise ValueError("manual_annotation_count_mismatch")
    if duplicate_ids:
        raise ValueError(f"duplicate_evaluation_ids:{','.join(duplicate_ids)}")
    if duplicate_queries:
        raise ValueError("duplicate_evaluation_queries")
    return {
        "route_counts": dict(route_counts),
        "workflow_count": len(bundle.workflow_cases),
        "paid_local_count": len(bundle.paid_local_cases),
        "manual_annotation_count": len(bundle.manual_annotations),
        "duplicate_ids": duplicate_ids,
        "duplicate_queries": duplicate_queries,
    }


def route_classification_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, object]:
    if len(expected) != len(predicted):
        raise ValueError("route_prediction_length_mismatch")
    if not expected:
        raise ValueError("route_predictions_empty")

    correct = sum(
        actual == wanted
        for wanted, actual in zip(expected, predicted, strict=True)
    )
    per_route: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for route in ROUTE_LABELS:
        true_positive = sum(
            wanted == route and actual == route
            for wanted, actual in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            wanted != route and actual == route
            for wanted, actual in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            wanted == route and actual != route
            for wanted, actual in zip(expected, predicted, strict=True)
        )
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        f1_values.append(f1)
        per_route[route] = {
            "support": Counter(expected)[route],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "sample_count": len(expected),
        "accuracy": correct / len(expected),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_route": per_route,
    }


def workflow_effect_metrics(
    cases: Sequence[dict[str, Any]],
    payloads: Sequence[dict[str, Any]],
) -> dict[str, object]:
    if len(cases) != len(payloads):
        raise ValueError("workflow_prediction_length_mismatch")
    answerable_count = sum(bool(case.get("answerable")) for case in cases)
    unanswerable_count = len(cases) - answerable_count
    hits = 0
    safe_refusals = 0
    unsafe_generations = 0
    rows: list[dict[str, object]] = []

    for case, payload in zip(cases, payloads, strict=True):
        answerable = bool(case.get("answerable"))
        hit = _evidence_hit_at_3(case, payload)
        generation_mode = str(
            (payload.get("execution_status") or {}).get(
                "generation_mode",
                "none",
            )
        )
        fallback_reason = payload.get("fallback_reason")
        safe_refusal = bool(
            not answerable
            and _is_safe_refusal_payload(payload)
        )
        unsafe_generation = bool(
            not answerable
            and not safe_refusal
            and (
                generation_mode != "none"
                or _has_valid_evidence(payload)
            )
        )
        if answerable and hit:
            hits += 1
        if safe_refusal:
            safe_refusals += 1
        if unsafe_generation:
            unsafe_generations += 1
        rows.append(
            {
                "id": case.get("id"),
                "answerable": answerable,
                "route_correct": (
                    payload.get("route") == case.get("expected_route")
                ),
                "evidence_hit_at_3": hit,
                "safe_refusal": safe_refusal,
                "unsafe_generation": unsafe_generation,
                "fallback_reason": fallback_reason,
                "generation_mode": generation_mode,
            }
        )

    return {
        "sample_count": len(cases),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "evidence_hit_at_3": _safe_ratio(hits, answerable_count),
        "safe_refusal_rate": _safe_ratio(
            safe_refusals,
            unanswerable_count,
        ),
        "unsafe_generation_rate": _safe_ratio(
            unsafe_generations,
            unanswerable_count,
        ),
        "cases": rows,
    }


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        raise ValueError("percentile_values_empty")
    if not 0 <= percent <= 100:
        raise ValueError("percentile_out_of_range")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def manual_faithfulness_metrics(
    annotations: Sequence[dict[str, Any]],
) -> dict[str, object]:
    if not annotations or any(
        annotation.get("review_status") != "completed"
        for annotation in annotations
    ):
        raise ValueError("manual_annotations_incomplete")

    claim_counts = Counter()
    answers_with_unsupported = 0
    for annotation in annotations:
        claims = annotation.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError("manual_annotation_claims_missing")
        labels: list[str] = []
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError("manual_annotation_claim_invalid")
            label = str(claim.get("label") or "")
            if label not in {
                "supported",
                "unsupported",
                "not_verifiable",
            }:
                raise ValueError("manual_annotation_label_invalid")
            labels.append(label)
            claim_counts[label] += 1
        if "unsupported" in labels:
            answers_with_unsupported += 1

    verifiable = claim_counts["supported"] + claim_counts["unsupported"]
    if verifiable == 0:
        raise ValueError("manual_annotations_no_verifiable_claims")
    return {
        "answer_count": len(annotations),
        "claim_counts": dict(claim_counts),
        "claim_support_rate": claim_counts["supported"] / verifiable,
        "answer_hallucination_rate": (
            answers_with_unsupported / len(annotations)
        ),
    }


def _evidence_hit_at_3(
    case: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    expected_entities = {
        str(value)
        for value in case.get("expected_evidence_entities", [])
    }
    expected_intents = {
        str(value)
        for value in case.get("expected_evidence_intents", [])
    }
    for result in list(payload.get("retrieved") or [])[:3]:
        if not isinstance(result, dict):
            continue
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get("evidence_valid") is not True:
            continue
        matched_entities = {
            str(value)
            for value in metadata.get("matched_entities", [])
        }
        matched_intents = {
            str(value)
            for value in metadata.get("matched_intents", [])
        }
        if (
            expected_entities <= matched_entities
            and expected_intents <= matched_intents
        ):
            return True
    return False


def _has_valid_evidence(payload: dict[str, Any]) -> bool:
    return any(
        isinstance(result, dict)
        and isinstance(result.get("metadata"), dict)
        and result["metadata"].get("evidence_valid") is True
        for result in payload.get("retrieved") or []
    )


def _is_safe_refusal_payload(payload: dict[str, Any]) -> bool:
    if not payload.get("fallback_reason"):
        return False
    generation_mode = str(
        (payload.get("execution_status") or {}).get(
            "generation_mode",
            "none",
        )
    )
    if generation_mode == "none":
        return True
    answer = str(payload.get("answer") or "")
    return any(
        marker in answer
        for marker in (
            "资料不足",
            "没有找到足够证据",
            "未找到足够证据",
            "无法生成具体",
            "无法可靠回答",
            "无法回答",
            "不生成具体",
            "不属于",
            "请先输入",
            "未成功",
            "不代表官方",
        )
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
