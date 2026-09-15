#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
content_validator.py — Content quality validator for carousel generation.
Validates carousel content before rendering to ensure quality and variety.
"""

import re
from typing import Dict, List, Tuple, Any


class ValidationResult:
    """Result of a validation check."""

    def __init__(self):
        self.passed: List[str] = []
        self.warnings: List[str] = []
        self.failures: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.failures) == 0

    @property
    def score(self) -> float:
        total = len(self.passed) + len(self.warnings) + len(self.failures)
        if total == 0:
            return 1.0
        return len(self.passed) / total

    def add_pass(self, msg: str):
        self.passed.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_failure(self, msg: str):
        self.failures.append(msg)

    def __str__(self) -> str:
        lines = []
        if self.passed:
            lines.append(f"✅ Passed: {len(self.passed)}")
        if self.warnings:
            lines.append(f"⚠️  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"   - {w}")
        if self.failures:
            lines.append(f"❌ Failures: {len(self.failures)}")
            for f in self.failures:
                lines.append(f"   - {f}")
        return "\n".join(lines)


def _check_slide_purposes(carousel: Dict, result: ValidationResult):
    """Check that every slide has a clear purpose."""
    slides = carousel.get("slides", [])
    if not slides:
        result.add_failure("No slides found")
        return

    for i, slide in enumerate(slides):
        slide_type = slide.get("type") or slide.get("purpose", "")
        if not slide_type:
            result.add_failure(f"Slide {i + 1} has no type/purpose")
        title = slide.get("title", "")
        if not title or len(title.strip()) < 3:
            result.add_warning(f"Slide {i + 1} has weak/missing title: '{title}'")
        else:
            result.add_pass(f"Slide {i + 1} has purpose: {slide_type}")


def _check_real_world_examples(carousel: Dict, result: ValidationResult):
    """Check for at least one real-world example."""
    slides = carousel.get("slides", [])
    has_example = False

    example_indicators = [
        "real_world", "example", "scenario", "analogy",
        "imagine", "company", "application",
    ]

    for slide in slides:
        slide_type = (slide.get("type") or slide.get("purpose", "")).lower()
        body = str(slide.get("body", "")).lower()

        if any(ind in slide_type for ind in example_indicators):
            has_example = True
            break
        if any(ind in body for ind in ["imagine", "think of", "for example", "real-world", "scenario:"]):
            has_example = True
            break

    if has_example:
        result.add_pass("Contains at least one real-world example")
    else:
        result.add_warning("No clear real-world example found")


def _check_visual_variety(carousel: Dict, result: ValidationResult):
    """Check that slides use varied layout types."""
    slides = carousel.get("slides", [])
    types = [s.get("type") or s.get("layout_type", "") for s in slides]

    unique_types = set(types)
    if len(unique_types) < min(3, len(types)):
        result.add_warning(f"Low visual variety: only {len(unique_types)} unique layout types for {len(types)} slides")
    else:
        result.add_pass(f"Good visual variety: {len(unique_types)} unique layout types")


def _check_text_lengths(carousel: Dict, result: ValidationResult):
    """Check text lengths for mobile readability."""
    slides = carousel.get("slides", [])

    for i, slide in enumerate(slides):
        title = slide.get("title", "")
        body = slide.get("body", "")

        # Title checks
        if len(title) > 80:
            result.add_warning(f"Slide {i + 1} title too long ({len(title)} chars): may overflow on mobile")

        # Body checks
        if isinstance(body, str):
            if len(body) > 500:
                result.add_warning(f"Slide {i + 1} body too long ({len(body)} chars): reduce for mobile readability")
            elif len(body) > 0:
                result.add_pass(f"Slide {i + 1} body length OK ({len(body)} chars)")
        elif isinstance(body, list):
            for j, item in enumerate(body):
                if isinstance(item, str) and len(item) > 150:
                    result.add_warning(f"Slide {i + 1} bullet {j + 1} too long ({len(item)} chars)")


def _check_hook_quality(carousel: Dict, result: ValidationResult):
    """Check that the first slide is a compelling hook."""
    slides = carousel.get("slides", [])
    if not slides:
        return

    first = slides[0]
    title = first.get("title", "")
    body = str(first.get("body", ""))

    # Hook should be short and punchy
    if len(title) > 60:
        result.add_warning(f"Hook title may be too long for impact ({len(title)} chars)")
    elif len(title) < 5:
        result.add_failure("Hook title is too short / missing")
    else:
        result.add_pass("Hook title is appropriately sized")

    # Check for engagement patterns
    hook_patterns = ["?", "!", "how", "why", "what", "when", "imagine", "did you know"]
    has_hook_pattern = any(p in (title + body).lower() for p in hook_patterns)
    if has_hook_pattern:
        result.add_pass("Hook uses engagement pattern (question/exclamation)")
    else:
        result.add_warning("Hook could be more engaging (consider using a question or exclamation)")


def _check_final_slide(carousel: Dict, result: ValidationResult):
    """Check that the final slide has a clear CTA."""
    slides = carousel.get("slides", [])
    if not slides:
        return

    last = slides[-1]
    slide_type = (last.get("type") or last.get("purpose", "")).lower()
    body = str(last.get("body", "")).lower()

    cta_indicators = ["takeaway", "cta", "follow", "save", "share", "subscribe", "key points", "summary"]
    has_cta = any(ind in slide_type or ind in body for ind in cta_indicators)

    if has_cta:
        result.add_pass("Final slide has clear CTA/takeaway")
    else:
        result.add_warning("Final slide missing clear CTA or takeaway")


def _check_code_validity(carousel: Dict, result: ValidationResult):
    """Basic check that code examples look syntactically reasonable."""
    slides = carousel.get("slides", [])

    for i, slide in enumerate(slides):
        slide_type = (slide.get("type") or slide.get("layout_type", "")).lower()
        body = str(slide.get("body", ""))
        code = slide.get("code", "")

        if "code" in slide_type or code:
            code_text = code if code else body
            # Basic checks
            if len(code_text.strip()) < 10:
                result.add_warning(f"Slide {i + 1} code example seems too short")
            else:
                result.add_pass(f"Slide {i + 1} has code content")


def _check_technical_accuracy_basics(carousel: Dict, result: ValidationResult):
    """Basic checks for common technical inaccuracies."""
    slides = carousel.get("slides", [])
    all_text = " ".join(
        str(s.get("title", "")) + " " + str(s.get("body", ""))
        for s in slides
    ).lower()

    # Check for common misspellings of tech terms
    misspellings = {
        "kubernettes": "kubernetes",
        "kubernets": "kubernetes",
        "javascipt": "javascript",
        "pytohn": "python",
        "dockr": "docker",
        "terrafrom": "terraform",
    }
    for wrong, correct in misspellings.items():
        if wrong in all_text:
            result.add_failure(f"Possible misspelling: '{wrong}' → should be '{correct}'")

    result.add_pass("Basic technical accuracy check passed")


def validate_carousel(carousel: Dict) -> ValidationResult:
    """
    Run all validation checks on a carousel.

    Args:
        carousel: The carousel JSON dict

    Returns:
        ValidationResult with pass/warn/fail details
    """
    result = ValidationResult()

    _check_slide_purposes(carousel, result)
    _check_real_world_examples(carousel, result)
    _check_visual_variety(carousel, result)
    _check_text_lengths(carousel, result)
    _check_hook_quality(carousel, result)
    _check_final_slide(carousel, result)
    _check_code_validity(carousel, result)
    _check_technical_accuracy_basics(carousel, result)

    return result


def validate_and_report(carousel: Dict) -> bool:
    """Validate and print a report. Returns True if valid."""
    result = validate_carousel(carousel)
    print(f"\n📋 Content Validation Report:")
    print(str(result))
    print(f"   Score: {result.score:.0%}")
    print(f"   Valid: {'✅ Yes' if result.is_valid else '❌ No'}")
    return result.is_valid
