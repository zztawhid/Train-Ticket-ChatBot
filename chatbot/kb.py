from __future__ import annotations

"""
kb.py
-----
Knowledge Base (KB) component.

Loads the domain knowledge from data/knowledge_base.json and provides
lookup methods for:
  - Intent-based responses (greetings, farewells, etc.)
  - FAQ answers (railcard info, peak hours, refund policy, etc.)
  - Conversation-flow rules (what to ask next)
  - Fallback responses
"""

import os
import json
import random
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Load the knowledge base JSON file
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_KB_PATH = os.path.join(_DATA_DIR, "knowledge_base.json")

_kb: dict = {}


def _load_kb() -> None:
    """Load the KB from disk (called once at import time)."""
    global _kb
    if _kb:
        return
    with open(_KB_PATH, "r", encoding="utf-8") as fh:
        _kb = json.load(fh)


_load_kb()


# ---------------------------------------------------------------------------
# Intent responses
# ---------------------------------------------------------------------------

def get_intent_response(intent_tag: str) -> str | None:
    """
    Return a random response string for the given intent tag,
    or None if the intent is not in the KB.
    """
    for intent in _kb.get("intents", []):
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return None


# ---------------------------------------------------------------------------
# FAQ lookup (fuzzy-matched)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sensitive FAQ entries that MUST require an explicit keyword to match.
# This prevents accidental triggering on lexically-similar but unrelated
# phrases — e.g. "i want a train ticket" used to fuzzy-match
# "I want to suicide" because they share the "i want" prefix.
# Keys are matched against item["question"] lowercased.
# ---------------------------------------------------------------------------
_SENSITIVE_FAQ_TRIGGERS = {
    "i want to suicide": [
        "suicide", "suicidal", "kill myself", "end my life", "end it all",
        "hurt myself", "self harm", "self-harm", "want to die", "no reason to live",
    ],
}


def _sensitive_keyword_hit(question_lower: str, user_text_lower: str) -> bool:
    """Return True if the user's input contains a trigger keyword for a
    sensitive FAQ entry."""
    triggers = _SENSITIVE_FAQ_TRIGGERS.get(question_lower)
    if not triggers:
        return False
    return any(kw in user_text_lower for kw in triggers)


def get_faq_answer(question: str, threshold: int = 75) -> str | None:
    """
    Search the FAQ section of the KB for a question that fuzzy-matches
    the user's input. Returns the answer string, or None.

    Behaviour:
      1. If the input contains a trigger keyword for a sensitive FAQ
         entry (e.g. 'suicide', 'suicidal', 'kill myself'), return
         that entry directly - this is a hard keyword match, not a
         fuzzy one, so it both catches paraphrasings like
         "I am suicidal" and is never accidentally triggered by
         unrelated phrases like "I want a train ticket".
      2. Otherwise do fuzzy matching against the NON-sensitive entries
         only, with a threshold of 75. Sensitive entries are
         deliberately excluded from the fuzzy pool so they can never
         be reached by accident.
    """
    user_lower = question.lower()

    # Step 1: keyword-based hard match for sensitive entries
    for item in _kb.get("faq", []):
        q_lower = item["question"].lower()
        if _sensitive_keyword_hit(q_lower, user_lower):
            return item["answer"]

    # Step 2: fuzzy match the non-sensitive entries
    best_score = 0
    best_answer = None
    for item in _kb.get("faq", []):
        q_lower = item["question"].lower()
        if q_lower in _SENSITIVE_FAQ_TRIGGERS:
            continue  # sensitive entries skip fuzzy matching entirely
        score = fuzz.token_sort_ratio(user_lower, q_lower)
        if score > best_score:
            best_score = score
            best_answer = item["answer"]

    if best_score < threshold:
        return None
    return best_answer


# ---------------------------------------------------------------------------
# Conversation-flow rules
# ---------------------------------------------------------------------------

def get_rule_response(condition: str) -> str | None:
    """
    Look up the conversation-flow rules and return the response
    for the given condition (e.g. 'origin_missing').
    """
    rules = _kb.get("rules", {}).get("conversation_flow", [])
    for rule in rules:
        if rule["condition"] == condition:
            return rule["response"]
    return None


def get_all_rules() -> list[dict]:
    """Return all conversation-flow rules for the reasoning engine."""
    return _kb.get("rules", {}).get("conversation_flow", [])


# ---------------------------------------------------------------------------
# Fallback responses
# ---------------------------------------------------------------------------

def get_fallback_response() -> str:
    """Return a random fallback response when the chatbot cannot understand."""
    fallbacks = _kb.get("rules", {}).get("fallback_responses", [
        "Sorry, I didn't understand that. Could you rephrase?"
    ])
    return random.choice(fallbacks)
