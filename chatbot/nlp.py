"""
nlp.py
------
Natural Language Processing / Understanding component.

Uses spaCy for:
  - Named Entity Recognition (DATE, TIME, GPE, ORG)
  - Rule-based intent classification via Matcher and PhraseMatcher
  - Dependency parsing to distinguish origin vs destination stations

Uses dateparser for flexible date/time parsing ("next Friday",
"tomorrow morning", "after 10am", "before 2pm", etc.).
"""

import re
from datetime import datetime
from typing import Optional

import dateparser
import spacy
from spacy.matcher import Matcher, PhraseMatcher

from chatbot.station_lookup import lookup_station, get_all_station_names

# ---------------------------------------------------------------------------
# Load spaCy model (small English model)
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------
# Intent definitions — each intent has a set of spaCy Matcher patterns
# ---------------------------------------------------------------------------
# Pattern format follows spaCy's rule-based matching:
# https://spacy.io/usage/rule-based-matching

INTENT_PATTERNS = {
    "greet": [
        [{"LOWER": {"IN": ["hello", "hi", "hey", "hiya", "howdy", "yo", "greetings"]}}],
        [{"LOWER": "good"}, {"LOWER": {"IN": ["morning", "afternoon", "evening"]}}],
        [{"LOWER": {"IN": ["morning", "afternoon", "evening"]}}],
        [{"LOWER": {"IN": ["hi", "hey", "hello"]}}, {"LOWER": "there"}],
    ],
    "goodbye": [
        [{"LOWER": {"IN": ["bye", "goodbye", "cheers", "thanks", "later"]}}],
        [{"LOWER": "see"}, {"LOWER": {"IN": ["you", "ya"]}}],
        [{"LOWER": "thank"}, {"LOWER": "you"}],
        [{"LOWER": "take"}, {"LOWER": "care"}],
        [{"LOWER": "have"}, {"LOWER": "a"}, {"LOWER": {"IN": ["good", "nice", "great"]}}],
        [{"LOWER": "good"}, {"LOWER": "night"}],
        [{"LOWER": "goodnight"}],
    ],
    "find_ticket": [
        # Verb-led: "find/book/buy/get ... ticket/train/fare/journey"
        [{"LOWER": {"IN": ["find", "search", "book", "get", "buy", "look", "show"]}},
         {"OP": "*"}, {"LOWER": {"IN": ["ticket", "tickets", "train", "trains", "fare", "fares", "journey", "trip"]}}],
        # "I want/need/would like to go/travel/get/book ..."
        [{"LOWER": "i"}, {"LOWER": {"IN": ["want", "need", "would", "like"]}},
         {"OP": "*"}, {"LOWER": {"IN": ["go", "travel", "get", "book"]}}],
        # "I want/need a/the/an ... ticket/train/fare/journey"
        [{"LOWER": "i"}, {"LOWER": {"IN": ["want", "need", "would", "like"]}},
         {"OP": "*"},
         {"LOWER": {"IN": ["ticket", "tickets", "train", "trains", "fare", "fares", "journey"]}}],
        # "I'd like ..." - spaCy tokenises "I'd" as ['i', "'d"]
        [{"LOWER": "i"}, {"LOWER": "'d"}, {"LOWER": "like"},
         {"OP": "*"}, {"LOWER": {"IN": ["go", "travel", "book", "ticket", "tickets", "train", "trains"]}}],
        # Variant without apostrophe: "id like a ticket"
        [{"LOWER": "id"}, {"LOWER": "like"},
         {"OP": "*"}, {"LOWER": {"IN": ["go", "travel", "book", "ticket", "tickets", "train", "trains"]}}],
        # "Can/Could I/you book/find/get me a ticket/train"
        [{"LOWER": {"IN": ["can", "could"]}}, {"LOWER": {"IN": ["i", "you", "we"]}},
         {"OP": "*"}, {"LOWER": {"IN": ["book", "find", "buy", "get", "search", "look"]}},
         {"OP": "*"}, {"LOWER": {"IN": ["ticket", "tickets", "train", "trains", "fare", "fares", "journey"]}}],
        # "Please find/book/get me a ticket/train"
        [{"LOWER": "please"}, {"OP": "*"},
         {"LOWER": {"IN": ["find", "book", "buy", "get", "search", "show"]}},
         {"OP": "*"}, {"LOWER": {"IN": ["ticket", "tickets", "train", "trains"]}}],
        # "Cheapest ... ticket/fare/train"
        [{"LOWER": "cheapest"}, {"OP": "*"}, {"LOWER": {"IN": ["ticket", "tickets", "fare", "fares", "train", "trains"]}}],
        # Pricing questions: "How much ...", "What's the price ..."
        [{"LOWER": "how"}, {"LOWER": "much"}],
        # "What's the price/cost/fare" — spaCy tokenises "What's" as ['what', "'s"]
        [{"LOWER": "what"}, {"LOWER": "'s"}, {"LOWER": "the"}, {"LOWER": {"IN": ["price", "cost", "fare"]}}],
        [{"LOWER": "whats"}, {"LOWER": "the"}, {"LOWER": {"IN": ["price", "cost", "fare"]}}],
        [{"LOWER": "what"}, {"LOWER": "does"}, {"OP": "*"}, {"LOWER": "cost"}],
        # "Train to/from ..." / "Trains to/from ..."
        [{"LOWER": {"IN": ["train", "trains"]}}, {"LOWER": {"IN": ["to", "from"]}}],
        # "Going to/from ..."
        [{"LOWER": "going"}, {"LOWER": {"IN": ["to", "from"]}}],
        # "Help me book/find a ticket/train"
        [{"LOWER": "help"}, {"OP": "*"},
         {"LOWER": {"IN": ["book", "find", "buy", "get"]}},
         {"OP": "*"}, {"LOWER": {"IN": ["ticket", "tickets", "train", "trains"]}}],
    ],
    "single_ticket": [
        [{"LOWER": "single"}],
        [{"LOWER": "singles"}],
        [{"LOWER": "one"}, {"LOWER": "way"}],
        # spaCy splits "one-way" into ['one', '-', 'way']
        [{"LOWER": "one"}, {"LOWER": "-"}, {"LOWER": "way"}],
    ],
    "return_ticket": [
        [{"LOWER": "return"}],
        [{"LOWER": "returns"}],
        [{"LOWER": "round"}, {"LOWER": "trip"}],
        [{"LOWER": "round"}, {"LOWER": "-"}, {"LOWER": "trip"}],
        [{"LOWER": "come"}, {"LOWER": "back"}],
        [{"LOWER": "two"}, {"LOWER": "way"}],
        [{"LOWER": "both"}, {"LOWER": "ways"}],
    ],
    "affirm": [
        [{"LOWER": {"IN": ["yes", "yeah", "yep", "yea", "yup", "ya",
                            "sure", "correct", "right", "absolutely", "definitely",
                            "ok", "okay", "k", "fine", "confirm", "confirmed",
                            "perfect", "great"]}}],
        [{"LOWER": "that's"}, {"LOWER": "right"}],
        [{"LOWER": "thats"}, {"LOWER": "right"}],
        [{"LOWER": "go"}, {"LOWER": "ahead"}],
        [{"LOWER": "do"}, {"LOWER": "it"}],
        [{"LOWER": "sounds"}, {"LOWER": "good"}],
        [{"LOWER": "looks"}, {"LOWER": "good"}],
    ],
    "deny": [
        [{"LOWER": {"IN": ["no", "nope", "nah", "wrong", "incorrect", "not",
                            "cancel", "stop"]}}],
        [{"LOWER": "start"}, {"LOWER": {"IN": ["over", "again"]}}],
        [{"LOWER": "go"}, {"LOWER": "back"}],
        [{"LOWER": "change"}],
        [{"LOWER": "that's"}, {"LOWER": {"IN": ["wrong", "not"]}}],
        [{"LOWER": "not"}, {"LOWER": {"IN": ["really", "quite"]}}],
    ],
    "help": [
        [{"LOWER": {"IN": ["help", "info", "information"]}}],
        [{"LOWER": "what"}, {"LOWER": "can"}, {"LOWER": "you"}, {"LOWER": "do"}],
        [{"LOWER": "what"}, {"LOWER": "do"}, {"LOWER": "you"}, {"LOWER": "do"}],
        [{"LOWER": "how"}, {"LOWER": "do"}, {"LOWER": "you"}, {"LOWER": "work"}],
        [{"LOWER": "how"}, {"LOWER": "does"}, {"LOWER": "this"}, {"LOWER": "work"}],
        [{"LOWER": {"IN": ["who", "what"]}}, {"LOWER": {"IN": ["are", "is"]}}, {"LOWER": "you"}],
        [{"LOWER": "guide"}, {"LOWER": "me"}],
        [{"LOWER": "explain"}],
    ],
}

# ---------------------------------------------------------------------------
# Build the spaCy Matcher with the intent patterns
# ---------------------------------------------------------------------------
_matcher = Matcher(nlp.vocab)

for intent_name, patterns in INTENT_PATTERNS.items():
    _matcher.add(intent_name, patterns)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def classify_intent(text: str) -> str:
    """
    Determine the user's intent from their message.

    Returns one of the intent names defined in INTENT_PATTERNS,
    or 'fallback' if no pattern matches.
    """
    doc = nlp(text.lower())
    matches = _matcher(doc)

    if not matches:
        return "fallback"

    # Pick the intent with the longest span (most specific match)
    best_intent = None
    best_length = 0
    for match_id, start, end in matches:
        intent_name = nlp.vocab.strings[match_id]
        span_length = end - start
        if span_length > best_length:
            best_length = span_length
            best_intent = intent_name

    return best_intent or "fallback"


def extract_stations(text: str) -> dict:
    """
    Extract origin and destination station names from the user's message.

    Uses keyword context ("from X", "to Y") and fallback ordering.

    Returns
    -------
    dict with keys:
        'origin'      : (crs, display_name, score) or None
        'destination'  : (crs, display_name, score) or None
        'ambiguous'    : list of candidates if fuzzy match is uncertain
    """
    result = {"origin": None, "destination": None, "ambiguous": []}
    text_lower = text.lower()

    # --- Strategy 1: Look for "from <station> to <station>" combined pattern first ---
    combined_match = re.search(
        r'\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+on\b|\s+at\b|\s+departing|\s+leaving|\s+tomorrow|\s+next|\s+this|\s+after|\s+before|\s+around|\s+for\b|$)',
        text_lower,
    )
    if combined_match:
        from_match_text = combined_match.group(1).strip().rstrip(".,!?")
        to_match_text = combined_match.group(2).strip().rstrip(".,!?")
    else:
        from_m = re.search(r'\bfrom\s+(.+?)(?:\s+to\b|\s+on\b|\s+at\b|\s+departing|\s+leaving|$)', text_lower)
        from_match_text = from_m.group(1).strip().rstrip(".,!?") if from_m else None

        # For "to", skip common verbs that follow "to" (go, travel, get, book, find)
        to_m = re.search(
            r'\bto\s+(?!go\b|travel\b|get\b|book\b|find\b|do\b|be\b)(.+?)(?:\s+from\b|\s+on\b|\s+at\b|\s+departing|\s+leaving|\s+returning|\s+tomorrow|\s+next|\s+this|\s+after|\s+before|\s+around|\s+for\b|$)',
            text_lower,
        )
        to_match_text = to_m.group(1).strip().rstrip(".,!?") if to_m else None

    if from_match_text:
        crs, candidates = lookup_station(from_match_text)
        if crs:
            display = candidates[0][0] if candidates else from_match_text.title()
            result["origin"] = (crs, display, candidates[0][2] if candidates else 100)
        elif candidates:
            result["ambiguous"].append(("origin", candidates))

    if to_match_text:
        crs, candidates = lookup_station(to_match_text)
        if crs:
            display = candidates[0][0] if candidates else to_match_text.title()
            result["destination"] = (crs, display, candidates[0][2] if candidates else 100)
        elif candidates:
            result["ambiguous"].append(("destination", candidates))

    # --- Strategy 1b: Handle "<station> to <station>" with no "from" keyword ---
    if result["destination"] and not result["origin"] and not combined_match:
        # Try to extract the station name before the "to" keyword
        before_to = re.search(r'(\b\w[\w\s]*?)\s+to\s+', text_lower)
        if before_to:
            before_text = before_to.group(1).strip()
            # Remove leading common words (I, want, need, would, like, go, etc.)
            # Use \b so the pattern matches even when there is no trailing space
            # (e.g. "i want" with nothing after "want" would otherwise be kept).
            before_text = re.sub(
                r'^(?:i\s+)?(?:want|need|would|like|am|do|can|please|go|going)\b\s*', '', before_text
            ).strip()
            if before_text:
                crs, candidates = lookup_station(before_text)
                if crs and crs != result["destination"][0]:
                    display = candidates[0][0] if candidates else before_text.title()
                    result["origin"] = (crs, display, candidates[0][2] if candidates else 100)

    # --- Strategy 2: Use spaCy NER for GPE / ORG entities as station names ---
    if not result["origin"] or not result["destination"]:
        doc = nlp(text)
        found_stations = []

        for ent in doc.ents:
            if ent.label_ in ("GPE", "ORG", "FAC", "LOC"):
                crs, candidates = lookup_station(ent.text)
                if crs:
                    display = candidates[0][0] if candidates else ent.text.title()
                    score = candidates[0][2] if candidates else 100
                    # Avoid duplicates
                    if not any(s[0] == crs for s in found_stations):
                        found_stations.append((crs, display, score))

        # Assign found entities to empty slots in order
        for station_info in found_stations:
            if not result["origin"] or result["origin"][0] == station_info[0]:
                if not result["origin"]:
                    result["origin"] = station_info
            elif not result["destination"]:
                result["destination"] = station_info

    return result


def extract_station_direct(text: str):
    """
    Try to resolve the entire text as a single station name.
    Used when the chatbot has asked specifically for a station.

    Returns (crs, display_name, score) or None, plus candidates list.
    """
    text = text.strip().rstrip(".,!?")
    crs, candidates = lookup_station(text)
    if crs and candidates:
        return (crs, candidates[0][0], candidates[0][2]), candidates
    return None, candidates


def _preprocess_date_text(text: str) -> str:
    """
    Normalise date expressions that dateparser struggles with.

    Converts:
      - 'next Friday'      → 'Friday'
      - 'this Friday'      → 'Friday'
      - 'two days later'   → 'in 2 days'
      - 'three days later' → 'in 3 days'
      - 'coming Monday'    → 'Monday'
    """
    text = re.sub(r'\b(next|this|this coming|upcoming)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcoming\s+', '', text, flags=re.IGNORECASE)

    # "X days later" → "in X days"
    word_to_num = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "a couple of": "2", "a few": "3",
    }
    for word, num in word_to_num.items():
        text = re.sub(rf'\b{word}\b', num, text, flags=re.IGNORECASE)

    text = re.sub(r'(\d+)\s+(days?|weeks?|months?)\s+later', r'in \1 \2', text, flags=re.IGNORECASE)

    # Convert time-of-day words to approximate times
    time_of_day = {
        "morning": "09:00",
        "afternoon": "14:00",
        "evening": "18:00",
        "night": "20:00",
        "midday": "12:00",
        "noon": "12:00",
        "lunchtime": "12:30",
    }
    for word, time_val in time_of_day.items():
        if word in text.lower():
            text = re.sub(rf'\b{word}\b', time_val, text, flags=re.IGNORECASE)

    return text.strip()


def extract_datetime(text: str, relative_base: datetime = None) -> tuple[Optional[str], str]:
    """
    Parse a date/time expression from user text.

    Uses spaCy NER to first extract DATE/TIME entities, then falls back
    to parsing the entire cleaned text through dateparser.

    Parameters
    ----------
    text          : the user's input text
    relative_base : optional datetime to use as "now" for relative
                    expressions like "in 2 days". Useful for parsing
                    return dates relative to the outward date.

    Returns
    -------
    (formatted_datetime: str | None, constraint: str)
        - formatted_datetime in ISO format 'YYYY-MM-DDTHH:MM:SS'
        - constraint is 'departBy' or 'arriveBy'
    """
    text_lower = text.lower()

    # Determine time constraint from keywords.
    # Use \bby\b (whole-word) so station names like Whitby/Grimsby/Selby don't
    # accidentally trigger "arrive by" mode.
    constraint = "departBy"
    arrive_phrases = ["before", "arrive by", "arriving by", "get there by", "no later than"]
    if any(phrase in text_lower for phrase in arrive_phrases) or bool(
        re.search(r"\bby\b", text_lower)
    ):
        constraint = "arriveBy"

    base = relative_base or datetime.now()
    _dp_settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": base,
        "RETURN_AS_TIMEZONE_AWARE": False,
    }

    # Detect whether the user typed an explicit time token. If they did,
    # we must make sure the final parse captured it — spaCy may miss tokens
    # like "4pm" when there is no "at" before them, in which case Strategy A
    # would return the current time inherited from RELATIVE_BASE.
    _has_time_token = bool(re.search(
        r"\b\d{1,2}\s*(?:am|pm)\b|\b\d{1,2}[:.]\d{2}\b", text_lower
    ))

    # --- Strategy A: Use spaCy to extract DATE/TIME entities first ---
    doc = nlp(text)
    date_entities = [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
    _entity_has_time_token = bool(re.search(
        r"\b\d{1,2}\s*(?:am|pm)\b|\b\d{1,2}[:.]\d{2}\b",
        " ".join(date_entities).lower()
    )) if date_entities else False

    if date_entities:
        combined = " ".join(date_entities)
        preprocessed = _preprocess_date_text(combined)
        parsed = dateparser.parse(preprocessed, settings=_dp_settings)
        # Return Strategy A's result only if:
        #   (a) it isn't midnight (would mean no time captured), AND
        #   (b) either the user didn't type an explicit time token at all,
        #       OR spaCy actually captured that token in its entities.
        # Otherwise fall through to Strategy B which parses the full raw text.
        if (parsed
            and parsed.strftime("%H:%M:%S") != "00:00:00"
            and (not _has_time_token or _entity_has_time_token)):
            return parsed.strftime("%Y-%m-%dT%H:%M:%S"), constraint
        _strategy_a_result = parsed
    else:
        _strategy_a_result = None

    # --- Strategy B: Clean and preprocess the full text ---
    cleaned = text_lower
    for kw in ["before", "after", "by", "around", "at", "departing", "leaving",
               "arriving", "depart", "arrive"]:
        cleaned = cleaned.replace(kw, " ")
    cleaned = _preprocess_date_text(cleaned)

    parsed = dateparser.parse(cleaned, settings=_dp_settings)
    if parsed and parsed.strftime("%H:%M:%S") != "00:00:00":
        return parsed.strftime("%Y-%m-%dT%H:%M:%S"), constraint

    # --- Strategy C: Try the raw preprocessed text ---
    preprocessed_raw = _preprocess_date_text(text)
    parsed = dateparser.parse(preprocessed_raw, settings=_dp_settings)
    if parsed and parsed.strftime("%H:%M:%S") != "00:00:00":
        return parsed.strftime("%Y-%m-%dT%H:%M:%S"), constraint

    # All strategies found midnight or nothing — return the best available result
    # (Strategy A's date-only result, or B/C's midnight, or None)
    for candidate in [parsed, _strategy_a_result]:
        if candidate is not None:
            return candidate.strftime("%Y-%m-%dT%H:%M:%S"), constraint

    return None, constraint


def extract_ticket_type(text: str) -> Optional[str]:
    """
    Detect whether the user wants a 'single' or 'return' ticket.

    Returns 'single', 'return', or None if not detected.
    """
    doc = nlp(text.lower())
    matches = _matcher(doc)

    for match_id, start, end in matches:
        intent = nlp.vocab.strings[match_id]
        if intent == "single_ticket":
            return "single"
        elif intent == "return_ticket":
            return "return"

    return None
