from __future__ import annotations

from datetime import datetime as _dt_now  # used for past-date validation

"""
conversation.py
---------------
Dialogue Manager — controls the conversation flow using a state machine.

States:
    GREETING           → Initial state, waiting for user to start
    COLLECT_ORIGIN     → Asking for departure station
    COLLECT_DESTINATION→ Asking for destination station
    COLLECT_DATE       → Asking for travel date
    COLLECT_TIME       → Asking for travel time
    COLLECT_TICKET_TYPE→ Asking single or return
    COLLECT_RETURN_DATE→ Asking for return date (if return ticket)
    COLLECT_RETURN_TIME→ Asking for return time (if return ticket)
    CONFIRM            → Showing summary, waiting for confirmation
    SEARCHING          → API call in progress
    RESULTS            → Displaying results
    DONE               → Conversation complete

The state is stored in Streamlit's st.session_state so it persists
across re-runs.
"""

from chatbot.nlp import (
    classify_intent,
    extract_stations,
    extract_station_direct,
    extract_datetime,
    extract_ticket_type,
)
from chatbot.reasoning import infer_next_action, handle_intent, get_fallback
from chatbot.station_lookup import crs_to_name
from chatbot.ticket_search import search_cheapest_ticket
from chatbot.database import save_message

# ---------------------------------------------------------------------------
# Off-topic detection
# ---------------------------------------------------------------------------

# Words / phrases that can never begin a UK station name — used to detect
# off-topic messages before attempting fuzzy station lookup.
_QUESTION_STARTERS = (
    # Question words
    "what ", "what's", "whats",
    "how ", "how's", "hows",
    "why ", "who ", "when is", "when are",
    "is ", "are ", "do ", "does ",
    "will ", "would ", "could ", "should ",
    "can you", "could you",
    # Explanation requests
    "tell me", "explain",
    # First-person statements unrelated to travel
    "i want", "i need", "i'd like", "i would like",
    "i'm ", "i am ", "i've ", "i have ", "im ",
    "i don't", "i dont", "i'm not", "i am not", "i dont",
    "i just", "i only", "i already",
    # Common off-topic openers
    "the weather", "the time",
    "please help", "can you help",
    "my name", "my order",
    "help me",
)


def _is_off_topic_query(user_text: str, intent: str) -> bool:
    """
    Return True if the message is clearly off-topic — i.e. not a station
    name, date, or travel-related input.

    Prevents RapidFuzz from fuzzy-matching phrases like "What's the weather
    like" against station names (e.g. "Wetherby").

    Only fires when intent is 'fallback'; booking-related intents always
    pass through to normal processing.
    """
    # Booking-related intents are never off-topic
    if intent in ("find_ticket", "single_ticket", "return_ticket"):
        return False
    # Only flag as off-topic for fallback intent
    if intent != "fallback":
        return False

    lower = user_text.lower().strip()

    # Explicit question mark
    if "?" in user_text:
        return True

    # Starts with a word that cannot begin any UK station name
    return any(lower.startswith(q) for q in _QUESTION_STARTERS)

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------
GREETING = "GREETING"
COLLECT_ORIGIN = "COLLECT_ORIGIN"
CONFIRM_ORIGIN = "CONFIRM_ORIGIN"
COLLECT_DESTINATION = "COLLECT_DESTINATION"
CONFIRM_DESTINATION = "CONFIRM_DESTINATION"
COLLECT_DATE = "COLLECT_DATE"
COLLECT_TIME = "COLLECT_TIME"
COLLECT_TICKET_TYPE = "COLLECT_TICKET_TYPE"
COLLECT_RETURN_DATE = "COLLECT_RETURN_DATE"
COLLECT_RETURN_TIME = "COLLECT_RETURN_TIME"
CONFIRM = "CONFIRM"
SEARCHING = "SEARCHING"
RESULTS = "RESULTS"
DONE = "DONE"


def confirm_station(state: dict, field: str, crs: str, name: str) -> tuple[str, dict]:
    """
    Called by the UI dropdown when the user picks a station.
    Sets the station directly and advances the conversation.
    """
    state["pending_station"] = None
    state["awaiting_clarification"] = None
    if field == "origin":
        state["origin_crs"] = crs
        state["origin_name"] = name
        # If a destination was captured in the same message, carry it forward
        if state.get("_pending_dest"):
            d_crs, d_name, d_score = state.pop("_pending_dest")
            if d_score == 100:
                state["dest_crs"] = d_crs
                state["dest_name"] = d_name
            else:
                from chatbot.station_lookup import lookup_station
                _, d_candidates = lookup_station(d_name)
                return _ask_station_confirm(
                    "destination", d_crs, d_name,
                    d_candidates if d_candidates else [(d_name, d_crs, d_score)],
                    state,
                )
    else:
        state["dest_crs"] = crs
        state["dest_name"] = name
    return _advance_state(state)


def get_initial_state() -> dict:
    """Return a fresh conversation state dictionary."""
    return {
        "state": GREETING,
        "origin_crs": None,
        "origin_name": None,
        "dest_crs": None,
        "dest_name": None,
        "travel_date": None,
        "travel_time": None,
        "outward_datetime": None,
        "outward_constraint": "departBy",
        "ticket_type": None,
        "return_date": None,
        "return_time": None,
        "return_datetime": None,
        "return_constraint": "departBy",
        "awaiting_clarification": None,   # stores (field, candidates)
        "pending_station": None,              # stores (field, crs, name, candidates) awaiting user confirm
        "search_results": None,
    }


def get_greeting_message() -> str:
    """Return the chatbot's initial greeting."""
    return (
        "Hello! 🚆 Welcome to the **National Rail Ticket Assistant**.\n\n"
        "I can help you find the cheapest train tickets across the UK. "
        "Just tell me about your journey - for example:\n\n"
        '*"I want to travel from Norwich to London"*\n\n'
        "Or simply tell me your departure station to get started!"
    )


def process_message(user_text: str, state: dict, session_id: str) -> tuple[str, dict]:
    """
    Process a user message and return the bot's response + updated state.

    Parameters
    ----------
    user_text  : the raw text the user typed
    state      : current conversation state dict
    session_id : session identifier for DB logging

    Returns
    -------
    (response: str, updated_state: dict)
    """
    # Save user message to DB
    save_message(session_id, "user", user_text)

    current_state = state["state"]
    response = ""

    # ------------------------------------------------------------------
    # Handle clarification responses (user picking from candidates)
    # ------------------------------------------------------------------
    if state.get("awaiting_clarification"):
        response, state = _handle_clarification(user_text, state)
        save_message(session_id, "assistant", response)
        return response, state

    # ------------------------------------------------------------------
    # Handle station confirmation (user confirming a matched station)
    # ------------------------------------------------------------------
    if state.get("pending_station"):
        response, state = _handle_station_confirmation(user_text, state)
        save_message(session_id, "assistant", response)
        return response, state

    # ------------------------------------------------------------------
    # GREETING — first message from user
    # ------------------------------------------------------------------
    if current_state == GREETING:
        response, state = _handle_greeting(user_text, state)

    # ------------------------------------------------------------------
    # COLLECT_ORIGIN — waiting for departure station
    # ------------------------------------------------------------------
    elif current_state == COLLECT_ORIGIN:
        response, state = _handle_collect_origin(user_text, state)

    # ------------------------------------------------------------------
    # COLLECT_DESTINATION — waiting for destination station
    # ------------------------------------------------------------------
    elif current_state == COLLECT_DESTINATION:
        response, state = _handle_collect_destination(user_text, state)

    # ------------------------------------------------------------------
    # COLLECT_DATE — waiting for travel date
    # ------------------------------------------------------------------
    elif current_state == COLLECT_DATE:
        response, state = _handle_collect_date(user_text, state)

    # ------------------------------------------------------------------
    # COLLECT_TIME — waiting for travel time
    # ------------------------------------------------------------------
    elif current_state == COLLECT_TIME:
        response, state = _handle_collect_time(user_text, state)

    # ------------------------------------------------------------------
    # COLLECT_TICKET_TYPE — single or return
    # ------------------------------------------------------------------
    elif current_state == COLLECT_TICKET_TYPE:
        response, state = _handle_collect_ticket_type(user_text, state)

    # ------------------------------------------------------------------
    # COLLECT_RETURN_DATE
    # ------------------------------------------------------------------
    elif current_state == COLLECT_RETURN_DATE:
        response, state = _handle_collect_return_date(user_text, state)

    # ------------------------------------------------------------------
    # COLLECT_RETURN_TIME
    # ------------------------------------------------------------------
    elif current_state == COLLECT_RETURN_TIME:
        response, state = _handle_collect_return_time(user_text, state)

    # ------------------------------------------------------------------
    # CONFIRM — user confirms or denies the summary
    # ------------------------------------------------------------------
    elif current_state == CONFIRM:
        response, state = _handle_confirm(user_text, state)

    # ------------------------------------------------------------------
    # SEARCHING — API call was interrupted; recover gracefully
    # ------------------------------------------------------------------
    elif current_state == SEARCHING:
        if state.get("search_results"):
            state["state"] = RESULTS
            results = state["search_results"]
            if results.get("success"):
                response = _format_results(results, state)
            else:
                error_msg = results.get("error", "Unknown error")
                response = (
                    f"❌ Sorry, I couldn't find any tickets. {error_msg}\n\n"
                    "Would you like to try a different journey? Just say *yes* to start again."
                )
        else:
            state["state"] = CONFIRM
            response = "It looks like your search was interrupted. Shall we try again?"

    # ------------------------------------------------------------------
    # RESULTS / DONE — after results shown
    # ------------------------------------------------------------------
    elif current_state in (RESULTS, DONE):
        response, state = _handle_post_results(user_text, state)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    else:
        response = get_fallback(user_text)

    # Save assistant response to DB
    save_message(session_id, "assistant", response)
    return response, state


# ===================================================================
# State handler functions
# ===================================================================

def _handle_greeting(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle the user's first message — try to extract as much info as possible."""
    intent = classify_intent(user_text)

    # If they said hello, respond with greeting and ask for origin
    if intent == "greet":
        state["state"] = COLLECT_ORIGIN
        action, rule_response = infer_next_action(state)
        return (
            "Hello! Great to have you here. 😊\n\n" + rule_response
        ), state

    # If they ask for help
    if intent == "help":
        resp = handle_intent("help", user_text, state)
        state["state"] = COLLECT_ORIGIN
        return resp, state

    # They might have given journey info right away — try to extract everything
    return _extract_all_from_message(user_text, state)


def _extract_all_from_message(user_text: str, state: dict) -> tuple[str, dict]:
    """
    Try to extract all possible entities (stations, date, time, ticket type)
    from a single message.  Then advance to the next missing field.
    """
    # Extract stations
    station_info = extract_stations(user_text)

    # Also extract date/time and ticket type before possibly branching to confirmation
    dt_str, constraint = extract_datetime(user_text)
    if dt_str:
        state["outward_datetime"] = dt_str
        state["outward_constraint"] = constraint
        state["travel_date"] = dt_str[:10]
        time_part = dt_str[11:]
        if time_part and time_part != "00:00:00":
            state["travel_time"] = time_part

    tt = extract_ticket_type(user_text)
    if tt:
        state["ticket_type"] = tt

    # Handle ambiguous stations first
    if station_info["ambiguous"]:
        field, candidates = station_info["ambiguous"][0]
        state["awaiting_clarification"] = (field, candidates)
        options = "\n".join(
            [f"  {i+1}. **{name}** ({crs})" for i, (name, crs, score) in enumerate(candidates)]
        )
        return (
            f"I found a few stations that could match. Did you mean:\n{options}\n\n"
            "Please type the number or the station name."
        ), state

    # Decide strategy: if text has journey keywords, use NLP first; otherwise direct lookup
    text_lower = user_text.lower()
    has_keywords = any(kw in text_lower for kw in ("from ", " to ", "travel", "train", "ticket", "go to", "going to"))

    if has_keywords:
        # NLP-extracted stations (handles full sentences like "from X to Y")
        if station_info["origin"]:
            crs, name, score = station_info["origin"]
            if station_info["destination"]:
                state["_pending_dest"] = station_info["destination"]
            from chatbot.station_lookup import lookup_station
            _, candidates = lookup_station(name)
            return _ask_station_confirm("origin", crs, name, candidates if candidates else [(name, crs, score)], state)

        if station_info["destination"]:
            crs, name, score = station_info["destination"]
            from chatbot.station_lookup import lookup_station
            _, candidates = lookup_station(name)
            return _ask_station_confirm("destination", crs, name, candidates if candidates else [(name, crs, score)], state)

    # Guard: if the message looks like an off-topic question (e.g. "What's the
    # weather like?" or "Is the train running today?"), don't attempt fuzzy
    # station matching on the raw text — RapidFuzz would otherwise return a
    # spurious match like "Wetherby" or "The Hawthorns".
    # Note: we run this check even when has_keywords=True because a word like
    # "train" appearing in an off-topic question should not trigger lookup.
    intent = classify_intent(user_text)
    if _is_off_topic_query(user_text, intent):
        state["state"] = COLLECT_ORIGIN
        return get_fallback(user_text), state

    # Direct lookup — treats the whole text as a station name
    station_direct, direct_candidates = extract_station_direct(user_text)
    if station_direct:
        crs, name, score = station_direct
        return _ask_station_confirm("origin", crs, name, direct_candidates if direct_candidates else [(name, crs, score)], state)

    # Final fallback: NLP results even without keywords
    if not has_keywords:
        if station_info["origin"]:
            crs, name, score = station_info["origin"]
            if station_info["destination"]:
                state["_pending_dest"] = station_info["destination"]
            from chatbot.station_lookup import lookup_station
            _, candidates = lookup_station(name)
            return _ask_station_confirm("origin", crs, name, candidates if candidates else [(name, crs, score)], state)

    # Determine next question
    return _advance_state(state)


def _advance_state(state: dict) -> tuple[str, dict]:
    """
    Check what info is still missing and advance the conversation state.
    Returns the next question for the user.
    """
    acknowledgment = ""

    # Build acknowledgment of what we just learned
    if state.get("origin_name") and state["state"] in (GREETING, COLLECT_ORIGIN):
        acknowledgment += f"✅ Departing from **{state['origin_name']}**\n"
    if state.get("dest_name") and state["state"] in (GREETING, COLLECT_ORIGIN, COLLECT_DESTINATION):
        acknowledgment += f"✅ Travelling to **{state['dest_name']}**\n"

    # Determine next missing field using the reasoning engine
    action, rule_response = infer_next_action(state)

    # Map actions to states
    action_to_state = {
        "ask_origin": COLLECT_ORIGIN,
        "ask_destination": COLLECT_DESTINATION,
        "ask_date": COLLECT_DATE,
        "ask_time": COLLECT_TIME,
        "ask_ticket_type": COLLECT_TICKET_TYPE,
        "ask_return_date": COLLECT_RETURN_DATE,
        "ask_return_time": COLLECT_RETURN_TIME,
        "confirm_details": CONFIRM,
    }

    new_state = action_to_state.get(action, CONFIRM)
    state["state"] = new_state

    if new_state == CONFIRM:
        return _build_confirmation(state), state

    if acknowledgment:
        return acknowledgment + "\n" + rule_response, state
    return rule_response, state


def _handle_collect_origin(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle input when we're waiting for the departure station."""
    intent = classify_intent(user_text)

    # Check for non-booking intents first
    intent_response = handle_intent(intent, user_text, state)
    if intent_response and intent not in ("find_ticket", "single_ticket", "return_ticket"):
        return intent_response, state

    # Off-topic guard: "What's the weather like?" should not be fuzzy-matched
    # to a station name (e.g. "Wetherby"). Return fallback + re-ask the question.
    if _is_off_topic_query(user_text, intent):
        return get_fallback(user_text), state

    # Extract date/time and ticket type early
    dt_str, constraint = extract_datetime(user_text)
    if dt_str:
        state["outward_datetime"] = dt_str
        state["outward_constraint"] = constraint
        state["travel_date"] = dt_str[:10]
        time_part = dt_str[11:]
        if time_part and time_part != "00:00:00":
            state["travel_time"] = time_part
    tt = extract_ticket_type(user_text)
    if tt:
        state["ticket_type"] = tt

    # Try direct station lookup FIRST (user likely typed a station name)
    # This avoids spaCy NER splitting multi-word names like "london liverpool street"
    station, candidates = extract_station_direct(user_text)
    if station:
        crs, name, score = station
        return _ask_station_confirm("origin", crs, name, candidates if candidates else [(name, crs, score)], state)

    # Try full NLP extraction (handles "from X to Y" patterns)
    station_info = extract_stations(user_text)
    if station_info["origin"]:
        crs, name, score = station_info["origin"]
        # Store destination for later if found
        if station_info["destination"]:
            state["_pending_dest"] = station_info["destination"]
        from chatbot.station_lookup import lookup_station
        _, cands = lookup_station(name)
        return _ask_station_confirm("origin", crs, name, cands if cands else [(name, crs, score)], state)

    # Surface NLP ambiguity (stations found but uncertain which one)
    if station_info.get("ambiguous"):
        field_amb, amb_candidates = station_info["ambiguous"][0]
        if field_amb == "origin":
            state["awaiting_clarification"] = ("origin", amb_candidates)
            return (
                "I found a few stations that could match your departure. "
                "Please select one from the list below, or type the station name again."
            ), state

    # Ambiguous match from direct lookup
    if candidates:
        state["awaiting_clarification"] = ("origin", candidates)
        return (
            "I found a few stations that could match. "
            "Please select one from the list below, or type the station name again."
        ), state

    return "I couldn't find that station. Could you try again? For example: *Norwich* or *London Kings Cross*.", state


def _handle_collect_destination(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle input when we're waiting for the destination station."""
    intent = classify_intent(user_text)

    intent_response = handle_intent(intent, user_text, state)
    if intent_response and intent not in ("find_ticket", "single_ticket", "return_ticket"):
        return intent_response, state

    # Off-topic guard: same as in _handle_collect_origin
    if _is_off_topic_query(user_text, intent):
        return get_fallback(user_text), state

    # Extract date/time and ticket type early
    dt_str, constraint = extract_datetime(user_text)
    if dt_str:
        state["outward_datetime"] = dt_str
        state["outward_constraint"] = constraint
        state["travel_date"] = dt_str[:10]
        time_part = dt_str[11:]
        if time_part and time_part != "00:00:00":
            state["travel_time"] = time_part
    tt = extract_ticket_type(user_text)
    if tt:
        state["ticket_type"] = tt

    # Try direct station lookup FIRST (user likely typed a station name)
    station, candidates = extract_station_direct(user_text)
    if station:
        crs, name, score = station
        if crs == state.get("origin_crs"):
            return "That's the same as your departure station! Please choose a different destination.", state
        return _ask_station_confirm("destination", crs, name, candidates if candidates else [(name, crs, score)], state)

    # Try full NLP extraction (handles "to X" patterns)
    station_info = extract_stations(user_text)
    dest_info = station_info["destination"] or station_info["origin"]
    if dest_info:
        crs, name, score = dest_info
        if crs == state.get("origin_crs"):
            return "That's the same as your departure station! Please choose a different destination.", state
        from chatbot.station_lookup import lookup_station
        _, cands = lookup_station(name)
        return _ask_station_confirm("destination", crs, name, cands if cands else [(name, crs, score)], state)

    # Surface NLP ambiguity for destination
    if station_info.get("ambiguous"):
        for field_amb, amb_candidates in station_info["ambiguous"]:
            if field_amb == "destination":
                state["awaiting_clarification"] = ("destination", amb_candidates)
                return (
                    "I found a few stations that could match your destination. "
                    "Please select one from the list below, or type the station name again."
                ), state

    if candidates:
        state["awaiting_clarification"] = ("destination", candidates)
        return (
            "I found a few stations that could match. "
            "Please select one from the list below, or type the station name again."
        ), state

    return "I couldn't find that station. Could you try again with the full station name?", state


def _handle_collect_date(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle input when we're waiting for the travel date (and optionally time)."""
    dt_str, constraint = extract_datetime(user_text)
    if dt_str:
        if _dt_now.fromisoformat(dt_str) < _dt_now.now():
            return (
                "That date/time is in the past. Please enter a future date and time"
            ), state
        state["outward_datetime"] = dt_str
        state["outward_constraint"] = constraint
        state["travel_date"] = dt_str[:10]
        # Use the parsed time, or default to 09:00:00 if only a date was given
        time_part = dt_str[11:]
        state["travel_time"] = time_part if (time_part and time_part != "00:00:00") else "09:00:00"
        # Rebuild outward_datetime with the final time
        state["outward_datetime"] = state["travel_date"] + "T" + state["travel_time"]
        return _advance_state(state)

    return (
        "I couldn't understand that. Could you try again with a date and time? "
        "For example: *next Friday at 9am*, *15th July at 14:30*, or *tomorrow morning*."
    ), state


def _handle_collect_time(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle input when we're waiting for the travel time."""
    dt_str, constraint = extract_datetime(user_text)
    if dt_str:
        try:
            time_part = dt_str[11:] if len(dt_str) > 11 else "09:00:00"
            combined = (state["travel_date"] + "T" + time_part) if state.get("travel_date") else dt_str
            parsed = _dt_now.fromisoformat(combined)
        except (ValueError, IndexError):
            return (
                "I couldn't understand that time. Could you try something like "
                "*9am*, *after 10am*, *before 2pm*, or *14:30*?"
            ), state
        if parsed < _dt_now.now():
            return (
                "That time has already passed today. Please enter a future time — "
                "for example: *9am*, *after 2pm*, or *14:30*."
            ), state
        state["outward_constraint"] = constraint
        state["outward_datetime"] = combined
        state["travel_time"] = time_part
        return _advance_state(state)

    return (
        "I couldn't understand that time. Could you try something like "
        "*9am*, *after 10am*, *before 2pm*, or *14:30*?"
    ), state


def _handle_collect_ticket_type(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle input when we're waiting for single/return choice."""
    tt = extract_ticket_type(user_text)
    if tt:
        state["ticket_type"] = tt
        return _advance_state(state)

    # Also check raw keywords as backup
    lower = user_text.lower()
    if "single" in lower or "one way" in lower or "one-way" in lower:
        state["ticket_type"] = "single"
        return _advance_state(state)
    elif "return" in lower or "round" in lower or "come back" in lower or "both ways" in lower:
        state["ticket_type"] = "return"
        return _advance_state(state)

    return "Would you like a **single** (one-way) or **return** (round-trip) ticket?", state


def _handle_collect_return_date(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle input for return date (and optionally time)."""
    # Use outward date as the relative base so 'two days later' works correctly
    relative_base = None
    if state.get("outward_datetime"):
        try:
            relative_base = _dt_now.fromisoformat(state["outward_datetime"])
        except (ValueError, TypeError):
            pass
    dt_str, constraint = extract_datetime(user_text, relative_base=relative_base)
    if dt_str:
        parsed = _dt_now.fromisoformat(dt_str)
        if parsed < _dt_now.now():
            return (
                "That return date/time is in the past. Please enter a future date and time — "
                "for example: *two days later at 3pm*, *next Monday at 10am*, or *July 17th*."
            ), state
        # Return must be on or after the outward date
        if state.get("outward_datetime"):
            try:
                outward = _dt_now.fromisoformat(state["outward_datetime"])
                if parsed.date() < outward.date():
                    out_str = outward.strftime("%A %d %B")
                    return (
                        f"Your return date can't be before your outward date ({out_str}). "
                        "When would you like to return? Include a time too — e.g. *next Monday at 3pm*."
                    ), state
            except (ValueError, TypeError):
                pass
        state["return_date"] = dt_str[:10]
        time_part = dt_str[11:]
        # Default to 09:00:00 if only a date was given
        state["return_time"] = time_part if (time_part and time_part != "00:00:00") else "09:00:00"
        state["return_constraint"] = constraint
        state["return_datetime"] = state["return_date"] + "T" + state["return_time"]
        return _advance_state(state)

    return (
        "I couldn't understand that. When would you like to return? "
        "Try: *two days later at 3pm*, *July 17th at 10am*, or *next Monday morning*."
    ), state


def _handle_collect_return_time(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle input for return time."""
    # Use return date as relative base if available
    relative_base = None
    if state.get("return_date"):
        try:
            relative_base = _dt_now.fromisoformat(state["return_date"])
        except (ValueError, TypeError):
            pass
    dt_str, constraint = extract_datetime(user_text, relative_base=relative_base)
    if dt_str:
        try:
            time_part = dt_str[11:] if len(dt_str) > 11 else "09:00:00"
            combined = (state["return_date"] + "T" + time_part) if state.get("return_date") else dt_str
            parsed = _dt_now.fromisoformat(combined)
        except (ValueError, IndexError):
            return (
                "I couldn't understand that time. What time would you like to return? "
                "Try: *2pm*, *after 14:00*, *in the afternoon*."
            ), state
        if parsed < _dt_now.now():
            return (
                "That return time has already passed. Please enter a future time — "
                "for example: *2pm*, *after 14:00*, or *in the afternoon*."
            ), state
        state["return_constraint"] = constraint
        state["return_datetime"] = combined
        state["return_time"] = time_part
        return _advance_state(state)

    return (
        "I couldn't understand that time. What time would you like to return? "
        "Try: *2pm*, *after 14:00*, *in the afternoon*."
    ), state


def _handle_confirm(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle confirmation or denial of journey details."""
    intent = classify_intent(user_text)

    if intent == "affirm":
        state["state"] = SEARCHING
        # Perform the search
        results = search_cheapest_ticket(
            origin_crs=state["origin_crs"],
            dest_crs=state["dest_crs"],
            outward_datetime=state["outward_datetime"],
            outward_constraint=state["outward_constraint"],
            inward_datetime=state.get("return_datetime"),
            inward_constraint=state.get("return_constraint", "departBy"),
        )
        state["search_results"] = results
        state["state"] = RESULTS

        if results["success"]:
            return _format_results(results, state), state
        else:
            error_msg = results.get("error", "Unknown error")
            return (
                f"❌ Sorry, I couldn't find any tickets. {error_msg}\n\n"
                "Would you like to try a different journey? Just say *yes* to start again."
            ), state

    elif intent == "deny":
        # Reset and start over
        new_state = get_initial_state()
        new_state["state"] = COLLECT_ORIGIN
        return (
            "No problem! Let's start over.\n\n"
            "Which station will you be departing from?"
        ), new_state

    return "Please confirm: is this correct? Reply **yes** to search or **no** to start over.", state


def _handle_post_results(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle messages after results have been shown."""
    intent = classify_intent(user_text)
    lower = user_text.lower().strip()

    # "yes / new / another / again" → start a fresh booking
    if intent == "affirm" or any(kw in lower for kw in ("new", "another", "again")):
        new_state = get_initial_state()
        new_state["state"] = COLLECT_ORIGIN
        return (
            "Let's plan another journey! Which station will you be departing from?"
        ), new_state

    # Booking-related intent — restart straight into a new search
    if intent in ("find_ticket", "single_ticket", "return_ticket"):
        new_state = get_initial_state()
        new_state["state"] = COLLECT_ORIGIN
        # Run the new turn through the origin handler so any stations
        # mentioned in the same message are captured
        response, new_state = _handle_collect_origin(user_text, new_state)
        return response, new_state

    # "no / cancel / stop / done" → end the conversation gracefully
    if intent == "deny" or lower in ("done", "im done", "i'm done",
                                      "that's all", "thats all", "nothing",
                                      "no thanks", "no thank you"):
        state["state"] = DONE
        return (
            "No problem — thanks for using the National Rail Ticket Assistant! "
            "Have a great day. 🚆\n\n"
            "If you change your mind, just type a new request and I'll start a new search."
        ), state

    # Explicit goodbye intent → same as deny
    if intent == "goodbye":
        state["state"] = DONE
        return "Goodbye! Have a wonderful journey! 🚆", state

    # FAQ or other passive intents (help, etc.) — answer then re-offer
    intent_response = handle_intent(intent, user_text, state)
    if intent_response:
        return (
            intent_response
            + "\n\nIs there anything else I can help with? Say **yes** to plan a new "
              "journey or **no** if you're done."
        ), state

    # Fallback — unrecognised input after results
    return (
        "Would you like to search for another ticket? "
        "Say **yes** to start a new search, **no** to finish, or ask me a question."
    ), state


def _handle_done(user_text: str, state: dict) -> tuple[str, dict]:
    """
    Handle messages after the user has chosen to end the conversation.
    Any input restarts the booking flow.
    """
    lower = user_text.lower().strip()
    intent = classify_intent(user_text)

    # If the user just says goodbye again, stay in DONE
    if intent == "goodbye":
        return "Take care! 🚆", state

    # Anything else — restart cleanly
    new_state = get_initial_state()
    new_state["state"] = COLLECT_ORIGIN
    return (
        "Welcome back! Let's plan a new journey. "
        "Which station will you be departing from?"
    ), new_state


def _handle_clarification(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle the user's response when we asked them to clarify a station."""
    field, candidates = state["awaiting_clarification"]
    state["awaiting_clarification"] = None

    # Try direct lookup with their text
    station, new_candidates = extract_station_direct(user_text)
    if station:
        crs, name, score = station
        if field == "origin":
            state["origin_crs"] = crs
            state["origin_name"] = name
        else:
            state["dest_crs"] = crs
            state["dest_name"] = name
        return _advance_state(state)

    # Still can't resolve
    return (
        "I still couldn't match that station. Could you try the full station name? "
        "For example: *London Kings Cross* or *Birmingham New Street*."
    ), state


# ===================================================================
# Station confirmation helpers
# ===================================================================

def _ask_station_confirm(field: str, crs: str, name: str, candidates: list, state: dict) -> tuple[str, dict]:
    """
    Present the matched station and ask the user to confirm.
    Stores the pending station info in state for follow-up.
    If there are multiple candidates, the UI will show a dropdown.
    """
    state["pending_station"] = {
        "field": field,
        "crs": crs,
        "name": name,
        "candidates": candidates,
    }

    label = "departure" if field == "origin" else "destination"
    msg = f"I found **{name}** ({crs}) as your {label} station. Is that correct?\n\n"
    if len(candidates) > 1:
        msg += "If not, select a different station from the dropdown below, or type the name again."
    else:
        msg += "Reply **yes** to confirm, or type a different station name."
    return msg, state


def _handle_station_confirmation(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle user response to a station confirmation prompt."""
    pending = state["pending_station"]
    field = pending["field"]
    crs = pending["crs"]
    name = pending["name"]
    candidates = pending["candidates"]
    state["pending_station"] = None

    intent = classify_intent(user_text)

    # User said yes — accept the station
    if intent == "affirm" or user_text.strip().lower() in ("yes", "y", "yeah", "yep", "correct", "right", "1"):
        if field == "origin":
            state["origin_crs"] = crs
            state["origin_name"] = name
            # If we had a pending destination from the same message, confirm it next
            if state.get("_pending_dest"):
                d_crs, d_name, d_score = state.pop("_pending_dest")
                if d_score == 100:
                    state["dest_crs"] = d_crs
                    state["dest_name"] = d_name
                else:
                    from chatbot.station_lookup import lookup_station
                    _, d_candidates = lookup_station(d_name)
                    return _ask_station_confirm("destination", d_crs, d_name, d_candidates if d_candidates else [(d_name, d_crs, d_score)], state)
        else:
            state["dest_crs"] = crs
            state["dest_name"] = name
        return _advance_state(state)

    # User typed a number to pick from candidates
    text_stripped = user_text.strip()
    if text_stripped.isdigit():
        idx = int(text_stripped) - 1
        if 0 <= idx < len(candidates):
            picked_name, picked_crs, picked_score = candidates[idx]
            if field == "origin":
                state["origin_crs"] = picked_crs
                state["origin_name"] = picked_name
            else:
                state["dest_crs"] = picked_crs
                state["dest_name"] = picked_name
            return _advance_state(state)

    # User said no — ask again
    if intent == "deny" or user_text.strip().lower() in ("no", "n", "nope", "wrong"):
        if field == "origin":
            state["state"] = COLLECT_ORIGIN
            return "No problem! What's your departure station?", state
        else:
            state["state"] = COLLECT_DESTINATION
            return "No problem! What's your destination station?", state

    # User typed a new station name — try to match it
    station, new_candidates = extract_station_direct(user_text)
    if station:
        new_crs, new_name, new_score = station
        # If exact match, accept directly
        if new_score == 100:
            if field == "origin":
                state["origin_crs"] = new_crs
                state["origin_name"] = new_name
            else:
                state["dest_crs"] = new_crs
                state["dest_name"] = new_name
            return _advance_state(state)
        # Otherwise, ask to confirm this new match
        return _ask_station_confirm(field, new_crs, new_name, new_candidates, state)

    return "I couldn't find that station. Could you try again with the full station name?", state


# ===================================================================
# Formatting helpers
# ===================================================================

def _build_confirmation(state: dict) -> str:
    """Build a confirmation summary of the journey details."""
    from datetime import datetime as dt

    ticket_label = "Return" if state.get("ticket_type") == "return" else "Single"
    origin = f"{state.get('origin_name', 'Unknown')} ({state.get('origin_crs', '?')})"
    dest   = f"{state.get('dest_name', 'Unknown')} ({state.get('dest_crs', '?')})"

    rows = ["**Here's your journey summary — does everything look right?**", ""]
    rows.append(f"| | |")
    rows.append(f"|---|---|")
    rows.append(f"| 🚉 **From** | {origin} |")
    rows.append(f"| 🚉 **To** | {dest} |")
    rows.append(f"| 🎫 **Ticket** | {ticket_label} |")

    if state.get("outward_datetime"):
        try:
            out_dt = dt.fromisoformat(state["outward_datetime"])
            date_str = out_dt.strftime("%A %d %B %Y")
            time_str = out_dt.strftime("%H:%M")
            label = "Arrive by" if state.get("outward_constraint") == "arriveBy" else "Depart at"
            rows.append(f"| 📅 **Outward** | {date_str} · {label} {time_str} |")
        except (ValueError, TypeError):
            rows.append(f"| 📅 **Outward** | {state.get('outward_datetime', '?')} |")

    if state.get("ticket_type") == "return" and state.get("return_datetime"):
        try:
            ret_dt = dt.fromisoformat(state["return_datetime"])
            date_str = ret_dt.strftime("%A %d %B %Y")
            time_str = ret_dt.strftime("%H:%M")
            label = "Arrive by" if state.get("return_constraint") == "arriveBy" else "Depart at"
            rows.append(f"| 📅 **Return** | {date_str} · {label} {time_str} |")
        except (ValueError, TypeError):
            rows.append(f"| 📅 **Return** | {state.get('return_datetime', '?')} |")

    rows.append("")
    rows.append("Reply **yes** to search or **no** to start over.")

    return "\n".join(rows)


def _append_legs_detail(lines: list, journey: dict) -> None:
    """Show per-leg changes and warn when a ticket switch is likely needed."""
    changes = journey.get("changes", 0)
    change_stations = journey.get("change_stations", [])
    legs_detail = journey.get("legs_detail", [])

    if changes == 0:
        lines.append("Direct train")
        return

    # Build full station sequence from the already-working change_stations list
    origin = journey.get("origin", "?")
    destination = journey.get("destination", "?")
    stations = [origin] + change_stations + [destination]
    operators = [leg.get("operator") for leg in legs_detail]

    lines.append("")  # blank line before legs
    for i in range(len(stations) - 1):
        op = operators[i] if i < len(operators) else None
        op_str = f" *({op})*" if op else ""
        lines.append(f"- {stations[i]} → {stations[i + 1]}{op_str}")

    unique_ops = list(dict.fromkeys(op for op in operators if op))
    if len(unique_ops) >= 2:
        lines.append(f"\n⚠️ **Two tickets likely needed** — this journey switches between "
                     f"{' and '.join(unique_ops)}. Book each leg separately.")
    elif changes >= 1:
        lines.append("\nOne through ticket covers the full journey.")


def _format_results(results: dict, state: dict) -> str:
    """Format the API results into a readable summary for the person booking."""
    fare = results["cheapest_fare"]
    outward = results.get("outward")
    inward = results.get("inward")
    booking_url = results.get("booking_url", "")

    fare_type = fare["fare_type"]
    # Collapse "X + X" duplicates e.g. "Advance (Standard Class) + Advance (Standard Class)"
    parts = [p.strip() for p in fare_type.split("+")]
    if len(parts) == 2 and parts[0] == parts[1]:
        fare_type = parts[0]

    lines = [f"**Cheapest fare: £{fare['total_price']:.2f}** ({fare_type})\n"]

    if fare.get("return_fare_missing"):
        lines.append(
            "⚠️ No return fare was found — the price shown is for the outward journey only. "
            "Please check the operator's website for return pricing.\n"
        )

    if outward:
        dep = outward.get("departure_time", "?")
        arr = outward.get("arrival_time", "?")
        date = outward.get("departure_date", "")
        dur = outward.get("duration", "")
        header = f"**Outward** — {dep} → {arr}"
        if date:
            header += f"  ({date})"
        if dur:
            header += f"  ·  {dur}"
        lines.append(header)
        _append_legs_detail(lines, outward)

    if inward:
        lines.append("")
        dep = inward.get("departure_time", "?")
        arr = inward.get("arrival_time", "?")
        date = inward.get("departure_date", "")
        dur = inward.get("duration", "")
        header = f"**Return** — {dep} → {arr}"
        if date:
            header += f"  ({date})"
        if dur:
            header += f"  ·  {dur}"
        lines.append(header)
        _append_legs_detail(lines, inward)

    if booking_url:
        lines.append(f"\n🔗 [Book now]({booking_url})")

    lines.append("\nWould you like to search for another journey?")

    return "\n".join(lines)
