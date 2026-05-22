"""
conversation.py  (Task 2)
-------------------------
Dialogue manager for the Train Delay Prediction assistant.

Collects the following information step-by-step:
  1. Current station the train is at
  2. Destination station
  3. Current delay in minutes
  4. Planned arrival time at destination
  5. Date of travel (for day_of_week and month features)

Then calls predict_arrival_time() and returns a friendly answer.

State is stored in a plain dict so it works with Streamlit session_state.
"""

import re
import logging
from datetime import datetime, date

import dateparser

from task2.predict import predict_arrival_time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation states
# ---------------------------------------------------------------------------
T2_GREETING          = "T2_GREETING"
T2_COLLECT_STATION   = "T2_COLLECT_STATION"      # current station
T2_CONFIRM_STATION   = "T2_CONFIRM_STATION"       # confirm current station
T2_COLLECT_DEST      = "T2_COLLECT_DEST"           # destination station
T2_CONFIRM_DEST      = "T2_CONFIRM_DEST"           # confirm destination
T2_COLLECT_DELAY     = "T2_COLLECT_DELAY"          # delay in minutes
T2_COLLECT_ARRIVAL   = "T2_COLLECT_ARRIVAL"        # planned arrival time at dest
T2_COLLECT_DATE      = "T2_COLLECT_DATE"           # date of travel
T2_RESULT            = "T2_RESULT"                 # prediction delivered
T2_DONE              = "T2_DONE"


def get_initial_state() -> dict:
    """Return a fresh Task 2 conversation state."""
    return {
        "state": T2_GREETING,
        "current_station_crs": None,
        "current_station_name": None,
        "dest_crs": None,
        "dest_name": None,
        "delay_minutes": None,
        "planned_arrival": None,       # "HH:MM"
        "travel_date": None,           # date object
        "pending_station": None,       # for confirmation flow
    }


def get_greeting_message() -> str:
    """Return the opening message for Task 2."""
    return (
        "🚆 **Train Delay Predictor**\n\n"
        "I can predict when your delayed train will actually arrive.\n\n"
        "Which station is your train currently at? "
        "You can type the station name or its 3-letter code (e.g. *SOU* or *Southampton*)."
    )


# ---------------------------------------------------------------------------
# Station lookup helper — reuses Task 1's station_lookup module
# ---------------------------------------------------------------------------
def _lookup_station(text: str):
    """
    Try to resolve user text to a CRS code.
    Returns (crs, name, score, candidates) or (None, None, 0, candidates).
    """
    from chatbot.station_lookup import lookup_station
    from chatbot.nlp import extract_station_direct

    # First try direct lookup (handles full station names well)
    station, candidates = extract_station_direct(text)
    if station:
        crs, name, score = station
        return crs, name, score, candidates

    # Fallback: raw lookup
    crs, candidates = lookup_station(text)
    if crs:
        # Get name from candidates
        name = text.title()
        if candidates:
            name = candidates[0][0]
        return crs, name, 90, candidates

    return None, None, 0, candidates or []


# ---------------------------------------------------------------------------
# Main message processor
# ---------------------------------------------------------------------------
def process_message(user_text: str, state: dict) -> tuple[str, dict]:
    """
    Process a user message in the Task 2 conversation.

    Parameters
    ----------
    user_text : the message the user typed
    state     : current conversation state dict

    Returns
    -------
    (response_text, updated_state)
    """
    user_text = user_text.strip()
    current = state["state"]

    # Dispatch to the right handler
    if current == T2_GREETING:
        return _handle_greeting(user_text, state)
    elif current == T2_COLLECT_STATION:
        return _handle_collect_station(user_text, state)
    elif current == T2_CONFIRM_STATION:
        return _handle_confirm_station(user_text, state)
    elif current == T2_COLLECT_DEST:
        return _handle_collect_dest(user_text, state)
    elif current == T2_CONFIRM_DEST:
        return _handle_confirm_dest(user_text, state)
    elif current == T2_COLLECT_DELAY:
        return _handle_collect_delay(user_text, state)
    elif current == T2_COLLECT_ARRIVAL:
        return _handle_collect_arrival(user_text, state)
    elif current == T2_COLLECT_DATE:
        return _handle_collect_date(user_text, state)
    elif current in (T2_RESULT, T2_DONE):
        return _handle_done(user_text, state)
    else:
        state["state"] = T2_COLLECT_STATION
        return _handle_collect_station(user_text, state)


# ---------------------------------------------------------------------------
# State handlers
# ---------------------------------------------------------------------------

def _handle_greeting(user_text: str, state: dict) -> tuple[str, dict]:
    """First message — try to extract a station, otherwise ask."""
    state["state"] = T2_COLLECT_STATION
    return _handle_collect_station(user_text, state)


def _handle_collect_station(user_text: str, state: dict) -> tuple[str, dict]:
    """Collect the station the train is currently at."""
    crs, name, score, candidates = _lookup_station(user_text)

    if crs:
        state["pending_station"] = {
            "field": "current",
            "crs": crs,
            "name": name,
            "candidates": candidates,
        }
        state["state"] = T2_CONFIRM_STATION
        resp = f"I found **{name}** ({crs}) as your current station. Is that correct?"
        if candidates and len(candidates) > 1:
            options = "\n".join(
                f"  {i+1}. **{c[0]}** ({c[1]})"
                for i, c in enumerate(candidates[:5])
            )
            resp += f"\n\nOther options:\n{options}"
            resp += "\n\nReply **yes** to confirm, type a **number** to pick another, or type a different station name."
        else:
            resp += "\nReply **yes** to confirm or type a different station name."
        return resp, state

    if candidates:
        options = "\n".join(
            f"  {i+1}. **{c[0]}** ({c[1]})"
            for i, c in enumerate(candidates[:5])
        )
        return (
            f"I found a few stations that could match:\n{options}\n\n"
            "Please type the number or the station name."
        ), state

    return (
        "I couldn't find that station. Please try again with the full station "
        "name or its 3-letter CRS code (e.g. *SOU* for Southampton)."
    ), state


def _handle_confirm_station(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle yes/no/number response for current station confirmation."""
    pending = state.get("pending_station")
    if not pending:
        state["state"] = T2_COLLECT_STATION
        return "Which station is your train currently at?", state

    lower = user_text.lower().strip()

    # User confirms
    if lower in ("yes", "y", "yep", "yeah", "correct", "that's right", "thats right"):
        state["current_station_crs"] = pending["crs"]
        state["current_station_name"] = pending["name"]
        state["pending_station"] = None
        state["state"] = T2_COLLECT_DEST
        return (
            f"✅ Current station: **{pending['name']}** ({pending['crs']})\n\n"
            "Where is your destination? "
            "Type the station name or CRS code."
        ), state

    # User rejects
    if lower in ("no", "n", "nope", "wrong"):
        state["pending_station"] = None
        state["state"] = T2_COLLECT_STATION
        return "No problem. Which station is your train currently at?", state

    # Numeric selection from candidates
    if lower.isdigit():
        idx = int(lower) - 1
        candidates = pending.get("candidates", [])
        if 0 <= idx < len(candidates):
            name, crs = candidates[idx][0], candidates[idx][1]
            state["current_station_crs"] = crs
            state["current_station_name"] = name
            state["pending_station"] = None
            state["state"] = T2_COLLECT_DEST
            return (
                f"✅ Current station: **{name}** ({crs})\n\n"
                "Where is your destination?"
            ), state

    # Treat as a new station name
    state["pending_station"] = None
    state["state"] = T2_COLLECT_STATION
    return _handle_collect_station(user_text, state)


def _handle_collect_dest(user_text: str, state: dict) -> tuple[str, dict]:
    """Collect the destination station."""
    crs, name, score, candidates = _lookup_station(user_text)

    if crs:
        if crs == state.get("current_station_crs"):
            return "That's the same as your current station! Please enter a different destination.", state

        state["pending_station"] = {
            "field": "destination",
            "crs": crs,
            "name": name,
            "candidates": candidates,
        }
        state["state"] = T2_CONFIRM_DEST
        resp = f"I found **{name}** ({crs}) as your destination. Is that correct?"
        if candidates and len(candidates) > 1:
            options = "\n".join(
                f"  {i+1}. **{c[0]}** ({c[1]})"
                for i, c in enumerate(candidates[:5])
            )
            resp += f"\n\nOther options:\n{options}"
            resp += "\n\nReply **yes** to confirm, type a **number** to pick another, or type a different station name."
        else:
            resp += "\nReply **yes** to confirm or type a different station name."
        return resp, state

    if candidates:
        options = "\n".join(
            f"  {i+1}. **{c[0]}** ({c[1]})"
            for i, c in enumerate(candidates[:5])
        )
        return (
            f"I found a few stations that could match:\n{options}\n\n"
            "Please type the number or the station name."
        ), state

    return (
        "I couldn't find that station. Please try again with the full name "
        "or CRS code."
    ), state


def _handle_confirm_dest(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle yes/no/number response for destination confirmation."""
    pending = state.get("pending_station")
    if not pending:
        state["state"] = T2_COLLECT_DEST
        return "Where is your destination?", state

    lower = user_text.lower().strip()

    if lower in ("yes", "y", "yep", "yeah", "correct", "that's right", "thats right"):
        state["dest_crs"] = pending["crs"]
        state["dest_name"] = pending["name"]
        state["pending_station"] = None
        state["state"] = T2_COLLECT_DELAY
        return (
            f"✅ Destination: **{pending['name']}** ({pending['crs']})\n\n"
            "How many minutes is your train currently delayed? "
            "Just type a number (e.g. *10*)."
        ), state

    if lower in ("no", "n", "nope", "wrong"):
        state["pending_station"] = None
        state["state"] = T2_COLLECT_DEST
        return "No problem. Where is your destination?", state

    if lower.isdigit():
        idx = int(lower) - 1
        candidates = pending.get("candidates", [])
        if 0 <= idx < len(candidates):
            name, crs = candidates[idx][0], candidates[idx][1]
            state["dest_crs"] = crs
            state["dest_name"] = name
            state["pending_station"] = None
            state["state"] = T2_COLLECT_DELAY
            return (
                f"✅ Destination: **{name}** ({crs})\n\n"
                "How many minutes is your train currently delayed?"
            ), state

    state["pending_station"] = None
    state["state"] = T2_COLLECT_DEST
    return _handle_collect_dest(user_text, state)


def _handle_collect_delay(user_text: str, state: dict) -> tuple[str, dict]:
    """Collect the current delay in minutes."""
    # Extract a number from the text
    match = re.search(r"(\d+)", user_text)
    if match:
        delay = int(match.group(1))
        if delay < 0 or delay > 300:
            return "That doesn't seem right. Please enter a delay between 0 and 300 minutes.", state
        state["delay_minutes"] = delay
        state["state"] = T2_COLLECT_ARRIVAL
        return (
            f"✅ Current delay: **{delay} minutes**\n\n"
            "What is the **scheduled (planned) arrival time** at your destination? "
            "For example: *14:35* or *2:35pm*."
        ), state

    return (
        "I need a number for the delay. How many minutes is your train delayed? "
        "For example: *10* or *15 minutes*."
    ), state


def _handle_collect_arrival(user_text: str, state: dict) -> tuple[str, dict]:
    """Collect the planned arrival time at the destination."""
    # Try to parse a time from the input
    time_match = re.search(r"(\d{1,2})[:\.](\d{2})", user_text)
    if time_match:
        h, m = int(time_match.group(1)), int(time_match.group(2))
        # Handle PM
        if "pm" in user_text.lower() and h < 12:
            h += 12
        if "am" in user_text.lower() and h == 12:
            h = 0
        if 0 <= h <= 23 and 0 <= m <= 59:
            state["planned_arrival"] = f"{h:02d}:{m:02d}"
            state["state"] = T2_COLLECT_DATE
            return (
                f"✅ Planned arrival: **{h:02d}:{m:02d}**\n\n"
                "What date are you travelling? "
                "You can say *today*, *tomorrow*, or a specific date like *14 April 2025*."
            ), state

    # Try dateparser for expressions like "2:35pm"
    parsed = dateparser.parse(user_text, settings={"PREFER_DATES_FROM": "future"})
    if parsed and parsed.hour != 0:
        h, m = parsed.hour, parsed.minute
        state["planned_arrival"] = f"{h:02d}:{m:02d}"
        state["state"] = T2_COLLECT_DATE
        return (
            f"✅ Planned arrival: **{h:02d}:{m:02d}**\n\n"
            "What date are you travelling? "
            "You can say *today*, *tomorrow*, or a specific date."
        ), state

    return (
        "I couldn't understand that time. Please enter the scheduled arrival "
        "time at your destination, e.g. *14:35* or *2:35pm*."
    ), state


def _handle_collect_date(user_text: str, state: dict) -> tuple[str, dict]:
    """Collect the date of travel."""
    lower = user_text.lower().strip()

    # Quick shortcuts
    if lower == "today":
        d = date.today()
    elif lower == "tomorrow":
        from datetime import timedelta
        d = date.today() + timedelta(days=1)
    else:
        parsed = dateparser.parse(user_text, settings={"PREFER_DATES_FROM": "future"})
        if parsed:
            d = parsed.date()
        else:
            return (
                "I couldn't understand that date. Please try again, "
                "e.g. *today*, *tomorrow*, or *15 April 2025*."
            ), state

    state["travel_date"] = d.isoformat()
    state["state"] = T2_RESULT

    # --- Run the prediction ---
    return _run_prediction(state)


def _run_prediction(state: dict) -> tuple[str, dict]:
    """Call the ML model and format a friendly response."""
    d = date.fromisoformat(state["travel_date"])

    try:
        result = predict_arrival_time(
            current_station=state["current_station_crs"],
            destination_station=state["dest_crs"],
            current_delay_minutes=state["delay_minutes"],
            planned_arrival_time_str=state["planned_arrival"],
            day_of_week=d.weekday(),
            month=d.month,
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        state["state"] = T2_DONE
        return (
            "Sorry, I wasn't able to make a prediction. "
            f"There was an error: {e}\n\n"
            "Please make sure the models have been trained by running:\n"
            "`python -m task2.train_models`"
        ), state

    delay = result["predicted_delay_min"]
    arrival = result["predicted_arrival"]
    planned = result["planned_arrival"]
    dest = state["dest_name"]
    current = state["current_station_name"]
    model = result.get("model_name", "ML model")

    # Build a friendly message
    if delay <= 0.5:
        delay_desc = "roughly on time"
    else:
        delay_desc = f"around **{abs(delay):.0f} minute{'s' if abs(delay) != 1 else ''}** late"

    response = (
        f"📊 **Prediction Result**\n\n"
        f"Based on your current delay of **{state['delay_minutes']} minutes** "
        f"at **{current}**, I predict your train will arrive at "
        f"**{dest}** at approximately **{arrival}**, {delay_desc}.\n\n"
        f"- 🕐 **Scheduled arrival:** {planned}\n"
        f"- 🕐 **Predicted arrival:** {arrival}\n"
        f"- ⏱️ **Predicted delay:** {delay:.1f} minutes\n"
        f"- 🤖 **Model used:** {model}\n\n"
        "Would you like to check another train? Just tell me which station "
        "your train is at!"
    )

    state["state"] = T2_DONE
    return response, state


def _handle_done(user_text: str, state: dict) -> tuple[str, dict]:
    """After delivering a result, allow the user to start over."""
    # Reset for a new query but keep the conversation going
    new_state = get_initial_state()
    new_state["state"] = T2_COLLECT_STATION
    return _handle_collect_station(user_text, new_state)
