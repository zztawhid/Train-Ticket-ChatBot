"""
conversation_live.py  (Task 2 — Live Delay)
-------------------------------------------
Streamlit-integrated dialogue manager for the Live Train Delay Predictor.

Unlike the original conversation.py (which asks for a future travel date),
this version uses datetime.now() automatically — no date/time question.

Conversation flow
-----------------
1. Greeting — explains the tool, shows date/time is from NOW
2. Ask direction: Weymouth → London, or London → Weymouth?
3. Ask current station — via dropdown (st.selectbox in app.py)
4. Ask destination — via dropdown, only stations ahead in travel direction
5. Ask current delay (minutes) — free-text number
6. Run prediction → show result

Dropdown interaction pattern
------------------------------
When state is T2L_COLLECT_STATION or T2L_COLLECT_DEST, the app.py UI renders
a st.selectbox above the chat input.  When the user selects a station and
clicks "Confirm", app.py sends the CRS code string as the user message
(e.g. "SOU").  process_message_live() detects a valid CRS code and accepts it
directly without further confirmation.

State keys
----------
  state["state"]                — current conversation state constant
  state["direction"]            — 0=WEY2WAT, 1=WAT2WEY, None if not yet set
  state["current_station_crs"]  — CRS code of current station
  state["current_station_name"] — Display name of current station
  state["dest_crs"]             — CRS code of destination
  state["dest_name"]            — Display name of destination
  state["delay_minutes"]        — float
  state["pending_dropdown"]     — dict or None:
                                    {"field": "current"|"destination",
                                     "options": [(crs, name), ...]}
"""

import re
import logging
from datetime import datetime

from task2.predict_v2 import predict_arrival_time_v2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation state constants
# ---------------------------------------------------------------------------
T2L_GREETING         = "T2L_GREETING"
T2L_COLLECT_DIRECTION = "T2L_COLLECT_DIRECTION"
T2L_COLLECT_STATION  = "T2L_COLLECT_STATION"
T2L_COLLECT_DEST     = "T2L_COLLECT_DEST"
T2L_COLLECT_DELAY    = "T2L_COLLECT_DELAY"
T2L_RESULT           = "T2L_RESULT"
T2L_DONE             = "T2L_DONE"


# ---------------------------------------------------------------------------
# Route stations — ordered from Weymouth to London Waterloo
# CRS codes verified against data/StationNameAndCode.csv
# ---------------------------------------------------------------------------
ROUTE_STATIONS = [
    ("WEY", "Weymouth"),
    ("DCH", "Dorchester South"),    # DCH not DCG
    ("WOO", "Wool"),
    ("WRM", "Wareham"),             # WRM not WAR
    ("POO", "Poole"),
    ("BMH", "Bournemouth"),
    ("CHR", "Christchurch"),
    ("NWM", "New Milton"),
    ("BCU", "Brockenhurst"),        # BCU not BRK
    ("SOA", "Southampton Airport Parkway"),
    ("SOU", "Southampton Central"),
    ("ESL", "Eastleigh"),
    ("WIN", "Winchester"),
    ("BSK", "Basingstoke"),
    ("WOK", "Woking"),
    ("WAT", "London Waterloo"),
]

# Fast lookup: CRS → name, and name-like → CRS
_CRS_TO_NAME  = {crs: name for crs, name in ROUTE_STATIONS}
_ALL_CRS      = {crs for crs, _ in ROUTE_STATIONS}


def _stations_for_direction(direction: int):
    """
    Return station list in the correct travel order.
    direction=0 → WEY2WAT (WEY first)
    direction=1 → WAT2WEY (WAT first)
    """
    if direction == 1:
        return list(reversed(ROUTE_STATIONS))
    return list(ROUTE_STATIONS)


def _stations_ahead(current_crs: str, direction: int):
    """
    Return stations that are AHEAD of current_crs in the travel direction
    (not including current_crs itself).
    """
    ordered = _stations_for_direction(direction)
    crs_list = [crs for crs, _ in ordered]
    try:
        idx = crs_list.index(current_crs)
    except ValueError:
        return ordered  # fallback: return all
    return ordered[idx + 1:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_initial_state_live() -> dict:
    """Return a fresh Task 2 Live conversation state."""
    return {
        "state":                T2L_GREETING,
        "direction":            None,
        "current_station_crs":  None,
        "current_station_name": None,
        "dest_crs":             None,
        "dest_name":            None,
        "delay_minutes":        None,
        "pending_dropdown":     None,
    }


def get_greeting_message_live() -> str:
    """Return the opening message for Task 2 Live."""
    now = datetime.now()
    return (
        "🚆 **Live Train Delay Predictor**\n\n"
        "I can predict when your delayed train will arrive at your destination "
        "using today's real date and time.\n\n"
        f"📅 **Today:** {now.strftime('%A, %d %B %Y')}  "
        f"🕐 **Time now:** {now.strftime('%H:%M')}\n\n"
        "Are you travelling **FROM Weymouth TO London Waterloo**, or "
        "**FROM London Waterloo TO Weymouth**?\n\n"
        "Reply **1** for Weymouth → London, or **2** for London → Weymouth."
    )


def process_message_live(user_text: str, state: dict) -> tuple[str, dict]:
    """
    Process a user message and advance the conversation state.

    Parameters
    ----------
    user_text : str   — what the user typed (or a CRS code sent by the UI)
    state     : dict  — current conversation state

    Returns
    -------
    (response_text, updated_state)
    """
    user_text = user_text.strip()
    current = state["state"]

    if current == T2L_GREETING:
        state["state"] = T2L_COLLECT_DIRECTION
        return _handle_collect_direction(user_text, state)

    elif current == T2L_COLLECT_DIRECTION:
        return _handle_collect_direction(user_text, state)

    elif current == T2L_COLLECT_STATION:
        return _handle_collect_station(user_text, state)

    elif current == T2L_COLLECT_DEST:
        return _handle_collect_dest(user_text, state)

    elif current == T2L_COLLECT_DELAY:
        return _handle_collect_delay(user_text, state)

    elif current in (T2L_RESULT, T2L_DONE):
        return _handle_done(user_text, state)

    else:
        state["state"] = T2L_COLLECT_DIRECTION
        return get_greeting_message_live(), state


# ---------------------------------------------------------------------------
# State handlers
# ---------------------------------------------------------------------------

def _handle_collect_direction(user_text: str, state: dict) -> tuple[str, dict]:
    """Collect travel direction (Weymouth→London or London→Weymouth)."""
    lower = user_text.lower().strip()

    direction = None

    # Numeric shortcuts
    if lower in ("1",):
        direction = 0  # WEY2WAT
    elif lower in ("2",):
        direction = 1  # WAT2WEY

    # Natural language
    elif any(kw in lower for kw in ("wey", "weymouth", "wey2wat", "wey to wat")):
        direction = 0
    elif any(kw in lower for kw in ("wat", "waterloo", "london", "wat2wey", "wat to wey")):
        direction = 1
    elif any(kw in lower for kw in ("from weymouth", "weymouth to")):
        direction = 0
    elif any(kw in lower for kw in ("from london", "from waterloo", "london to")):
        direction = 1

    if direction is None:
        return (
            "I didn't catch that. Please reply:\n\n"
            "**1** - I'm travelling from Weymouth to London Waterloo\n"
            "**2** - I'm travelling from London Waterloo to Weymouth"
        ), state

    state["direction"] = direction
    dir_label = "Weymouth → London Waterloo" if direction == 0 else "London Waterloo → Weymouth"

    # Prepare dropdown options for the current station
    ordered = _stations_for_direction(direction)
    # Exclude the final terminus (can't be "current station")
    current_options = ordered[:-1]

    state["state"] = T2L_COLLECT_STATION
    state["pending_dropdown"] = {
        "field":   "current",
        "options": current_options,
    }

    return (
        f"✅ Direction: **{dir_label}**\n\n"
        "Please select the station your train is currently at from the dropdown below."
    ), state


def _handle_collect_station(user_text: str, state: dict) -> tuple[str, dict]:
    """
    Accept the current station selection.
    The UI sends the CRS code directly from the selectbox.
    """
    crs = user_text.strip().upper()

    if crs in _ALL_CRS:
        name = _CRS_TO_NAME[crs]
        state["current_station_crs"]  = crs
        state["current_station_name"] = name
        state["pending_dropdown"]      = None

        # Prepare ahead-only options for destination
        direction  = state["direction"]
        dest_options = _stations_ahead(crs, direction)

        if not dest_options:
            # Edge case: already at terminus
            state["state"] = T2L_COLLECT_DIRECTION
            return (
                "That station appears to be the terminus. "
                "Please restart and choose a different current station."
            ), state

        state["state"] = T2L_COLLECT_DEST
        state["pending_dropdown"] = {
            "field":   "destination",
            "options": dest_options,
        }
        return (
            f"✅ Current station: **{name}** ({crs})\n\n"
            "Now please select your destination from the dropdown below. "
            "Only stations ahead of you on the route are shown."
        ), state

    # If not a direct CRS code, try fuzzy match against station names
    for crs_code, sname in ROUTE_STATIONS:
        if user_text.lower() in sname.lower():
            # Re-issue as CRS
            return _handle_collect_station(crs_code, state)

    return (
        "I didn't recognise that station. "
        "Please use the dropdown to select your current station."
    ), state


def _handle_collect_dest(user_text: str, state: dict) -> tuple[str, dict]:
    """
    Accept the destination selection.
    The UI sends the CRS code directly from the selectbox.
    """
    crs = user_text.strip().upper()

    if crs in _ALL_CRS:
        if crs == state.get("current_station_crs"):
            return (
                "That's the same as your current station! "
                "Please choose a station further along the route."
            ), state

        name = _CRS_TO_NAME[crs]
        state["dest_crs"]  = crs
        state["dest_name"] = name
        state["pending_dropdown"] = None
        state["state"] = T2L_COLLECT_DELAY

        return (
            f"✅ Destination: **{name}** ({crs})\n\n"
            "How many minutes is your train currently delayed? "
            "Please type a number (e.g. *10*)."
        ), state

    # Fuzzy match fallback
    for crs_code, sname in ROUTE_STATIONS:
        if user_text.lower() in sname.lower():
            return _handle_collect_dest(crs_code, state)

    return (
        "I didn't recognise that station. "
        "Please use the dropdown to select your destination."
    ), state


def _handle_collect_delay(user_text: str, state: dict) -> tuple[str, dict]:
    """Collect the current delay in minutes."""
    match = re.search(r"(\d+(?:\.\d+)?)", user_text)
    if match:
        delay = float(match.group(1))
        if delay < 0 or delay > 300:
            return (
                "That doesn't seem right. "
                "Please enter a delay between 0 and 300 minutes."
            ), state
        state["delay_minutes"] = delay
        state["state"] = T2L_RESULT
        return _run_prediction(state)

    return (
        "I need a number for the delay. "
        "How many minutes is your train delayed? (e.g. *10*)"
    ), state


def _run_prediction(state: dict) -> tuple[str, dict]:
    """Call the v2 ML model and format a friendly response."""
    try:
        result = predict_arrival_time_v2(
            current_station=state["current_station_crs"],
            destination_station=state["dest_crs"],
            current_delay_minutes=state["delay_minutes"],
            planned_arrival_time_str=_estimate_planned_arrival(state),
            direction=state["direction"],
        )
    except FileNotFoundError as exc:
        state["state"] = T2L_DONE
        return (
            "Sorry — the v2 prediction model has not been trained yet.\n\n"
            f"Please run:\n```\npython -m task2.train_models_v2\n```\n\n"
            f"Technical detail: {exc}"
        ), state
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        state["state"] = T2L_DONE
        return (
            f"Sorry, there was an error making the prediction: {exc}\n\n"
            "Please ensure the v2 models are trained."
        ), state

    delay   = result["predicted_delay_min"]
    arrival = result["predicted_arrival"]
    planned = result["planned_arrival"]
    dest    = state["dest_name"]
    current = state["current_station_name"]
    model   = result.get("model_name", "ML model")

    now = datetime.now()
    day_name = now.strftime("%A")

    if delay <= 0.5:
        delay_desc = "roughly **on time**"
    elif delay < 0:
        delay_desc = f"about **{abs(delay):.0f} minute{'s' if abs(delay) != 1 else ''} early**"
    else:
        delay_desc = f"around **{delay:.0f} minute{'s' if delay != 1 else ''} late**"

    direction_str = (
        "Weymouth → London Waterloo"
        if state["direction"] == 0
        else "London Waterloo → Weymouth"
    )

    response = (
        f"📊 **Prediction Result**\n\n"
        f"Based on your current delay of **{state['delay_minutes']:.0f} minutes** "
        f"at **{current}**, my model predicts your train will arrive at "
        f"**{dest}** at approximately **{arrival}**, {delay_desc}.\n\n"
        f"- 🗓️  **Date/Day:** {now.strftime('%d %b %Y')} ({day_name})\n"
        f"- 🚆 **Route:** {direction_str}\n"
        f"- 🕐 **Scheduled arrival:** {planned}\n"
        f"- 🕐 **Predicted arrival:** {arrival}\n"
        f"- ⏱️  **Predicted delay:** {delay:.1f} minutes\n"
        f"- 🤖 **Model:** {model}\n\n"
        "Would you like to check another train? Just say **yes** or describe "
        "your situation and I'll start again."
    )

    state["state"] = T2L_DONE
    return response, state


# ─── Data-driven travel-time lookup (replaces the old 4-min-per-stop guess) ──
# Loaded lazily on first use from models/station_travel_times_v2.pkl.
# Keys: (direction:int, from_crs:str, to_crs:str) -> mean travel time (min)
# Built by task2/build_artifacts_v2.py from the actual planned timetable
# in the training data.
_TRAVEL_TIMES_CACHE = None  # type: ignore  # Optional[dict]

def _load_travel_times() -> dict:
    """Load and cache the station-to-station travel time lookup."""
    global _TRAVEL_TIMES_CACHE
    if _TRAVEL_TIMES_CACHE is not None:
        return _TRAVEL_TIMES_CACHE
    import os
    import joblib
    path = os.path.join(os.path.dirname(__file__), "..", "models",
                        "station_travel_times_v2.pkl")
    try:
        _TRAVEL_TIMES_CACHE = joblib.load(path)
    except FileNotFoundError:
        logger.warning("station_travel_times_v2.pkl not found — falling back "
                       "to 5 min/stop heuristic. Run "
                       "`python -m task2.build_artifacts_v2` to generate it.")
        _TRAVEL_TIMES_CACHE = {}
    return _TRAVEL_TIMES_CACHE


def _estimate_planned_arrival(state: dict) -> str:
    """
    Estimate the scheduled arrival time at the destination using the
    pairwise PLANNED travel-time lookup built from the training data.

    Algorithm (in priority order):
      1. Direct lookup of (direction, current_crs, dest_crs). This is the
         actual mean timetabled time observed in services that included
         both stations on the same journey - including long-distance pairs
         like WAT -> WEY where intermediate consecutive pairs are missing.
      2. If the direct pair isn't in the lookup, walk the route stop by
         stop and sum the consecutive-pair times, using the route-specific
         mean as fallback for any missing segments (a much better default
         than the global mean across all SWR stations).
      3. If the lookup file is missing entirely, fall back to a coarse
         8 minutes per stop estimate.

    Returns the predicted scheduled arrival as an "HH:MM" string.
    """
    travel_times = _load_travel_times()
    coarse_per_stop = 8  # used only if no lookup at all

    try:
        direction = int(state["direction"])
        ordered  = _stations_for_direction(direction)
        crs_list = [crs for crs, _ in ordered]
        curr_idx = crs_list.index(state["current_station_crs"])
        dest_idx = crs_list.index(state["dest_crs"])
        a_crs    = crs_list[curr_idx]
        b_crs    = crs_list[dest_idx]

        mins_remaining = None

        # Strategy 1: direct pair lookup
        if travel_times:
            direct = travel_times.get((direction, a_crs, b_crs))
            if direct is not None:
                mins_remaining = direct
            else:
                # Strategy 2: sum consecutive segments along the route
                # Use route-specific mean for missing pairs (much better
                # than the global mean which includes dense urban segments)
                route_pair_times = [
                    travel_times[(direction, crs_list[i], crs_list[i + 1])]
                    for i in range(len(crs_list) - 1)
                    if (direction, crs_list[i], crs_list[i + 1]) in travel_times
                ]
                route_mean = (
                    sum(route_pair_times) / len(route_pair_times)
                    if route_pair_times else 10.0
                )
                mins_remaining = 0.0
                for i in range(curr_idx, dest_idx):
                    key = (direction, crs_list[i], crs_list[i + 1])
                    mins_remaining += travel_times.get(key, route_mean)

        if mins_remaining is None:
            # Strategy 3: coarse fallback
            stops = dest_idx - curr_idx
            mins_remaining = max(stops * coarse_per_stop, 5)

        mins_remaining = max(mins_remaining, 3.0)
    except (ValueError, TypeError, KeyError):
        mins_remaining = 30.0

    now = datetime.now()
    total_minutes = now.hour * 60 + now.minute + int(round(mins_remaining))
    total_minutes %= 1440
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}"


def _handle_done(user_text: str, state: dict) -> tuple[str, dict]:
    """After delivering a result, allow the user to start a new query."""
    lower = user_text.lower().strip()
    if any(kw in lower for kw in ("yes", "another", "again", "new", "restart")):
        new_state = get_initial_state_live()
        new_state["state"] = T2L_COLLECT_DIRECTION
        return (
            "Sure! Let's start a new prediction.\n\n"
            "Are you travelling **FROM Weymouth TO London Waterloo** or "
            "**FROM London Waterloo TO Weymouth**?\n\n"
            "Reply **1** for Weymouth → London, or **2** for London → Weymouth."
        ), new_state

    return (
        "If you'd like to check another train, just say **yes** or type your "
        "new query and I'll restart the prediction."
    ), state
