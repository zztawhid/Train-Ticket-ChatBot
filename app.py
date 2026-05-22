"""
app.py
------
Streamlit entry point for the National Rail Ticket Assistant chatbot.

Run with:
    streamlit run app.py

Provides a clean, professional chat interface with a train/travel theme.
Uses st.chat_message and st.chat_input for the conversational UI and
manages all state through st.session_state.
"""

import uuid
from datetime import date, time, datetime, timedelta
import streamlit as st

from chatbot.conversation import (
    get_initial_state as t1_get_initial_state,
    get_greeting_message as t1_get_greeting_message,
    process_message as t1_process_message,
    confirm_station,
    COLLECT_DATE,
    COLLECT_TIME,
    COLLECT_RETURN_DATE,
    COLLECT_RETURN_TIME,
    DONE,
)
from chatbot.database import get_all_sessions, get_history, clear_session, save_message

# Task 2 is optional — if its models or dependencies are missing the whole app
# should not crash; Task 1 remains fully functional.
try:
    from task2.conversation import (
        get_initial_state as t2_get_initial_state,
        get_greeting_message as t2_get_greeting_message,
        process_message as t2_process_message,
    )
    _TASK2_AVAILABLE = True
except Exception:
    _TASK2_AVAILABLE = False

# Task 2 Live — live delay predictor (uses datetime.now(), no date question)
try:
    from task2.conversation_live import (
        get_initial_state_live as t2l_get_initial_state,
        get_greeting_message_live as t2l_get_greeting_message,
        process_message_live as t2l_process_message,
        ROUTE_STATIONS,
        T2L_COLLECT_STATION,
        T2L_COLLECT_DEST,
        T2L_DONE,
    )
    _TASK2_LIVE_AVAILABLE = True
except Exception:
    _TASK2_LIVE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="National Rail Ticket Assistant",
    page_icon="🚆",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a professional train/travel theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-size: 17px !important;
        background: linear-gradient(175deg, #ffffff 0%) !important;
    }

    /* ---------- Lock sidebar open — hide toggle & prevent slide-away ---------- */
    [data-testid="collapsedControl"]          { display: none !important; }
    [data-testid="stSidebarCollapseButton"]   { display: none !important; }
    button[aria-label="Close sidebar"],
    button[aria-label="Collapse sidebar"],
    button[title="Collapse sidebar"]          { display: none !important; }
    section[data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        width: 18rem !important;
        min-width: 18rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        width: 18rem !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    /* ---------- Header — flat bar, not a rounded bubble ---------- */
    .header-banner {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #2b6cb0 100%);
        border-radius: 0;
        padding: 1.25rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        border-bottom: 3px solid #4299e1;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4);
    }
    .header-banner h1 {
        color: #ffffff;
        font-size: 1.7rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .header-banner p {
        color: #bee3f8;
        font-size: 0.95rem;
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
    }

    /* ---------- Chat messages ---------- */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
        font-size: 18px !important;
    }
    [data-testid="stChatMessage-user"] p,
    [data-testid="stChatMessage-user"] span {
        font-size: 18px !important;
        color: #ffffff !important;
    }
    [data-testid="stChatMessage-assistant"] p,
    [data-testid="stChatMessage-assistant"] span,
    [data-testid="stChatMessage-assistant"] li {
        font-size: 18px !important;
        color: #e2e8f0 !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2f50 0%, #1a2333 100%) !important;
        font-size: 13px !important;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e0 !important;
        font-size: 18px !important;
    }
    /* Section labels */
    .sb-label {
        font-size: 18px !important;
        font-weight: 700 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        color: #7096b8 !important;
        margin: 0.9rem 0 0.3rem 0 !important;
    }
    /* Journey info card */
    .journey-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 6px;
        padding: 0.6rem 0.75rem;
        margin-bottom: 0.5rem;
        font-size: 13px !important;
        line-height: 1.6;
    }
    .journey-card span.jlabel {
        color: #7096b8 !important;
        font-size: 11px !important;
        display: block;
        margin-bottom: 1px;
    }
    .journey-card span.jval {
        color: #e2e8f0 !important;
        font-weight: 500;
        font-size: 13px !important;
    }
    /* Thin divider */
    section[data-testid="stSidebar"] hr {
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.08) !important;
        margin: 0.6rem 0 !important;
    }
    /* All sidebar buttons — compact base */
    section[data-testid="stSidebar"] [data-testid="stButton"] > button {
        font-size: 13px !important;
        padding: 0.3rem 0.6rem !important;
        border-radius: 4px !important;
        height: auto !important;
    }
    /* Past conversation row — remove gap between button and bin */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        gap: 4px !important;
        align-items: center !important;
        margin-bottom: 4px !important;
    }
    /* Past conversation buttons — left-aligned, single line, no pill */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]
        [data-testid="column"]:first-child button {
        color: #000000 !important;
        background-color: rgba(255,255,255,0.85) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 4px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        padding: 0.3rem 0.55rem !important;
        font-size: 12px !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]
        [data-testid="column"]:first-child button:hover {
        background-color: rgba(255,255,255,1) !important;
        border-color: rgba(66,153,225,0.5) !important;
        color: #000000 !important;
    }


    /* Bin icon — transparent, no box */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]
        [data-testid="column"]:last-child {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex: 0 0 28px !important;
        min-width: 28px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]
        [data-testid="column"]:last-child button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.15rem 0.2rem !important;
        color: #3d5570 !important;
        font-size: 13px !important;
        min-height: unset !important;
        height: auto !important;
    }
    .st-emotion-cache-9114l4{
        background: rgb(196, 196, 196) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]
        [data-testid="column"]:last-child button:hover {
        color: #fc8181 !important;
        background: transparent !important;
    }
    
    /* New conversation button */
    section[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type > button {
        background: rgba(66,153,225,0.18) !important;
        color: #90cdf4 !important;
        border: 1px solid rgba(66,153,225,0.4) !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.15s;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type > button:hover {
        background: #4299e1 !important;
        color: #ffffff !important;
    }
    /* Radio buttons in sidebar */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-size: 13px !important;
        padding: 0.25rem 0 !important;
    }

    /* Align station picker select and confirm button */
    [data-testid="stAppViewContainer"] div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    /* Prevent typing in selectbox input while keeping dropdown clickable */
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] {
        cursor: pointer;
    }
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] input,
    [data-testid="stAppViewContainer"] div[role="combobox"] input {
        pointer-events: none;
        caret-color: transparent;
        user-select: none;
    }

    /* ---------- Chat input ---------- */
    .stChatInput textarea {
        border-radius: 24px !important;
        font-size: 18px !important;
    }

    /* ---------- Buttons (main area) ---------- */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #63b3ed;
        background: transparent;
        color: #b7ddf7;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #4299e1;
        color: white;
    }

    .st-emotion-cache-3uj0rx { color: rgb(0, 0, 0); }
    .st-emotion-cache-17k2yau p { color: white; }
    .st-emotion-cache-128upt6 { background-color: rgb(103 103 103); }

    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="manage-app-button"] {display: none;}
    div[data-testid="stStatusWidget"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header (dynamic based on active task)
# ---------------------------------------------------------------------------
# We read the active task early so the header matches
_active = st.session_state.get("active_task", "task1")

if _active == "task2":
    st.markdown(
        """
        <div class="header-banner">
            <h1>🚆 Train Delay Predictor</h1>
            <p>Predict when your delayed train will actually arrive</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif _active == "task2live":
    st.markdown(
        """
        <div class="header-banner">
            <h1>🚆 Live Delay Predictor</h1>
            <p>Real-time prediction using today's date &amp; time automatically</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="header-banner">
            <h1>🚆 National Rail Ticket Assistant</h1>
            <p>Find the cheapest train tickets across the UK</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "active_task" not in st.session_state:
    st.session_state.active_task = "task1"

# Task 1 state
if "t1_messages" not in st.session_state:
    st.session_state.t1_messages = []
if "t1_conv_state" not in st.session_state:
    st.session_state.t1_conv_state = t1_get_initial_state()

# Task 2 state (only initialise if task2 loaded successfully)
if "t2_messages" not in st.session_state:
    st.session_state.t2_messages = []
if "t2_conv_state" not in st.session_state:
    st.session_state.t2_conv_state = t2_get_initial_state() if _TASK2_AVAILABLE else {}

# Task 2 Live state
if "t2live_messages" not in st.session_state:
    st.session_state.t2live_messages = []
if "t2live_state" not in st.session_state:
    st.session_state.t2live_state = t2l_get_initial_state() if _TASK2_LIVE_AVAILABLE else {}

# Backwards-compatible aliases (point to the active task's data)
def _sync_aliases():
    if st.session_state.active_task == "task1":
        st.session_state.messages = st.session_state.t1_messages
        st.session_state.conv_state = st.session_state.t1_conv_state
    elif st.session_state.active_task == "task2live":
        st.session_state.messages = st.session_state.t2live_messages
        st.session_state.conv_state = st.session_state.t2live_state
    else:
        st.session_state.messages = st.session_state.t2_messages
        st.session_state.conv_state = st.session_state.t2_conv_state

_sync_aliases()

# DB is already initialised when database.py is imported; no duplicate call needed.

# Send the greeting on first load
if not st.session_state.t1_messages:
    greeting = t1_get_greeting_message()
    st.session_state.t1_messages.append({"role": "assistant", "content": greeting})
if _TASK2_AVAILABLE and not st.session_state.t2_messages:
    greeting = t2_get_greeting_message()
    st.session_state.t2_messages.append({"role": "assistant", "content": greeting})
if _TASK2_LIVE_AVAILABLE and not st.session_state.t2live_messages:
    greeting = t2l_get_greeting_message()
    st.session_state.t2live_messages.append({"role": "assistant", "content": greeting})

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── Mode switcher ────────────────────────────────────────────────
    st.markdown("<p class='sb-label'>Mode</p>", unsafe_allow_html=True)
    _task_options = ["task1"]
    if _TASK2_AVAILABLE:
        _task_options.append("task2")
    if _TASK2_LIVE_AVAILABLE:
        _task_options.append("task2live")

    def _task_label(x):
        if x == "task1":
            return "🎫 Find Cheapest Ticket"
        elif x == "task2":
            return "⏱️ Predict Arrival Time"
        elif x == "task2live":
            return "🚆 Live Delay (Task 2)"
        return x

    _current_idx = _task_options.index(st.session_state.active_task) if st.session_state.active_task in _task_options else 0
    task_choice = st.radio(
        "mode",
        _task_options,
        format_func=_task_label,
        index=_current_idx,
        key="task_radio",
        label_visibility="collapsed",
    )
    if task_choice != st.session_state.active_task:
        st.session_state.active_task = task_choice
        _sync_aliases()
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Journey context card ─────────────────────────────────────────
    if st.session_state.active_task == "task1":
        cs = st.session_state.t1_conv_state
        any_info = any(cs.get(k) for k in ("origin_name", "dest_name", "travel_date", "ticket_type"))
        if any_info:
            st.markdown("<p class='sb-label'>Current Journey</p>", unsafe_allow_html=True)

            def _row(label, value):
                return (
                    f"<span class='jlabel'>{label}</span>"
                    f"<span class='jval'>{value}</span>"
                )

            rows = []
            if cs.get("origin_name"):
                rows.append(_row("From", f"{cs['origin_name']} ({cs.get('origin_crs','')})"))
            if cs.get("dest_name"):
                rows.append(_row("To", f"{cs['dest_name']} ({cs.get('dest_crs','')})"))
            if cs.get("travel_date"):
                try:
                    d = datetime.strptime(cs["travel_date"], "%Y-%m-%d")
                    rows.append(_row("Date", d.strftime("%a %d %b %Y")))
                except ValueError:
                    rows.append(_row("Date", cs["travel_date"]))
            if cs.get("travel_time"):
                t_raw = cs["travel_time"][:5]
                rows.append(_row("Time", t_raw))
            if cs.get("ticket_type"):
                rows.append(_row("Ticket", cs["ticket_type"].title()))

            card_inner = "".join(f"<div style='margin-bottom:6px'>{r}</div>" for r in rows)
            st.markdown(
                f"<div class='journey-card'>{card_inner}</div>",
                unsafe_allow_html=True,
            )
    elif st.session_state.active_task == "task2":
        cs = st.session_state.t2_conv_state
        any_info = any(cs.get(k) for k in ("current_station_name", "dest_name", "delay_minutes"))
        if any_info:
            st.markdown("<p class='sb-label'>Delay Prediction</p>", unsafe_allow_html=True)
            rows = []
            if cs.get("current_station_name"):
                rows.append(f"<span class='jlabel'>At</span><span class='jval'>{cs['current_station_name']}</span>")
            if cs.get("dest_name"):
                rows.append(f"<span class='jlabel'>To</span><span class='jval'>{cs['dest_name']}</span>")
            if cs.get("delay_minutes") is not None:
                rows.append(f"<span class='jlabel'>Delay</span><span class='jval'>{cs['delay_minutes']} min</span>")
            card_inner = "".join(f"<div style='margin-bottom:6px'>{r}</div>" for r in rows)
            st.markdown(f"<div class='journey-card'>{card_inner}</div>", unsafe_allow_html=True)
    else:
        # task2live
        cs = st.session_state.t2live_state
        any_info = any(cs.get(k) for k in ("current_station_name", "dest_name", "delay_minutes"))
        if any_info:
            st.markdown("<p class='sb-label'>Live Prediction</p>", unsafe_allow_html=True)
            rows = []
            if cs.get("direction") is not None:
                dir_str = "WEY→WAT" if cs["direction"] == 0 else "WAT→WEY"
                rows.append(f"<span class='jlabel'>Direction</span><span class='jval'>{dir_str}</span>")
            if cs.get("current_station_name"):
                rows.append(f"<span class='jlabel'>At</span><span class='jval'>{cs['current_station_name']}</span>")
            if cs.get("dest_name"):
                rows.append(f"<span class='jlabel'>To</span><span class='jval'>{cs['dest_name']}</span>")
            if cs.get("delay_minutes") is not None:
                rows.append(f"<span class='jlabel'>Delay</span><span class='jval'>{cs['delay_minutes']:.0f} min</span>")
            card_inner = "".join(f"<div style='margin-bottom:6px'>{r}</div>" for r in rows)
            st.markdown(f"<div class='journey-card'>{card_inner}</div>", unsafe_allow_html=True)

    # ── New conversation ─────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("＋  New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        if st.session_state.active_task == "task1":
            st.session_state.t1_messages = []
            st.session_state.t1_conv_state = t1_get_initial_state()
            st.session_state.t1_messages.append(
                {"role": "assistant", "content": t1_get_greeting_message()}
            )
        elif st.session_state.active_task == "task2live" and _TASK2_LIVE_AVAILABLE:
            st.session_state.t2live_messages = []
            st.session_state.t2live_state = t2l_get_initial_state()
            st.session_state.t2live_messages.append(
                {"role": "assistant", "content": t2l_get_greeting_message()}
            )
        elif _TASK2_AVAILABLE:
            st.session_state.t2_messages = []
            st.session_state.t2_conv_state = t2_get_initial_state()
            st.session_state.t2_messages.append(
                {"role": "assistant", "content": t2_get_greeting_message()}
            )
        _sync_aliases()
        st.rerun()

    # ── Past conversations ───────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='sb-label'>Past Conversations</p>", unsafe_allow_html=True)

    all_sessions = get_all_sessions()
    past_sessions = [s for s in all_sessions if s != st.session_state.session_id]

    if past_sessions:
        for sid in past_sessions:
            history = get_history(sid)
            if not history:
                continue
            preview = next(
                (m["message"][:35] + ("…" if len(m["message"]) > 35 else "")
                 for m in history if m["role"] == "user"),
                "(empty)"
            )
            timestamp = history[0]["timestamp"][:10]   # just the date

            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    preview,
                    key=f"load_{sid}",
                    use_container_width=True,
                    help=f"Started {timestamp}",
                ):
                    loaded = [{"role": m["role"], "content": m["message"]} for m in history]
                    if st.session_state.active_task == "task1":
                        st.session_state.t1_messages = loaded
                        st.session_state.t1_conv_state = t1_get_initial_state()
                        st.session_state.t1_conv_state["state"] = DONE
                    elif _TASK2_AVAILABLE:
                        st.session_state.t2_messages = loaded
                        st.session_state.t2_conv_state = t2_get_initial_state()
                        st.session_state.t2_conv_state["state"] = DONE
                    _sync_aliases()
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{sid}", help="Delete"):
                    clear_session(sid)
                    st.rerun()
    else:
        st.markdown(
            "<p style='color:#4a6080;font-size:12px;margin-top:4px;'>"
            "No past conversations yet.</p>",
            unsafe_allow_html=True,
        )

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#3a5272;font-size:11px;line-height:1.5;'>"
        "National Rail OJP API · CMP-6059B Group 06</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------
_sync_aliases()
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🚆" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Station picker dropdown
# Shown when the bot has found multiple station candidates and is waiting for
# the user to confirm or pick one. Replaces the old numbered list.
# ---------------------------------------------------------------------------
_t1_state = st.session_state.t1_conv_state
_pending = _t1_state.get("pending_station") if st.session_state.active_task == "task1" else None
_ambiguous = _t1_state.get("awaiting_clarification") if st.session_state.active_task == "task1" else None

if _pending and len(_pending.get("candidates", [])) > 1:
    _field = _pending["field"]
    _candidates = _pending["candidates"]
    _label = "departure" if _field == "origin" else "destination"
    _options = [f"{n} ({c})" for n, c, _ in _candidates]

    with st.container():
        st.markdown(f"##### 🚉 Select your {_label} station:")
        cols = st.columns([4, 1])
        with cols[0]:
            _selected = st.selectbox(
                "Station", options=_options,
                key=f"station_drop_{_field}", label_visibility="collapsed",
            )
        with cols[1]:
            if st.button("✅ Confirm", key=f"station_btn_{_field}", use_container_width=True):
                _idx = _options.index(_selected)
                _sel_name, _sel_crs, _ = _candidates[_idx]
                # Update pending_station to the chosen option then send "yes"
                st.session_state.t1_conv_state["pending_station"]["crs"] = _sel_crs
                st.session_state.t1_conv_state["pending_station"]["name"] = _sel_name
                with st.chat_message("user", avatar="👤"):
                    st.markdown(_selected)
                st.session_state.t1_messages.append({"role": "user", "content": _selected})
                with st.chat_message("assistant", avatar="🚆"):
                    with st.spinner("..."):
                        _resp, _new = t1_process_message(
                            "yes", st.session_state.t1_conv_state, st.session_state.session_id
                        )
                    st.markdown(_resp)
                st.session_state.t1_messages.append({"role": "assistant", "content": _resp})
                st.session_state.t1_conv_state = _new
                _sync_aliases()
                st.rerun()

elif _ambiguous:
    _field, _candidates = _ambiguous
    _label = "departure" if _field == "origin" else "destination"
    _options = [f"{n} ({c})" for n, c, _ in _candidates]

    with st.container():
        st.markdown(f"##### 🚉 Select your {_label} station:")
        cols = st.columns([4, 1])
        with cols[0]:
            _selected = st.selectbox(
                "Station", options=_options,
                key=f"amb_drop_{_field}", label_visibility="collapsed",
            )
        with cols[1]:
            if st.button("✅ Confirm", key=f"amb_btn_{_field}", use_container_width=True):
                _idx = _options.index(_selected)
                _sel_name, _sel_crs, _ = _candidates[_idx]
                with st.chat_message("user", avatar="👤"):
                    st.markdown(_selected)
                st.session_state.t1_messages.append({"role": "user", "content": _selected})
                with st.chat_message("assistant", avatar="🚆"):
                    with st.spinner("..."):
                        _resp, _new = t1_process_message(
                            _sel_name, st.session_state.t1_conv_state, st.session_state.session_id
                        )
                    st.markdown(_resp)
                st.session_state.t1_messages.append({"role": "assistant", "content": _resp})
                st.session_state.t1_conv_state = _new
                _sync_aliases()
                st.rerun()

# ---------------------------------------------------------------------------
# Task 2 Live — station dropdown picker
# Shown when the conversation is waiting for current station or destination.
# ---------------------------------------------------------------------------
if st.session_state.active_task == "task2live" and _TASK2_LIVE_AVAILABLE:
    _t2l_state = st.session_state.t2live_state
    _t2l_pending = _t2l_state.get("pending_dropdown")
    _t2l_conv_state_key = _t2l_state.get("state", "")

    if _t2l_pending and _t2l_conv_state_key in (T2L_COLLECT_STATION, T2L_COLLECT_DEST):
        _t2l_field   = _t2l_pending["field"]
        _t2l_options = _t2l_pending["options"]   # list of (crs, name) tuples
        _t2l_opt_labels = [f"{name} ({crs})" for crs, name in _t2l_options]
        _t2l_label = "current" if _t2l_field == "current" else "destination"

        with st.container():
            st.markdown(f"##### 🚉 Select your **{_t2l_label}** station:")
            _t2l_cols = st.columns([4, 1])
            with _t2l_cols[0]:
                _t2l_selected = st.selectbox(
                    "Station",
                    options=_t2l_opt_labels,
                    key=f"t2l_drop_{_t2l_field}",
                    label_visibility="collapsed",
                )
            with _t2l_cols[1]:
                if st.button("✅ Confirm", key=f"t2l_btn_{_t2l_field}", use_container_width=True):
                    # Extract CRS from the selected label "Name (CRS)"
                    _t2l_crs = _t2l_selected.split("(")[-1].rstrip(")")
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(_t2l_selected)
                    st.session_state.t2live_messages.append(
                        {"role": "user", "content": _t2l_selected}
                    )
                    with st.chat_message("assistant", avatar="🚆"):
                        with st.spinner("..."):
                            _t2l_resp, _t2l_new = t2l_process_message(
                                _t2l_crs, st.session_state.t2live_state
                            )
                        st.markdown(_t2l_resp)
                    st.session_state.t2live_messages.append(
                        {"role": "assistant", "content": _t2l_resp}
                    )
                    st.session_state.t2live_state = _t2l_new
                    _sync_aliases()
                    st.rerun()

# ---------------------------------------------------------------------------
# Date/Time picker (shown only for Task 1 when the conversation state requires it)
# ---------------------------------------------------------------------------
current_conv_state = st.session_state.conv_state.get("state", "")

if st.session_state.active_task == "task1" and current_conv_state in (COLLECT_DATE, COLLECT_TIME, COLLECT_RETURN_DATE, COLLECT_RETURN_TIME):
    is_return = current_conv_state in (COLLECT_RETURN_DATE, COLLECT_RETURN_TIME)
    label = "return" if is_return else "outward"

    with st.container():
        st.markdown(f"##### 📅 Or pick your {label} date & time:")
        cols = st.columns([2, 2, 1])

        with cols[0]:
            min_date = date.today()
            default_date = min_date
            if is_return and st.session_state.conv_state.get("travel_date"):
                try:
                    outward = datetime.strptime(st.session_state.conv_state["travel_date"], "%Y-%m-%d").date()
                    default_date = outward + timedelta(days=1)
                except ValueError:
                    pass
            picked_date = st.date_input("Date", value=default_date, min_value=min_date, key=f"dp_{label}")

        with cols[1]:
            default_time = time(9, 0)
            picked_time = st.time_input("Time", value=default_time, key=f"tp_{label}")

        with cols[2]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✅ Use this", key=f"btn_{label}", use_container_width=True):
                date_text = f"{picked_date.strftime('%d %B %Y')} at {picked_time.strftime('%H:%M')}"
                with st.chat_message("user", avatar="👤"):
                    st.markdown(date_text)
                st.session_state.t1_messages.append({"role": "user", "content": date_text})
                with st.chat_message("assistant", avatar="🚆"):
                    with st.spinner("Thinking..."):
                        response, new_state = t1_process_message(
                            date_text, st.session_state.t1_conv_state, st.session_state.session_id,
                        )
                    st.markdown(response)
                st.session_state.t1_messages.append({"role": "assistant", "content": response})
                st.session_state.t1_conv_state = new_state
                _sync_aliases()
                st.rerun()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if user_input := st.chat_input("Type your message here..."):
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="🚆"):
        with st.spinner("Thinking..."):
            if st.session_state.active_task == "task1":
                response, new_state = t1_process_message(
                    user_input, st.session_state.t1_conv_state, st.session_state.session_id,
                )
                st.session_state.t1_conv_state = new_state
            elif st.session_state.active_task == "task2live" and _TASK2_LIVE_AVAILABLE:
                response, new_state = t2l_process_message(
                    user_input, st.session_state.t2live_state,
                )
                st.session_state.t2live_state = new_state
            elif _TASK2_AVAILABLE and st.session_state.active_task == "task2":
                # Save user message BEFORE processing so DB order is consistent
                save_message(st.session_state.session_id, "user", user_input)
                response, new_state = t2_process_message(
                    user_input, st.session_state.t2_conv_state,
                )
                st.session_state.t2_conv_state = new_state
                save_message(st.session_state.session_id, "assistant", response)
            else:
                response = "Task 2 is currently unavailable."
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    _sync_aliases()
    st.rerun()
