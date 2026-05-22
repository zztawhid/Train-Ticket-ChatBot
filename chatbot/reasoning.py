from __future__ import annotations

"""
reasoning.py
------------
Reasoning Engine (RE) for the chatbot.

We use Experta - a Python port of CLIPS - to implement a
forward-chaining production-rule engine. Experta supports the classical
expert-system concepts taught in the module: Facts that describe the
current world state, Rules that match patterns over those facts, and
salience for explicit priority ordering when multiple rules could
fire.

The public surface of this module is unchanged from the previous
hand-written engine, so nothing else in the codebase needs to
change:

    infer_next_action(state: dict) -> tuple[str, str]
    handle_intent(intent: str, user_text: str, state: dict) -> str | None
    get_fallback(user_text: str) -> str

Internally, each call to infer_next_action():

  1. Creates a fresh ChatbotEngine.
  2. Declares a single JourneyState Fact whose fields mirror the
     current conversation state dictionary.
  3. Calls engine.run(), which fires the highest-salience matching
     rule.
  4. Returns the (action, response) tuple captured by the rule.

The response text itself still comes from data/knowledge_base.json
via chatbot.kb.get_rule_response(), so the JSON KB remains the
authoritative source for what the bot actually says.
"""

# ---------------------------------------------------------------------------
# Python 3.10+ compatibility shim for Experta
# ---------------------------------------------------------------------------
# Experta (last released 2018) imports collections.Mapping which was moved
# to collections.abc in Python 3.10. We patch it defensively here so this
# module works under either 3.9 or 3.10+.
import collections
import collections.abc as _abc
for _name in ("Mapping", "MutableMapping", "Sequence"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(_abc, _name))

from experta import KnowledgeEngine, Rule, Fact, Field, MATCH, P

from chatbot.kb import (
    get_intent_response,
    get_faq_answer,
    get_rule_response,
    get_fallback_response,
)


# ---------------------------------------------------------------------------
# Fact: the conversation state, declared so Experta can pattern-match it
# ---------------------------------------------------------------------------

class JourneyState(Fact):
    """
    Represents the current booking-conversation state as a single Fact.

    Each field corresponds to a key the conversation manager stores in
    its state dictionary. Missing values are represented as None so we
    can match on absence via P(lambda x: x is None).
    """
    origin_crs    = Field(object, default=None)
    dest_crs      = Field(object, default=None)
    travel_date   = Field(object, default=None)
    ticket_type   = Field(object, default=None)
    return_date   = Field(object, default=None)


# Convenience matchers
_IS_NONE = P(lambda x: x is None)
_NOT_NONE = P(lambda x: x is not None)


# ---------------------------------------------------------------------------
# ChatbotEngine: the rule-based reasoning engine
# ---------------------------------------------------------------------------

class ChatbotEngine(KnowledgeEngine):
    """
    Forward-chaining engine. Rules are checked in salience order
    (highest first); the first one whose pattern matches the current
    JourneyState fires, sets self.result to (action, response), and
    halts inference.
    """

    def __init__(self):
        super().__init__()
        # Output captured by whichever rule fires
        self.result = ("confirm_details",
                       "I think I have everything. Let me confirm your details.")

    # -- Salience 100: ask for origin if it's missing ---------------------
    @Rule(JourneyState(origin_crs=_IS_NONE), salience=100)
    def ask_origin(self):
        self.result = ("ask_origin", get_rule_response("origin_missing"))
        self.halt()

    # -- Salience 90: ask for destination if origin is set but dest isn't -
    @Rule(JourneyState(origin_crs=_NOT_NONE, dest_crs=_IS_NONE), salience=90)
    def ask_destination(self):
        self.result = ("ask_destination", get_rule_response("destination_missing"))
        self.halt()

    # -- Salience 80: ask for date if origin and dest are set --------------
    @Rule(JourneyState(origin_crs=_NOT_NONE,
                       dest_crs=_NOT_NONE,
                       travel_date=_IS_NONE), salience=80)
    def ask_date(self):
        self.result = ("ask_date", get_rule_response("date_missing"))
        self.halt()

    # -- Salience 70: ask for ticket type ----------------------------------
    @Rule(JourneyState(origin_crs=_NOT_NONE,
                       dest_crs=_NOT_NONE,
                       travel_date=_NOT_NONE,
                       ticket_type=_IS_NONE), salience=70)
    def ask_ticket_type(self):
        self.result = ("ask_ticket_type", get_rule_response("ticket_type_missing"))
        self.halt()

    # -- Salience 60: ask for return date (only if ticket_type=="return") --
    @Rule(JourneyState(origin_crs=_NOT_NONE,
                       dest_crs=_NOT_NONE,
                       travel_date=_NOT_NONE,
                       ticket_type=MATCH.t & P(lambda t: t == "return"),
                       return_date=_IS_NONE), salience=60)
    def ask_return_date(self):
        self.result = ("ask_return_date", get_rule_response("return_date_missing"))
        self.halt()

    # -- Salience 10: everything is collected -> confirm -------------------
    # Two firing conditions:
    #   (a) single ticket and all base slots filled
    #   (b) return ticket and return_date also filled
    @Rule(JourneyState(origin_crs=_NOT_NONE,
                       dest_crs=_NOT_NONE,
                       travel_date=_NOT_NONE,
                       ticket_type=MATCH.t & P(lambda t: t == "single")),
          salience=10)
    def confirm_single(self):
        self.result = ("confirm_details", get_rule_response("all_collected"))
        self.halt()

    @Rule(JourneyState(origin_crs=_NOT_NONE,
                       dest_crs=_NOT_NONE,
                       travel_date=_NOT_NONE,
                       ticket_type=MATCH.t & P(lambda t: t == "return"),
                       return_date=_NOT_NONE),
          salience=10)
    def confirm_return(self):
        self.result = ("confirm_details", get_rule_response("all_collected"))
        self.halt()


# ---------------------------------------------------------------------------
# Public API (unchanged signatures)
# ---------------------------------------------------------------------------

def infer_next_action(state: dict) -> tuple[str, str]:
    """
    Apply forward-chaining over the rule set to determine the next
    action in the conversation.

    Parameters
    ----------
    state : dict
        Current conversation state. The keys we care about are:
        origin_crs, dest_crs, travel_date, ticket_type, return_date.

    Returns
    -------
    (action, response) : tuple[str, str]
        action   - one of: 'ask_origin', 'ask_destination', 'ask_date',
                   'ask_ticket_type', 'ask_return_date', 'confirm_details'
        response - the prompt text to show the user, sourced from the
                   JSON knowledge base via get_rule_response()
    """
    engine = ChatbotEngine()
    engine.reset()
    engine.declare(JourneyState(
        origin_crs=state.get("origin_crs"),
        dest_crs=state.get("dest_crs"),
        travel_date=state.get("travel_date"),
        ticket_type=state.get("ticket_type"),
        return_date=state.get("return_date"),
    ))
    engine.run()
    return engine.result


def handle_intent(intent: str, user_text: str, state: dict):
    """
    Handle recognised intents that are NOT part of the main booking
    flow (greetings, goodbyes, help, FAQ questions). Returns the
    response string, or None if the intent should fall through to the
    conversation manager.
    """
    if intent in ("greet", "goodbye", "help"):
        return get_intent_response(intent)

    faq_answer = get_faq_answer(user_text)
    if faq_answer:
        return faq_answer

    return None


def get_fallback(user_text: str) -> str:
    """
    Last-resort response: try a lower-threshold FAQ match first, then
    fall back to a generic 'I didn't understand that' message.
    """
    faq_answer = get_faq_answer(user_text, threshold=50)
    if faq_answer:
        return faq_answer
    return get_fallback_response()
