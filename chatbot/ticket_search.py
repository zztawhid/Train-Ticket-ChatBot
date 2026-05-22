from __future__ import annotations

"""
ticket_search.py
----------------
National Rail Online Journey Planner (OJP) API integration.

Uses the zeep SOAP client to call the RealtimeJourneyPlan operation
on the OJP WSDL endpoint.  Parses the response to find the cheapest
available fare and builds a booking hyperlink.

WSDL: https://ojp.nationalrail.co.uk/webservices/jpservices.wsdl
Auth: HTTP Basic + WSSE UsernameToken (credentials from .env)
"""

import os
import logging
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv
from zeep import Client, Settings
from zeep.wsse.username import UsernameToken
from zeep.transports import Transport

from chatbot.station_lookup import crs_to_name

# ---------------------------------------------------------------------------
# Load credentials from .env
# ---------------------------------------------------------------------------
load_dotenv()
_USERNAME = os.getenv("RAIL_USERNAME", "")
_PASSWORD = os.getenv("RAIL_PASSWORD", "")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy SOAP client initialisation (avoids slow startup if not needed)
# ---------------------------------------------------------------------------
_client: Optional[Client] = None
_WSDL_URL = "https://ojp.nationalrail.co.uk/webservices/jpservices.wsdl"


def _get_client() -> Client:
    """Create (or return cached) zeep SOAP client."""
    global _client
    if _client is None:
        session = requests.Session()
        session.auth = requests.auth.HTTPBasicAuth(_USERNAME, _PASSWORD)
        transport = Transport(session=session, timeout=30)
        settings = Settings(strict=False, xml_huge_tree=True)
        _client = Client(
            wsdl=_WSDL_URL,
            transport=transport,
            wsse=UsernameToken(_USERNAME, _PASSWORD),
            settings=settings,
        )
    return _client


# ---------------------------------------------------------------------------
# Journey search
# ---------------------------------------------------------------------------

def search_cheapest_ticket(
    origin_crs: str,
    dest_crs: str,
    outward_datetime: str,
    outward_constraint: str = "departBy",
    inward_datetime: str | None = None,
    inward_constraint: str = "departBy",
    adults: int = 1,
    children: int = 0,
) -> dict:
    """
    Search for the cheapest train ticket using the OJP SOAP API.

    Parameters
    ----------
    origin_crs        : 3-letter CRS code of departure station
    dest_crs          : 3-letter CRS code of destination station
    outward_datetime  : ISO datetime string for outward travel
    outward_constraint: 'departBy' or 'arriveBy'
    inward_datetime   : ISO datetime string for return (None = single)
    inward_constraint : 'departBy' or 'arriveBy'
    adults            : number of adult passengers (1-8)
    children          : number of child passengers (0-8)

    Returns
    -------
    dict with keys:
        success       : bool
        error         : str | None
        cheapest_fare : dict | None  — fare details
        outward       : dict | None  — outward journey details
        inward        : dict | None  — inward journey details
        booking_url   : str | None   — National Rail booking link
        is_return     : bool
    """
    result = {
        "success": False,
        "error": None,
        "cheapest_fare": None,
        "outward": None,
        "inward": None,
        "booking_url": None,
        "is_return": inward_datetime is not None,
    }

    try:
        client = _get_client()

        # Build the outward time element
        outward_time = {outward_constraint: outward_datetime}

        # Build the request payload
        journey_request = {
            "origin": {"stationCRS": origin_crs},
            "destination": {"stationCRS": dest_crs},
            "realtimeEnquiry": "STANDARD",
            "outwardTime": outward_time,
            "directTrains": False,
            "fareRequestDetails": {
                "passengers": {"adult": adults, "child": children},
                "fareClass": "ANY",
            },
        }

        # Add inward time for return tickets
        if inward_datetime:
            journey_request["inwardTime"] = {inward_constraint: inward_datetime}

        # Make the SOAP call
        logger.info("Calling RealtimeJourneyPlan: %s -> %s", origin_crs, dest_crs)
        response = client.service.RealtimeJourneyPlan(**journey_request)

        # --- Parse outward journeys ---
        # Track cheapest single and cheapest return fare separately so we can
        # compare them properly instead of always picking the cheapest per-leg fare.
        cheapest_out_price = float("inf")      # cheapest single-leg outward fare
        cheapest_out_fare = None
        best_outward = None
        cheapest_out_return_price = float("inf")  # cheapest return fare on outward leg
        cheapest_out_return_fare = None
        best_outward_return = None

        if hasattr(response, "outwardJourney") and response.outwardJourney:
            for journey in response.outwardJourney:
                if hasattr(journey, "fare") and journey.fare:
                    fares = journey.fare if isinstance(journey.fare, list) else [journey.fare]
                    for fare in fares:
                        price = _get_fare_price(fare)
                        if price is None:
                            continue
                        if _is_return_fare(fare):
                            if price < cheapest_out_return_price:
                                cheapest_out_return_price = price
                                cheapest_out_return_fare = fare
                                best_outward_return = journey
                        else:
                            if price < cheapest_out_price:
                                cheapest_out_price = price
                                cheapest_out_fare = fare
                                best_outward = journey

        # Fall back: if no explicit single was found, use the return fare slot too
        if not best_outward and not best_outward_return:
            result["error"] = "No fares were found for the outward journey."
            return result

        # --- Parse inward journeys (if return) ---
        cheapest_in_price = float("inf")
        cheapest_in_fare = None
        best_inward = None

        if inward_datetime and hasattr(response, "inwardJourney") and response.inwardJourney:
            for journey in response.inwardJourney:
                if hasattr(journey, "fare") and journey.fare:
                    fares = journey.fare if isinstance(journey.fare, list) else [journey.fare]
                    for fare in fares:
                        price = _get_fare_price(fare)
                        if price is not None and price < cheapest_in_price:
                            cheapest_in_price = price
                            cheapest_in_fare = fare
                            best_inward = journey

        # --- Calculate total fare ---
        # For return requests compare: dedicated return fare vs sum of two singles.
        # Use whichever is cheaper.
        return_fare_missing = False

        if inward_datetime:
            # Price of a dedicated return ticket (covers both legs)
            return_ticket_total = cheapest_out_return_price  # already in same unit

            # Price of two singles added together
            singles_total = (
                cheapest_out_price + cheapest_in_price
                if cheapest_in_price != float("inf") and cheapest_out_price != float("inf")
                else float("inf")
            )

            if return_ticket_total <= singles_total and cheapest_out_return_fare is not None:
                # A dedicated return fare is cheaper (or equal)
                total_price = return_ticket_total
                fare_type = getattr(cheapest_out_return_fare, "description", "Return")
                best_outward = best_outward_return
                best_inward = best_inward  # still show the inward timetable if available
            elif singles_total != float("inf"):
                # Two singles are cheaper
                out_type = getattr(cheapest_out_fare, "description", "Single")
                in_type = getattr(cheapest_in_fare, "description", "Single")
                total_price = singles_total
                fare_type = f"{out_type} + {in_type}"
            elif cheapest_out_return_fare is not None:
                # Only a return fare was found (no singles available)
                total_price = return_ticket_total
                fare_type = getattr(cheapest_out_return_fare, "description", "Return")
                best_outward = best_outward_return
            elif cheapest_out_fare is not None:
                # Only an outward single found, no inward — warn user
                return_fare_missing = True
                total_price = cheapest_out_price
                fare_type = getattr(cheapest_out_fare, "description", "Single")
            else:
                result["error"] = "No fares were found for the requested journey."
                return result
        else:
            # Single journey — just use cheapest outward fare
            if cheapest_out_fare is not None:
                total_price = cheapest_out_price
                fare_type = getattr(cheapest_out_fare, "description", "Single")
            else:
                # Only a return fare was found even for a single request
                total_price = cheapest_out_return_price
                fare_type = getattr(cheapest_out_return_fare, "description", "Return")
                best_outward = best_outward_return

        # OJP returns prices in pence (integers); always divide by 100.
        # If the value is a small float (e.g. 42.50) it was already in pounds.
        if isinstance(total_price, float) and total_price < 500:
            total_pounds = total_price
        else:
            total_pounds = total_price / 100.0

        # --- Extract journey timetable details ---
        outward_details = _extract_journey_details(best_outward, origin_crs, dest_crs)
        inward_details = _extract_journey_details(best_inward, dest_crs, origin_crs) if best_inward else None

        result["cheapest_fare"] = {
            "total_price": total_pounds,
            "fare_type": fare_type,
            "return_fare_missing": return_fare_missing,
        }
        result["outward"] = outward_details
        result["inward"] = inward_details

        # --- Build booking URL ---
        result["booking_url"] = _build_booking_url(
            origin_crs, dest_crs,
            outward_datetime, inward_datetime,
            adults,
        )
        result["success"] = True

    except Exception as e:
        logger.error("OJP API error: %s", str(e))
        result["error"] = f"API error: {str(e)}"

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_return_fare(fare) -> bool:
    """
    Return True if this fare covers both outward and return legs
    (i.e. it is a return/round-trip ticket rather than a single).
    """
    desc = getattr(fare, "description", "") or ""
    desc_lower = str(desc).lower()
    return any(kw in desc_lower for kw in ("return", " rtn", "rtn ", "round trip", "open return"))


def _get_fare_price(fare) -> float | None:
    """
    Extract the numeric price from a fare object.
    Tries several common attribute names used by the OJP API.
    Returns the price as a number, or None if not found.
    """
    for attr in ("totalPrice", "ticketPrice", "price", "farePrice"):
        val = getattr(fare, attr, None)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def _extract_journey_details(journey, origin_crs: str, dest_crs: str) -> dict:
    """Pull departure/arrival times and other details from a journey object."""
    details = {
        "origin": crs_to_name(origin_crs),
        "origin_crs": origin_crs,
        "destination": crs_to_name(dest_crs),
        "dest_crs": dest_crs,
        "departure_time": None,
        "arrival_time": None,
        "changes": 0,
        "change_stations": [],
        "legs_detail": [],
        "duration": None,
    }

    try:
        timetable = getattr(journey, "timetable", None)
        if timetable and hasattr(timetable, "scheduled"):
            sched = timetable.scheduled
            if hasattr(sched, "departure") and sched.departure:
                details["departure_time"] = sched.departure.strftime("%H:%M")
                details["departure_date"] = sched.departure.strftime("%d/%m/%Y")
            if hasattr(sched, "arrival") and sched.arrival:
                details["arrival_time"] = sched.arrival.strftime("%H:%M")
                details["arrival_date"] = sched.arrival.strftime("%d/%m/%Y")

            # Calculate duration
            if sched.departure and sched.arrival:
                delta = sched.arrival - sched.departure
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                mins = remainder // 60
                details["duration"] = f"{hours}h {mins}m"

        # Number of changes (legs - 1)
        if hasattr(journey, "leg"):
            legs = journey.leg if isinstance(journey.leg, list) else [journey.leg]
            details["changes"] = max(0, len(legs) - 1)
            details["change_stations"] = _extract_change_stations(legs)
            details["legs_detail"] = _extract_legs_detail(legs)

    except Exception as e:
        logger.warning("Could not extract journey details: %s", e)

    return details


def _extract_change_stations(legs: list) -> list[str]:
    """Best-effort list of interchange stations from journey legs."""
    if not legs or len(legs) < 2:
        return []

    change_points = []
    for leg in legs[:-1]:
        label = _extract_leg_destination_label(leg)
        if label:
            change_points.append(label)

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for name in change_points:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _extract_legs_detail(legs: list) -> list[dict]:
    """Extract operator info per leg."""
    result = []
    for leg in legs:
        operator = None
        for attr in ("operator", "operatorCode", "toc", "trainOperator", "serviceOperator",
                     "operatorName", "tocCode", "atocCode", "atocName"):
            val = getattr(leg, attr, None)
            if val:
                operator = _extract_operator_name(val)
                break
        if not operator:
            svc = getattr(leg, "service", None) or getattr(leg, "train", None)
            if svc:
                for attr in ("operator", "operatorCode", "toc", "atocCode", "atocName", "operatorName"):
                    val = getattr(svc, attr, None)
                    if val:
                        operator = _extract_operator_name(val)
                        break
        result.append({"operator": operator})
    return result


def _extract_operator_name(val) -> str:
    """Return a clean operator name from a zeep object, dict, or string."""
    if hasattr(val, "name") and val.name:
        return str(val.name)
    if hasattr(val, "code") and val.code:
        return str(val.code)
    s = str(val)
    # Handle zeep rendering as "{ 'code': 'LE', 'name': 'Greater Anglia' }"
    import re
    m = re.search(r"'name':\s*'([^']+)'", s)
    if m:
        return m.group(1)
    return s


def _extract_leg_destination_label(leg) -> str | None:
    """Try to resolve a leg's destination into a displayable station label."""
    for attr in ("destination", "alight", "alighting", "to", "arrival", "arrive", "end"):
        obj = getattr(leg, attr, None)
        label = _extract_location_label(obj)
        if label:
            return label

    return _extract_location_label(getattr(leg, "location", None))


def _extract_location_label(obj) -> str | None:
    """Extract a station name or CRS from a nested location-like object."""
    if obj is None:
        return None

    loc = getattr(obj, "location", None) or obj
    name = None
    for attr in ("locationName", "name"):
        name = getattr(loc, attr, None)
        if name:
            break

    crs = None
    for attr in ("stationCRS", "locationCRS", "crs"):
        crs = getattr(loc, attr, None)
        if crs:
            break

    if name:
        return str(name)
    if crs:
        return crs_to_name(str(crs))
    return None


def _build_booking_url(
    origin_crs: str,
    dest_crs: str,
    outward_dt: str,
    inward_dt: str | None,
    adults: int,
) -> str:
    """
    Build a National Rail journey planner URL that pre-fills the search
    so the user can click through to book.
    """
    try:
        out_dt = datetime.fromisoformat(outward_dt)
        out_date = out_dt.strftime("%d%m%y")
        out_time = out_dt.strftime("%H%M")
    except (ValueError, TypeError):
        out_date = ""
        out_time = "0900"

    # OJP path format:
    #   Single: /service/timesandfares/{from}/{to}/{date}/{time}/dep
    #   Return: /service/timesandfares/{from}/{to}/{outDate}/{outTime}/dep/{inDate}/{inTime}/dep
    return_segment = ""

    if inward_dt:
        try:
            in_dt = datetime.fromisoformat(inward_dt)
            in_date = in_dt.strftime("%d%m%y")
            in_time = in_dt.strftime("%H%M")
            return_segment = f"/{in_date}/{in_time}/dep"
        except (ValueError, TypeError):
            pass

    url = (
        f"https://ojp.nationalrail.co.uk/service/timesandfares"
        f"/{origin_crs}/{dest_crs}/{out_date}/{out_time}/dep"
        f"{return_segment}"
    )


    return url
