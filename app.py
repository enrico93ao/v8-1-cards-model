import html
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from statistics import median
from zoneinfo import ZoneInfo

import requests
import streamlit as st

from predict_v8_1 import load_bundle, predict_match


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="V8.1 Cards Model",
    page_icon="🟨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ODDSPAPI_KEY = os.getenv("ODDSPAPI_KEY")
ODDSPAPI_BASE = "https://api.oddspapi.io/v4"
SOCCER_ID = 10
BOOKMAKER = "pinnacle"
TZ_ITALY = ZoneInfo("Europe/Rome")
ODDS_ENDPOINT_COOLDOWN = 1.15

BIG5 = {
    "Premier League": {
        "names": ["Premier League"],
        "country": "England",
        "model_id": "E0",
    },
    "La Liga": {
        "names": ["LaLiga", "La Liga"],
        "country": "Spain",
        "model_id": "SP1",
    },
    "Bundesliga": {
        "names": ["Bundesliga"],
        "country": "Germany",
        "model_id": "D1",
    },
    "Serie A": {
        "names": ["Serie A"],
        "country": "Italy",
        "model_id": "I1",
    },
    "Ligue 1": {
        "names": ["Ligue 1"],
        "country": "France",
        "model_id": "F1",
    },
}


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()


def pct(value):
    return f"{float(value) * 100:.1f}%"


# ============================================================
# VISUAL STYLE
# ============================================================

def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1020;
            --panel: #121a2d;
            --panel-2: #172238;
            --line: rgba(255,255,255,.08);
            --text: #f6f8fc;
            --muted: #9aa9c7;
            --blue: #6ea8fe;
            --green: #46e6a6;
            --yellow: #ffd166;
            --orange: #ff9f43;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(55, 91, 210, .16), transparent 28%),
                radial-gradient(circle at 95% 10%, rgba(255, 193, 7, .08), transparent 22%),
                linear-gradient(180deg, #0a0f1c 0%, #0d1424 100%);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }

        .v81-hero {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(23,34,56,.96), rgba(13,20,36,.96));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 23px 26px;
            box-shadow: 0 18px 60px rgba(0,0,0,.24);
            margin-bottom: 14px;
        }

        .v81-hero::after {
            content: "";
            position: absolute;
            width: 250px;
            height: 250px;
            right: -70px;
            top: -130px;
            border-radius: 50%;
            background: rgba(255, 209, 102, .08);
        }

        .v81-title {
            font-size: 2.15rem;
            line-height: 1.05;
            font-weight: 900;
            letter-spacing: -.03em;
            color: #fff;
        }

        .v81-subtitle {
            margin-top: 8px;
            color: var(--muted);
            font-size: .97rem;
        }

        .v81-badge {
            display: inline-block;
            margin-top: 13px;
            margin-right: 7px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(70,230,166,.25);
            background: rgba(70,230,166,.10);
            color: #8af0c1;
            font-size: .78rem;
            font-weight: 800;
        }

        .section-head {
            margin: 16px 0 9px;
            font-size: 1.18rem;
            font-weight: 900;
            color: #fff;
        }

        .kpi-card {
            height: 105px;
            background: linear-gradient(180deg, rgba(24,35,57,.96), rgba(17,26,45,.96));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 15px 17px;
            box-shadow: 0 10px 28px rgba(0,0,0,.16);
        }

        .kpi-label {
            color: var(--muted);
            font-size: .80rem;
            font-weight: 700;
        }

        .kpi-value {
            color: #fff;
            font-size: 1.78rem;
            line-height: 1.15;
            margin-top: 5px;
            font-weight: 900;
        }

        .kpi-sub {
            color: #7182a5;
            font-size: .71rem;
            margin-top: 4px;
        }

        .pick-card {
            min-height: 254px;
            background: linear-gradient(180deg, rgba(25,37,61,.98), rgba(15,23,40,.98));
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 17px;
            box-shadow: 0 15px 36px rgba(0,0,0,.20);
        }

        .pick-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 11px;
        }

        .pick-rank {
            color: #111827;
            background: linear-gradient(90deg, #ffd166, #ffb703);
            border-radius: 999px;
            padding: 5px 9px;
            font-size: .72rem;
            font-weight: 900;
        }

        .pick-league {
            color: #7f91b6;
            font-size: .72rem;
            font-weight: 700;
        }

        .pick-match {
            color: #fff;
            font-size: 1.07rem;
            font-weight: 900;
            line-height: 1.25;
            min-height: 47px;
        }

        .pick-date {
            color: var(--muted);
            font-size: .77rem;
            margin-top: 4px;
        }

        .market-pill {
            margin-top: 12px;
            display: inline-block;
            max-width: 100%;
            color: #a9caff;
            background: rgba(79, 129, 255, .12);
            border: 1px solid rgba(110,168,254,.20);
            border-radius: 10px;
            padding: 7px 9px;
            font-size: .80rem;
            font-weight: 800;
        }

        .prob-row {
            display: flex;
            justify-content: space-between;
            align-items: end;
            margin-top: 14px;
        }

        .prob-label {
            color: var(--muted);
            font-size: .72rem;
            font-weight: 700;
        }

        .prob-value {
            font-size: 1.72rem;
            line-height: 1;
            font-weight: 950;
        }

        .confidence-pill {
            padding: 5px 8px;
            border-radius: 999px;
            font-size: .67rem;
            font-weight: 900;
            border: 1px solid var(--line);
            background: rgba(255,255,255,.04);
            color: #c8d4eb;
        }

        .odds-line {
            margin-top: 13px;
            color: #90a1c2;
            font-size: .73rem;
        }

        .quad-card {
            min-height: 184px;
            background: linear-gradient(180deg, rgba(22,33,54,.96), rgba(14,22,38,.96));
            border: 1px solid var(--line);
            border-radius: 17px;
            padding: 14px;
        }

        .quad-number {
            color: #ffd166;
            font-size: .70rem;
            font-weight: 900;
        }

        .quad-match {
            color: #fff;
            margin-top: 7px;
            min-height: 43px;
            font-size: .91rem;
            line-height: 1.25;
            font-weight: 900;
        }

        .quad-market {
            color: #9fc4ff;
            font-size: .76rem;
            margin-top: 8px;
            min-height: 38px;
            font-weight: 700;
        }

        .quad-prob {
            margin-top: 8px;
            font-size: 1.26rem;
            font-weight: 950;
        }

        .mini-note {
            color: #7385a8;
            font-size: .73rem;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 12px !important;
        }

        .stButton > button {
            border-radius: 12px;
            font-weight: 800;
            min-height: 42px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 7px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 11px 11px 0 0;
            font-weight: 750;
        }

        [data-testid="stMetric"] {
            background: rgba(20,29,48,.55);
            border: 1px solid var(--line);
            padding: 12px;
            border-radius: 14px;
        }

        @media (max-width: 900px) {
            .v81-title { font-size: 1.72rem; }
            .pick-card { min-height: auto; }
            .quad-card { min-height: auto; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def probability_style(probability):
    p = float(probability)
    if p >= 0.80:
        return "#46e6a6", "ALTA"
    if p >= 0.70:
        return "#ffd166", "BUONA"
    return "#ff9f43", "MEDIA"


def render_hero(generated_at):
    generated = html.escape(str(generated_at))
    st.markdown(
        f"""
        <div class="v81-hero">
            <div class="v81-title">🟨 V8.1 Cards Model</div>
            <div class="v81-subtitle">
                Auto Scanner Big Five · ranking probabilistico pre-match sui cartellini
            </div>
            <span class="v81-badge">● LIVE</span>
            <span class="v81-badge">Aggiornato {generated}</span>
            <span class="v81-badge">Pinnacle · OddsPapi</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{html.escape(str(label))}</div>
            <div class="kpi-value">{html.escape(str(value))}</div>
            <div class="kpi-sub">{html.escape(str(sub))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pick_card(pick, index):
    color, confidence = probability_style(pick["probability"])
    st.markdown(
        f"""
        <div class="pick-card">
            <div class="pick-top">
                <span class="pick-rank">TOP {index}</span>
                <span class="pick-league">{html.escape(pick['league'])}</span>
            </div>
            <div class="pick-match">{html.escape(pick['home'])} – {html.escape(pick['away'])}</div>
            <div class="pick-date">{html.escape(pick['kickoff'])}</div>
            <div class="market-pill">{html.escape(pick['market'])}</div>
            <div class="prob-row">
                <div>
                    <div class="prob-label">PROBABILITÀ V8.1</div>
                    <div class="prob-value" style="color:{color}">{pct(pick['probability'])}</div>
                </div>
                <span class="confidence-pill">{confidence}</span>
            </div>
            <div class="odds-line">
                1 {pick['odds_h']:.2f} · X {pick['odds_d']:.2f} · 2 {pick['odds_a']:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quad_card(pick, index):
    color, _ = probability_style(pick["probability"])
    st.markdown(
        f"""
        <div class="quad-card">
            <div class="quad-number">SELEZIONE {index}</div>
            <div class="quad-match">{html.escape(pick['home'])} – {html.escape(pick['away'])}</div>
            <div class="mini-note">{html.escape(pick['league'])} · {html.escape(pick['kickoff'])}</div>
            <div class="quad-market">{html.escape(pick['market'])}</div>
            <div class="quad-prob" style="color:{color}">{pct(pick['probability'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TEAM NAME MATCHING
# ============================================================

def normalize_name(name):
    if not name:
        return ""

    value = unicodedata.normalize("NFKD", str(name))
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)

    removable = {
        "fc", "afc", "cf", "ac", "as", "ssc", "club", "football",
    }
    words = [word for word in value.split() if word not in removable]
    return " ".join(words)


RAW_ALIASES = {
    # Premier League
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",

    # La Liga
    "Real Sociedad": "Sociedad",
    "Real Sociedad San Sebastian": "Sociedad",
    "Real Sociedad San Sebastián": "Sociedad",
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Atlético Madrid": "Ath Madrid",
    "Real Betis": "Betis",
    "Espanyol Barcelona": "Espanol",
    "RCD Espanyol": "Espanol",
    "Celta Vigo": "Celta",
    "RC Celta de Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alaves": "Alaves",
    "Deportivo Alavés": "Alaves",
    "Real Mallorca": "Mallorca",

    # Bundesliga
    "Borussia Monchengladbach": "M'gladbach",
    "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Bayer Leverkusen": "Leverkusen",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "1899 Hoffenheim": "Hoffenheim",
    "TSG Hoffenheim": "Hoffenheim",
    "1. FC Heidenheim": "Heidenheim",
    "FC Heidenheim": "Heidenheim",
    "FC St. Pauli": "St Pauli",
    "FSV Mainz": "Mainz",
    "FSV Mainz 05": "Mainz",
    "1. FSV Mainz 05": "Mainz",

    # Serie A
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "FC Internazionale Milano": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",

    # Ligue 1
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Olympique Marseille": "Marseille",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "Stade Rennais": "Rennes",
    "AS Monaco": "Monaco",
    "RC Lens": "Lens",
    "LOSC Lille": "Lille",
    "Lille OSC": "Lille",
    "OGC Nice": "Nice",
}

TEAM_ALIASES = {normalize_name(key): value for key, value in RAW_ALIASES.items()}


def get_model_teams(model_league):
    return sorted({
        team
        for league, team in bundle["state"]["team_hist"].keys()
        if league == model_league
    })


def match_model_team(api_name, model_league):
    candidates = get_model_teams(model_league)
    if not candidates:
        return None

    normalized_candidates = {normalize_name(team): team for team in candidates}
    query = normalize_name(api_name)

    if query in normalized_candidates:
        return normalized_candidates[query]

    alias = TEAM_ALIASES.get(query)
    if alias:
        alias_normalized = normalize_name(alias)
        if alias_normalized in normalized_candidates:
            return normalized_candidates[alias_normalized]

    best_team = None
    best_score = 0.0
    for team in candidates:
        score = SequenceMatcher(None, query, normalize_name(team)).ratio()
        if score > best_score:
            best_score = score
            best_team = team

    return best_team if best_score >= 0.80 else None


def identify_big5_fixture(fixture):
    tournament_name = str(fixture.get("tournamentName", "")).strip()
    category_name = str(fixture.get("categoryName", "")).strip()

    for display_name, cfg in BIG5.items():
        valid_name = any(
            tournament_name.lower() == name.lower()
            for name in cfg["names"]
        )
        valid_country = category_name.lower() == cfg["country"].lower()
        if valid_name and valid_country:
            return {**cfg, "display_name": display_name}

    return None


# ============================================================
# ODDSPAPI
# ============================================================

def safe_message(message):
    value = str(message)
    if ODDSPAPI_KEY:
        value = value.replace(ODDSPAPI_KEY, "***API_KEY_NASCOSTA***")
    value = re.sub(r"apiKey=[^&\s]+", "apiKey=***API_KEY_NASCOSTA***", value)
    return value


def _retry_wait_seconds(response, default=1.15):
    try:
        body = response.json()
    except Exception:
        body = {}

    retry_ms = 0
    if isinstance(body, dict):
        error = body.get("error", {})
        if isinstance(error, dict):
            retry_ms = error.get("retryMs", 0) or 0

    try:
        retry_seconds = float(retry_ms) / 1000.0
    except (TypeError, ValueError):
        retry_seconds = 0.0

    return max(default, retry_seconds + 0.15)


def oddspapi_get(endpoint, params=None, max_retries=3):
    if not ODDSPAPI_KEY:
        raise RuntimeError("ODDSPAPI_KEY non configurata su Render.")

    final_params = dict(params or {})
    final_params["apiKey"] = ODDSPAPI_KEY

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                f"{ODDSPAPI_BASE}/{endpoint}",
                params=final_params,
                timeout=40,
            )
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(1.2 * (attempt + 1))
                continue
            raise RuntimeError(
                "Errore di connessione con OddsPapi: " + safe_message(exc)
            ) from exc

        if response.status_code == 429:
            if attempt < max_retries:
                time.sleep(_retry_wait_seconds(response))
                continue
            raise RuntimeError(
                "OddsPapi rate limit: tentativi automatici esauriti."
            )

        if not response.ok:
            body = response.text[:900]
            raise RuntimeError(
                f"OddsPapi HTTP {response.status_code}: {safe_message(body)}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError("OddsPapi ha restituito una risposta non JSON.") from exc

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError("OddsPapi: " + safe_message(data.get("error")))

        return data

    raise RuntimeError("Errore OddsPapi non gestito.")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_upcoming_fixtures():
    now_utc = datetime.now(timezone.utc)
    end_utc = now_utc + timedelta(days=7)

    from_iso = now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    to_iso = end_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    data = oddspapi_get(
        "fixtures",
        {
            "sportId": SOCCER_ID,
            "from": from_iso,
            "to": to_iso,
            "statusId": 0,
            "hasOdds": "true",
            "bookmakers": BOOKMAKER,
            "language": "en",
        },
    )

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("fixtures", "response", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def normalize_odds_response(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("fixtures", "response", "data", "items", "events"):
            if isinstance(data.get(key), list):
                return data[key]
        if data.get("fixtureId"):
            return [data]
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tournament_odds(tournament_id):
    data = oddspapi_get(
        "odds-by-tournaments",
        {
            "tournamentIds": str(tournament_id),
            "bookmakers": BOOKMAKER,
            "language": "en",
            "verbosity": 3,
        },
    )
    return normalize_odds_response(data)


def get_active_price(outcome):
    players = outcome.get("players", {})
    if not isinstance(players, dict):
        return None

    main_line_prices = []
    other_prices = []

    for player in players.values():
        if not isinstance(player, dict) or player.get("active") is False:
            continue
        try:
            price = float(player.get("price"))
        except (TypeError, ValueError):
            continue
        if price <= 1.0:
            continue
        if player.get("mainLine") is True:
            main_line_prices.append(price)
        else:
            other_prices.append(price)

    if main_line_prices:
        return float(median(main_line_prices))
    if other_prices:
        return float(median(other_prices))
    return None


def extract_1x2(odds_item):
    bookmaker_odds = odds_item.get("bookmakerOdds") or {}
    bookmaker_data = bookmaker_odds.get(BOOKMAKER) or {}

    if not bookmaker_data:
        return None
    if bookmaker_data.get("bookmakerIsActive") is False:
        return None
    if bookmaker_data.get("suspended") is True:
        return None

    markets = bookmaker_data.get("markets") or {}
    market = markets.get("101") or markets.get(101)
    if not market or market.get("marketActive") is False:
        return None

    outcomes = market.get("outcomes") or {}
    home_outcome = outcomes.get("101") or outcomes.get(101) or {}
    draw_outcome = outcomes.get("102") or outcomes.get(102) or {}
    away_outcome = outcomes.get("103") or outcomes.get(103) or {}

    home_price = get_active_price(home_outcome)
    draw_price = get_active_price(draw_outcome)
    away_price = get_active_price(away_outcome)

    if None in (home_price, draw_price, away_price):
        return None

    return {"home": home_price, "draw": draw_price, "away": away_price}


# ============================================================
# V8.1 RANKING
# ============================================================

def get_best_market(prediction, home_name, away_name):
    markets = prediction["markets"]
    possibilities = [
        ("Over 2.5 cartellini", markets["match_O2.5"]),
        ("Under 3.5 cartellini", markets["match_U3.5"]),
        (f"{home_name} Over 1.5 cartellini", markets["home_team_O1.5"]),
        (f"{home_name} Under 1.5 cartellini", markets["home_team_U1.5"]),
        (f"{away_name} Over 1.5 cartellini", markets["away_team_O1.5"]),
        (f"{away_name} Under 1.5 cartellini", markets["away_team_U1.5"]),
    ]
    market, probability = max(possibilities, key=lambda item: item[1])
    return market, float(probability)


def parse_datetime(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(TZ_ITALY)
    except Exception:
        return None


def kickoff_label(dt):
    return dt.strftime("%d/%m · %H:%M") if dt else "Data N/D"


@st.cache_data(ttl=86400, show_spinner=False)
def build_auto_scanner():
    fixtures = fetch_upcoming_fixtures()
    big5_fixtures = []

    for fixture in fixtures:
        cfg = identify_big5_fixture(fixture)
        if cfg is None:
            continue
        fixture_copy = dict(fixture)
        fixture_copy["_model_cfg"] = cfg
        big5_fixtures.append(fixture_copy)

    tournament_ids = sorted({
        int(fixture["tournamentId"])
        for fixture in big5_fixtures
        if fixture.get("tournamentId") is not None
    })

    odds_by_fixture = {}
    odds_errors = []
    last_odds_call = 0.0

    for tournament_id in tournament_ids:
        elapsed = time.monotonic() - last_odds_call
        if last_odds_call and elapsed < ODDS_ENDPOINT_COOLDOWN:
            time.sleep(ODDS_ENDPOINT_COOLDOWN - elapsed)

        try:
            odds_items = fetch_tournament_odds(tournament_id)
            last_odds_call = time.monotonic()

            for item in odds_items:
                fixture_id = str(item.get("fixtureId", ""))
                if fixture_id:
                    odds_by_fixture[fixture_id] = item
        except Exception as exc:
            last_odds_call = time.monotonic()
            odds_errors.append({
                "Tournament ID": tournament_id,
                "Errore": safe_message(exc),
            })

    rankings = []
    skipped = []
    fixture_meta = []

    for fixture in big5_fixtures:
        cfg = fixture["_model_cfg"]
        fixture_id = str(fixture.get("fixtureId", ""))
        api_home = fixture.get("participant1Name") or fixture.get("participant1ShortName") or ""
        api_away = fixture.get("participant2Name") or fixture.get("participant2ShortName") or ""
        kickoff_dt = parse_datetime(fixture.get("startTime"))

        fixture_meta.append({
            "fixture_id": fixture_id,
            "league": cfg["display_name"],
            "home": api_home,
            "away": api_away,
            "kickoff_dt": kickoff_dt,
        })

        if not api_home or not api_away:
            skipped.append({
                "Partita": fixture_id,
                "Campionato": cfg["display_name"],
                "Data": kickoff_label(kickoff_dt),
                "Motivo": "Nome squadra mancante",
            })
            continue

        model_home = match_model_team(api_home, cfg["model_id"])
        model_away = match_model_team(api_away, cfg["model_id"])

        if not model_home or not model_away:
            missing = api_home if not model_home else api_away
            skipped.append({
                "Partita": f"{api_home} - {api_away}",
                "Campionato": cfg["display_name"],
                "Data": kickoff_label(kickoff_dt),
                "Motivo": f"{missing} non presente nello storico V8.1 / nome non riconosciuto",
            })
            continue

        odds_item = odds_by_fixture.get(fixture_id)
        if not odds_item:
            skipped.append({
                "Partita": f"{api_home} - {api_away}",
                "Campionato": cfg["display_name"],
                "Data": kickoff_label(kickoff_dt),
                "Motivo": "Quote Pinnacle non trovate",
            })
            continue

        odds = extract_1x2(odds_item)
        if odds is None:
            skipped.append({
                "Partita": f"{api_home} - {api_away}",
                "Campionato": cfg["display_name"],
                "Data": kickoff_label(kickoff_dt),
                "Motivo": "Mercato 1X2 Pinnacle non disponibile",
            })
            continue

        try:
            result = predict_match(
                bundle=bundle,
                league=cfg["model_id"],
                home=model_home,
                away=model_away,
                oddsH=odds["home"],
                oddsD=odds["draw"],
                oddsA=odds["away"],
                referee=None,
                official=False,
            )
        except Exception as exc:
            skipped.append({
                "Partita": f"{api_home} - {api_away}",
                "Campionato": cfg["display_name"],
                "Data": kickoff_label(kickoff_dt),
                "Motivo": "Errore modello: " + safe_message(exc),
            })
            continue

        market, probability = get_best_market(result, api_home, api_away)
        rankings.append({
            "fixture_id": fixture_id,
            "league": cfg["display_name"],
            "home": api_home,
            "away": api_away,
            "kickoff_dt": kickoff_dt,
            "kickoff": kickoff_label(kickoff_dt),
            "market": market,
            "probability": probability,
            "score": probability,
            "odds_h": odds["home"],
            "odds_d": odds["draw"],
            "odds_a": odds["away"],
            "bookmaker": "Pinnacle",
        })

    rankings.sort(key=lambda item: item["score"], reverse=True)

    return {
        "rankings": rankings,
        "skipped": skipped,
        "odds_errors": odds_errors,
        "fixtures": fixture_meta,
        "generated_at": datetime.now(TZ_ITALY).strftime("%d/%m/%Y · %H:%M"),
    }


# ============================================================
# UI
# ============================================================

inject_custom_css()

if not ODDSPAPI_KEY:
    st.error("ODDSPAPI_KEY non configurata su Render.")
    st.stop()

with st.spinner("Aggiorno partite, quote Pinnacle e ranking V8.1..."):
    try:
        scanner = build_auto_scanner()
    except Exception as exc:
        st.error("Errore durante l'Auto Scanner.")
        st.code(safe_message(exc))
        st.stop()

render_hero(scanner["generated_at"])

# State for compact focus mode
if "focus_tripla" not in st.session_state:
    st.session_state.focus_tripla = False

control_a, control_b, control_c = st.columns([1.05, 1.6, 1.35])

with control_a:
    days_filter = st.selectbox(
        "Intervallo",
        [3, 5, 7],
        index=1,
        format_func=lambda x: f"Prossimi {x} giorni",
    )

with control_b:
    selected_leagues = st.multiselect(
        "Campionati",
        list(BIG5.keys()),
        default=list(BIG5.keys()),
        placeholder="Seleziona campionati",
    )

with control_c:
    minimum_probability = st.slider(
        "Probabilità minima",
        min_value=0.55,
        max_value=0.90,
        value=0.65,
        step=0.01,
        format="%.2f",
    )

now_italy = datetime.now(TZ_ITALY)
cutoff = now_italy + timedelta(days=days_filter)
selected_league_set = set(selected_leagues)

window_fixtures = [
    item for item in scanner["fixtures"]
    if (
        item["league"] in selected_league_set
        and item["kickoff_dt"] is not None
        and now_italy <= item["kickoff_dt"] <= cutoff
    )
]

rankings = [
    item for item in scanner["rankings"]
    if (
        item["league"] in selected_league_set
        and item["kickoff_dt"] is not None
        and now_italy <= item["kickoff_dt"] <= cutoff
    )
]

eligible = [
    item for item in rankings
    if item["probability"] >= minimum_probability
]

found_count = len(window_fixtures)
analyzed_count = len(rankings)
skipped_count = max(0, found_count - analyzed_count)

st.markdown('<div class="section-head">Panoramica scanner</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("Partite in programma", found_count, f"Finestra {days_filter} giorni")
with k2:
    render_kpi_card("Analizzate V8.1", analyzed_count, "Quote + squadre disponibili")
with k3:
    render_kpi_card("Sopra soglia", len(eligible), f"≥ {minimum_probability:.0%}")
with k4:
    render_kpi_card("Saltate", skipped_count, "Vedi Diagnostica")

focus1, focus2, spacer = st.columns([1, 1, 3.4])
with focus1:
    if st.button("🎯 Focus miglior tripla", use_container_width=True):
        st.session_state.focus_tripla = True
with focus2:
    if st.button("▦ Dashboard completa", use_container_width=True):
        st.session_state.focus_tripla = False

st.markdown('<div class="section-head">🏆 Miglior Tripla V8.1</div>', unsafe_allow_html=True)

if len(eligible) >= 3:
    top3 = eligible[:3]
    c1, c2, c3 = st.columns(3)
    with c1:
        render_pick_card(top3[0], 1)
    with c2:
        render_pick_card(top3[1], 2)
    with c3:
        render_pick_card(top3[2], 3)
else:
    st.warning("Non ci sono almeno 3 selezioni sopra la soglia impostata.")

if st.session_state.focus_tripla:
    st.caption(
        "Vista Focus attiva: mostra solo la tripla migliore. "
        "Usa “Dashboard completa” per tornare a tutte le sezioni."
    )
    st.stop()

st.markdown('<div class="section-head">🔥 Quadrupla V8.1</div>', unsafe_allow_html=True)

if len(eligible) >= 4:
    top4 = eligible[:4]
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        render_quad_card(top4[0], 1)
    with q2:
        render_quad_card(top4[1], 2)
    with q3:
        render_quad_card(top4[2], 3)
    with q4:
        render_quad_card(top4[3], 4)
else:
    st.warning("Non ci sono almeno 4 selezioni sopra la soglia impostata.")

st.caption(
    "Il ranking attuale ordina le selezioni per probabilità V8.1. "
    "Le quote 1X2 sono input del modello; l'EV sui mercati cartellini verrà aggiunto separatamente."
)

st.markdown('<div class="section-head">Dettagli</div>', unsafe_allow_html=True)

tab_dashboard, tab_all, tab_diag, tab_manual = st.tabs([
    "⭐ Selezioni",
    "📊 Tutte le analisi",
    "⚠️ Diagnostica",
    "🧪 Manuale",
])

with tab_dashboard:
    if eligible:
        rows = []
        for item in eligible:
            _, confidence = probability_style(item["probability"])
            rows.append({
                "Partita": f"{item['home']} - {item['away']}",
                "Campionato": item["league"],
                "Data": item["kickoff"],
                "Mercato": item["market"],
                "Probabilità": pct(item["probability"]),
                "Fascia": confidence,
                "1": round(item["odds_h"], 2),
                "X": round(item["odds_d"], 2),
                "2": round(item["odds_a"], 2),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Nessuna selezione sopra soglia.")

with tab_all:
    if rankings:
        rows = []
        for item in rankings:
            rows.append({
                "Partita": f"{item['home']} - {item['away']}",
                "Campionato": item["league"],
                "Data": item["kickoff"],
                "Mercato V8.1": item["market"],
                "Probabilità": pct(item["probability"]),
                "1": round(item["odds_h"], 2),
                "X": round(item["odds_d"], 2),
                "2": round(item["odds_a"], 2),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Nessuna partita analizzata nell'intervallo selezionato.")

with tab_diag:
    if scanner["odds_errors"]:
        st.warning("Sono rimasti errori su alcune richieste quote.")
        st.dataframe(scanner["odds_errors"], use_container_width=True, hide_index=True)

    if scanner["skipped"]:
        st.markdown("**Partite non analizzate**")
        st.dataframe(scanner["skipped"], use_container_width=True, hide_index=True)
    elif not scanner["odds_errors"]:
        st.success("Nessun problema rilevato nello scanner.")

with tab_manual:
    manual_league_name = st.selectbox(
        "Campionato",
        list(BIG5.keys()),
        key="manual_league",
    )
    manual_league = BIG5[manual_league_name]["model_id"]
    teams = get_model_teams(manual_league)

    left, right = st.columns(2)
    with left:
        home = st.selectbox("Squadra casa", teams, key="manual_home")
    away_options = [team for team in teams if team != home]
    with right:
        away = st.selectbox("Squadra ospite", away_options, key="manual_away")

    q1, qx, q2 = st.columns(3)
    with q1:
        odds_h = st.number_input(
            "Quota 1", min_value=1.01, value=2.00, step=0.05, key="manual_h"
        )
    with qx:
        odds_d = st.number_input(
            "Quota X", min_value=1.01, value=3.30, step=0.05, key="manual_d"
        )
    with q2:
        odds_a = st.number_input(
            "Quota 2", min_value=1.01, value=3.50, step=0.05, key="manual_a"
        )

    if st.button("Analizza manualmente", use_container_width=True):
        try:
            result = predict_match(
                bundle=bundle,
                league=manual_league,
                home=home,
                away=away,
                oddsH=odds_h,
                oddsD=odds_d,
                oddsA=odds_a,
                referee=None,
                official=False,
            )
            markets = result["markets"]
            st.success(f"{home} vs {away}")

            a, b = st.columns(2)
            with a:
                st.metric("Over 2.5 cartellini", pct(markets["match_O2.5"]))
                st.metric(f"{home} Over 1.5", pct(markets["home_team_O1.5"]))
                st.metric(f"{away} Over 1.5", pct(markets["away_team_O1.5"]))
            with b:
                st.metric("Under 3.5 cartellini", pct(markets["match_U3.5"]))
                st.metric(f"{home} Under 1.5", pct(markets["home_team_U1.5"]))
                st.metric(f"{away} Under 1.5", pct(markets["away_team_U1.5"]))
        except Exception as exc:
            st.error("Errore durante la previsione.")
            st.code(safe_message(exc))

st.divider()
st.caption(
    "V8.1 Cards Model · OddsPapi / Pinnacle · UI V8.1.1 · "
    "Previsioni probabilistiche, non garanzie di esito."
)
