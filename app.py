import os
import re
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
)

ODDSPAPI_KEY = os.getenv("ODDSPAPI_KEY")
ODDSPAPI_BASE = "https://api.oddspapi.io/v4"

SOCCER_ID = 10
TZ_ITALY = ZoneInfo("Europe/Rome")

# Usiamo due bookmaker per avere un input 1X2 robusto
BOOKMAKERS = "pinnacle,sbobet"

BIG5 = {
    "Premier League": {
        "tournament_name": "Premier League",
        "country": "England",
        "model_id": "E0",
    },
    "La Liga": {
        "tournament_name": "LaLiga",
        "country": "Spain",
        "model_id": "SP1",
    },
    "Bundesliga": {
        "tournament_name": "Bundesliga",
        "country": "Germany",
        "model_id": "D1",
    },
    "Serie A": {
        "tournament_name": "Serie A",
        "country": "Italy",
        "model_id": "I1",
    },
    "Ligue 1": {
        "tournament_name": "Ligue 1",
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
# NAME MATCHING
# ============================================================

def normalize_name(name):
    if not name:
        return ""

    value = unicodedata.normalize("NFKD", str(name))
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)

    removable = {
        "fc", "afc", "cf", "ac", "as",
        "ssc", "club", "football"
    }

    words = [
        w for w in value.split()
        if w not in removable
    ]

    return " ".join(words)


RAW_ALIASES = {
    # Premier League
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",

    # La Liga
    "Real Sociedad": "Sociedad",
    "Athletic Club": "Ath Bilbao",
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
    "Borussia Dortmund": "Dortmund",
    "1899 Hoffenheim": "Hoffenheim",
    "TSG Hoffenheim": "Hoffenheim",
    "1. FC Heidenheim": "Heidenheim",
    "FC St. Pauli": "St Pauli",

    # Serie A
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
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
    "OGC Nice": "Nice",
}

TEAM_ALIASES = {
    normalize_name(k): v
    for k, v in RAW_ALIASES.items()
}


def get_model_teams(model_league):
    return sorted({
        team
        for league, team
        in bundle["state"]["team_hist"].keys()
        if league == model_league
    })


def match_model_team(api_name, model_league):
    candidates = get_model_teams(model_league)

    if not candidates:
        return None

    normalized_candidates = {
        normalize_name(team): team
        for team in candidates
    }

    query = normalize_name(api_name)

    # 1. Exact match
    if query in normalized_candidates:
        return normalized_candidates[query]

    # 2. Alias
    alias = TEAM_ALIASES.get(query)

    if alias:
        alias_normalized = normalize_name(alias)

        if alias_normalized in normalized_candidates:
            return normalized_candidates[alias_normalized]

    # 3. Fuzzy fallback
    best_team = None
    best_score = 0.0

    for team in candidates:
        score = SequenceMatcher(
            None,
            query,
            normalize_name(team),
        ).ratio()

        if score > best_score:
            best_score = score
            best_team = team

    if best_score >= 0.80:
        return best_team

    return None


# ============================================================
# ODDSPAPI
# ============================================================

def oddspapi_get(endpoint, params=None):
    if not ODDSPAPI_KEY:
        raise RuntimeError(
            "ODDSPAPI_KEY non trovata nelle Environment Variables di Render."
        )

    final_params = dict(params or {})
    final_params["apiKey"] = ODDSPAPI_KEY

    response = requests.get(
        f"{ODDSPAPI_BASE}/{endpoint}",
        params=final_params,
        timeout=35,
    )

    if response.status_code == 429:
        raise RuntimeError(
            "OddsPapi: limite temporaneo di richieste raggiunto. "
            "Riprova tra poco."
        )

    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(
            f"OddsPapi: {data.get('error')}"
        )

    return data


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tournaments():
    return oddspapi_get(
        "tournaments",
        {
            "sportId": SOCCER_ID,
            "language": "en",
        },
    )


def resolve_big5_tournaments():
    tournaments = fetch_tournaments()

    resolved = {}
    missing = []

    for display_name, cfg in BIG5.items():
        found = None

        for tournament in tournaments:
            t_name = str(
                tournament.get("tournamentName", "")
            ).strip()

            country = str(
                tournament.get("categoryName", "")
            ).strip()

            if (
                t_name.lower()
                == cfg["tournament_name"].lower()
                and country.lower()
                == cfg["country"].lower()
            ):
                found = tournament
                break

        if found:
            resolved[int(found["tournamentId"])] = {
                **cfg,
                "display_name": display_name,
                "tournament_id": int(
                    found["tournamentId"]
                ),
            }
        else:
            missing.append(display_name)

    return resolved, missing


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_upcoming_fixtures():
    """
    Recuperiamo sempre i prossimi 7 giorni.
    Il selettore 3/5/7 giorni filtra poi localmente,
    senza generare altre richieste API.
    """

    now_utc = datetime.now(timezone.utc)

    end_utc = (
        now_utc + timedelta(days=7)
    )

    from_iso = (
        now_utc
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    to_iso = (
        end_utc
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    return oddspapi_get(
        "fixtures",
        {
            "sportId": SOCCER_ID,
            "from": from_iso,
            "to": to_iso,
            "statusId": 0,
            "hasOdds": "true",
            "language": "en",
        },
    )


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_big5_odds(tournament_ids):
    ids = ",".join(
        str(x)
        for x in tournament_ids
    )

    data = oddspapi_get(
        "odds-by-tournaments",
        {
            "tournamentIds": ids,
            "bookmakers": BOOKMAKERS,
            "language": "en",
            "verbosity": 3,
            "oddsFormat": "decimal",
        },
    )

    # La risposta normalmente è una lista.
    # Gestiamo anche eventuali wrapper.
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("response", "fixtures", "data"):
            if isinstance(data.get(key), list):
                return data[key]

        # Eventuale singolo fixture
        if data.get("fixtureId"):
            return [data]

    return []


# ============================================================
# 1X2 EXTRACTION
# ============================================================

def get_price_from_outcome(outcome):
    players = outcome.get("players") or {}

    prices = []

    for player in players.values():
        if player.get("active") is False:
            continue

        try:
            price = float(player.get("price"))
        except (TypeError, ValueError):
            continue

        if price > 1.0:
            prices.append(price)

    if not prices:
        return None

    return median(prices)


def extract_1x2(odds_item):
    """
    Market:
    101 = Full Time Result

    Outcomes:
    101 = Home
    102 = Draw
    103 = Away
    """

    home_prices = []
    draw_prices = []
    away_prices = []

    books_used = []

    bookmaker_odds = (
        odds_item.get("bookmakerOdds")
        or {}
    )

    for bookmaker, bookmaker_data in bookmaker_odds.items():
        if bookmaker_data.get("suspended") is True:
            continue

        markets = (
            bookmaker_data.get("markets")
            or {}
        )

        market = (
            markets.get("101")
            or markets.get(101)
        )

        if not market:
            continue

        if market.get("marketActive") is False:
            continue

        outcomes = (
            market.get("outcomes")
            or {}
        )

        home_outcome = (
            outcomes.get("101")
            or outcomes.get(101)
            or {}
        )

        draw_outcome = (
            outcomes.get("102")
            or outcomes.get(102)
            or {}
        )

        away_outcome = (
            outcomes.get("103")
            or outcomes.get(103)
            or {}
        )

        h = get_price_from_outcome(home_outcome)
        d = get_price_from_outcome(draw_outcome)
        a = get_price_from_outcome(away_outcome)

        if (
            h is not None
            and d is not None
            and a is not None
        ):
            home_prices.append(h)
            draw_prices.append(d)
            away_prices.append(a)
            books_used.append(bookmaker)

    if not home_prices:
        return None

    return {
        "home": float(median(home_prices)),
        "draw": float(median(draw_prices)),
        "away": float(median(away_prices)),
        "books": books_used,
    }


# ============================================================
# MODEL RANKING
# ============================================================

def get_best_market(prediction, home_name, away_name):
    markets = prediction["markets"]

    choices = [
        (
            "Over 2.5 cartellini",
            markets["match_O2.5"],
        ),
        (
            "Under 3.5 cartellini",
            markets["match_U3.5"],
        ),
        (
            f"{home_name} Over 1.5 cartellini",
            markets["home_team_O1.5"],
        ),
        (
            f"{home_name} Under 1.5 cartellini",
            markets["home_team_U1.5"],
        ),
        (
            f"{away_name} Over 1.5 cartellini",
            markets["away_team_O1.5"],
        ),
        (
            f"{away_name} Under 1.5 cartellini",
            markets["away_team_U1.5"],
        ),
    ]

    market, probability = max(
        choices,
        key=lambda x: x[1],
    )

    return market, float(probability)


def parse_datetime(value):
    if not value:
        return None

    try:
        dt = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(TZ_ITALY)

    except Exception:
        return None


def kickoff_label(dt):
    if not dt:
        return "Orario N/D"

    return dt.strftime(
        "%d/%m/%Y • %H:%M"
    )


@st.cache_data(ttl=21600, show_spinner=False)
def build_auto_scanner():
    tournament_map, missing_tournaments = (
        resolve_big5_tournaments()
    )

    if not tournament_map:
        raise RuntimeError(
            "Non sono riuscito a identificare i campionati Big Five su OddsPapi."
        )

    fixtures = fetch_upcoming_fixtures()

    target_ids = set(
        tournament_map.keys()
    )

    big5_fixtures = []

    for fixture in fixtures:
        try:
            tournament_id = int(
                fixture.get("tournamentId")
            )
        except (TypeError, ValueError):
            continue

        if tournament_id not in target_ids:
            continue

        big5_fixtures.append(fixture)

    odds_items = fetch_big5_odds(
        tuple(sorted(target_ids))
    )

    odds_by_fixture = {
        str(item.get("fixtureId")): item
        for item in odds_items
        if item.get("fixtureId")
    }

    rankings = []
    skipped = []

    for fixture in big5_fixtures:
        fixture_id = str(
            fixture.get("fixtureId")
        )

        tournament_id = int(
            fixture.get("tournamentId")
        )

        cfg = tournament_map[
            tournament_id
        ]

        api_home = (
            fixture.get("participant1Name")
            or ""
        )

        api_away = (
            fixture.get("participant2Name")
            or ""
        )

        if not api_home or not api_away:
            skipped.append({
                "Partita": fixture_id,
                "Motivo": "Nomi squadre mancanti",
            })
            continue

        model_home = match_model_team(
            api_home,
            cfg["model_id"],
        )

        model_away = match_model_team(
            api_away,
            cfg["model_id"],
        )

        if not model_home or not model_away:
            skipped.append({
                "Partita": f"{api_home} - {api_away}",
                "Motivo": "Squadra non riconosciuta dal bundle V8.1",
            })
            continue

        odds_item = odds_by_fixture.get(
            fixture_id
        )

        if not odds_item:
            skipped.append({
                "Partita": f"{api_home} - {api_away}",
                "Motivo": "Quote OddsPapi non trovate",
            })
            continue

        odds = extract_1x2(
            odds_item
        )

        if not odds:
            skipped.append({
                "Partita": f"{api_home} - {api_away}",
                "Motivo": "1X2 Pinnacle/SBOBET non disponibile",
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

            market, probability = (
                get_best_market(
                    result,
                    api_home,
                    api_away,
                )
            )

            kickoff_dt = parse_datetime(
                fixture.get("startTime")
            )

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
                "books": ", ".join(odds["books"]),
            })

        except Exception as exc:
            skipped.append({
                "Partita": f"{api_home} - {api_away}",
                "Motivo": f"Errore V8.1: {exc}",
            })

    rankings.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return {
        "rankings": rankings,
        "skipped": skipped,
        "fixtures_found": len(big5_fixtures),
        "missing_tournaments": missing_tournaments,
        "generated_at": datetime.now(
            TZ_ITALY
        ).strftime("%d/%m/%Y %H:%M"),
    }


# ============================================================
# UI
# ============================================================

st.title("🟨 V8.1 Cards Model")
st.caption(
    "Automatic Yellow Cards Prediction Engine"
)

st.divider()

st.header("⚡ Auto Scanner Big Five")

if not ODDSPAPI_KEY:
    st.error(
        "ODDSPAPI_KEY non configurata su Render."
    )
    st.stop()


col_a, col_b = st.columns(2)

with col_a:
    days_filter = st.selectbox(
        "Intervallo partite",
        [3, 5, 7],
        index=1,
        format_func=lambda x: f"Prossimi {x} giorni",
    )

with col_b:
    minimum_probability = st.slider(
        "Probabilità minima selezione",
        min_value=0.55,
        max_value=0.90,
        value=0.65,
        step=0.01,
        format="%.2f",
    )


with st.spinner(
    "Recupero partite e quote OddsPapi e analizzo con V8.1..."
):
    try:
        scanner = build_auto_scanner()
    except Exception as exc:
        st.error(
            "Errore durante l'Auto Scanner."
        )
        st.exception(exc)
        st.stop()


now_italy = datetime.now(TZ_ITALY)

cutoff = now_italy + timedelta(
    days=days_filter
)

rankings = [
    r for r in scanner["rankings"]
    if (
        r["kickoff_dt"] is not None
        and now_italy
        <= r["kickoff_dt"]
        <= cutoff
    )
]

eligible = [
    r for r in rankings
    if r["probability"]
    >= minimum_probability
]


m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Partite Big Five",
    scanner["fixtures_found"],
)

m2.metric(
    f"Nei prossimi {days_filter} gg",
    len(rankings),
)

m3.metric(
    "Sopra soglia",
    len(eligible),
)

m4.metric(
    "Ultimo aggiornamento",
    scanner["generated_at"],
)


if scanner["missing_tournaments"]:
    st.warning(
        "Campionati OddsPapi non identificati: "
        + ", ".join(
            scanner["missing_tournaments"]
        )
    )


st.divider()


# ============================================================
# TRIPLA
# ============================================================

st.subheader("🏆 Tripla V8.1")

if len(eligible) >= 3:
    top3 = eligible[:3]

    for index, pick in enumerate(
        top3,
        start=1,
    ):
        with st.container(border=True):

            st.markdown(
                f"### {index}. "
                f"{pick['home']} – {pick['away']}"
            )

            st.caption(
                f"{pick['league']} • {pick['kickoff']}"
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Selezione V8.1",
                pick["market"],
            )

            c2.metric(
                "Probabilità",
                pct(pick["probability"]),
            )

            st.caption(
                f"Input 1X2: "
                f"1 {pick['odds_h']:.2f} • "
                f"X {pick['odds_d']:.2f} • "
                f"2 {pick['odds_a']:.2f}"
            )

            st.caption(
                f"Fonte quote: {pick['books'] or 'OddsPapi'}"
            )

else:
    st.warning(
        "Non ci sono almeno 3 selezioni sopra la soglia impostata."
    )


# ============================================================
# QUADRUPLA
# ============================================================

st.subheader("🔥 Quadrupla V8.1")

if len(eligible) >= 4:
    top4 = eligible[:4]

    for index, pick in enumerate(
        top4,
        start=1,
    ):
        with st.container(border=True):

            st.markdown(
                f"**{index}. "
                f"{pick['home']} – {pick['away']}**"
            )

            st.write(
                f"🎯 **{pick['market']}**"
            )

            st.write(
                f"Probabilità V8.1: "
                f"**{pct(pick['probability'])}**"
            )

            st.caption(
                f"{pick['league']} • {pick['kickoff']}"
            )

else:
    st.warning(
        "Non ci sono almeno 4 selezioni sopra la soglia impostata."
    )


# ============================================================
# ALL MATCHES
# ============================================================

with st.expander(
    "📊 Tutte le partite analizzate"
):
    if rankings:
        table = []

        for r in rankings:
            table.append({
                "Partita":
                    f"{r['home']} - {r['away']}",
                "Campionato":
                    r["league"],
                "Data":
                    r["kickoff"],
                "Mercato migliore":
                    r["market"],
                "Probabilità":
                    pct(r["probability"]),
                "1":
                    round(r["odds_h"], 2),
                "X":
                    round(r["odds_d"], 2),
                "2":
                    round(r["odds_a"], 2),
            })

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Nessuna partita analizzabile nell'intervallo selezionato."
        )


with st.expander(
    "⚠️ Partite saltate / diagnostica"
):
    if scanner["skipped"]:
        st.dataframe(
            scanner["skipped"],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success(
            "Nessuna partita saltata."
        )


# ============================================================
# MANUAL MODE
# ============================================================

st.divider()

with st.expander(
    "🧪 Analisi manuale"
):
    league_name = st.selectbox(
        "Campionato",
        list(BIG5.keys()),
        key="manual_league",
    )

    league_code = BIG5[
        league_name
    ]["model_id"]

    teams = get_model_teams(
        league_code
    )

    c1, c2 = st.columns(2)

    with c1:
        home = st.selectbox(
            "Squadra casa",
            teams,
            key="manual_home",
        )

    away_options = [
        team for team in teams
        if team != home
    ]

    with c2:
        away = st.selectbox(
            "Squadra ospite",
            away_options,
            key="manual_away",
        )

    o1, ox, o2 = st.columns(3)

    with o1:
        odds_h = st.number_input(
            "Quota 1",
            min_value=1.01,
            value=2.00,
            step=0.05,
        )

    with ox:
        odds_d = st.number_input(
            "Quota X",
            min_value=1.01,
            value=3.30,
            step=0.05,
        )

    with o2:
        odds_a = st.number_input(
            "Quota 2",
            min_value=1.01,
            value=3.50,
            step=0.05,
        )

    if st.button(
        "Analizza manualmente",
        use_container_width=True,
    ):
        result = predict_match(
            bundle=bundle,
            league=league_code,
            home=home,
            away=away,
            oddsH=odds_h,
            oddsD=odds_d,
            oddsA=odds_a,
            referee=None,
            official=False,
        )

        markets = result["markets"]

        st.success(
            f"{home} vs {away}"
        )

        a, b = st.columns(2)

        with a:
            st.metric(
                "Over 2.5 cartellini",
                pct(markets["match_O2.5"]),
            )

            st.metric(
                f"{home} Over 1.5",
                pct(markets["home_team_O1.5"]),
            )

            st.metric(
                f"{away} Over 1.5",
                pct(markets["away_team_O1.5"]),
            )

        with b:
            st.metric(
                "Under 3.5 cartellini",
                pct(markets["match_U3.5"]),
            )

            st.metric(
                f"{home} Under 1.5",
                pct(markets["home_team_U1.5"]),
            )

            st.metric(
                f"{away} Under 1.5",
                pct(markets["away_team_U1.5"]),
            )


st.divider()

st.caption(
    "V8.1 Cards Model • "
    "Fixtures & 1X2: OddsPapi • "
    "Yellow cards only"
)
