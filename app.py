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
# CONFIGURAZIONE
# ============================================================

st.set_page_config(
    page_title="V8.1 Cards Model",
    page_icon="🟨",
    layout="wide",
)

ODDSPAPI_KEY = os.getenv("ODDSPAPI_KEY")
ODDSPAPI_BASE = "https://api.oddspapi.io/v4"

SOCCER_ID = 10
BOOKMAKER = "pinnacle"

TZ_ITALY = ZoneInfo("Europe/Rome")


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
# CARICAMENTO MODELLO
# ============================================================

@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()


def pct(value):
    return f"{float(value) * 100:.1f}%"


# ============================================================
# NOMI SQUADRE
# ============================================================

def normalize_name(name):
    if not name:
        return ""

    value = unicodedata.normalize(
        "NFKD",
        str(name)
    )

    value = "".join(
        c
        for c in value
        if not unicodedata.combining(c)
    )

    value = value.lower()
    value = value.replace("&", " and ")

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    removable = {
        "fc",
        "afc",
        "cf",
        "ac",
        "as",
        "ssc",
        "club",
        "football",
    }

    words = [
        word
        for word in value.split()
        if word not in removable
    ]

    return " ".join(words)


RAW_ALIASES = {

    # PREMIER LEAGUE
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

    # LA LIGA
    "Real Sociedad": "Sociedad",
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

    # BUNDESLIGA
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

    # SERIE A
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "FC Internazionale Milano": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",

    # LIGUE 1
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Paris SG": "Paris SG",
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
    normalize_name(key): value
    for key, value in RAW_ALIASES.items()
}


def get_model_teams(model_league):

    teams = {
        team
        for league, team
        in bundle["state"]["team_hist"].keys()
        if league == model_league
    }

    return sorted(teams)


def match_model_team(api_name, model_league):

    candidates = get_model_teams(
        model_league
    )

    if not candidates:
        return None

    normalized_candidates = {
        normalize_name(team): team
        for team in candidates
    }

    query = normalize_name(api_name)

    # MATCH ESATTO
    if query in normalized_candidates:
        return normalized_candidates[query]

    # ALIAS
    alias = TEAM_ALIASES.get(query)

    if alias:

        alias_normalized = normalize_name(
            alias
        )

        if alias_normalized in normalized_candidates:
            return normalized_candidates[
                alias_normalized
            ]

    # FUZZY MATCH
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
# IDENTIFICAZIONE BIG FIVE
# ============================================================

def identify_big5_fixture(fixture):

    tournament_name = str(
        fixture.get(
            "tournamentName",
            ""
        )
    ).strip()

    category_name = str(
        fixture.get(
            "categoryName",
            ""
        )
    ).strip()

    for display_name, cfg in BIG5.items():

        valid_name = any(
            tournament_name.lower()
            == name.lower()
            for name in cfg["names"]
        )

        valid_country = (
            category_name.lower()
            == cfg["country"].lower()
        )

        if valid_name and valid_country:

            return {
                **cfg,
                "display_name": display_name,
            }

    return None


# ============================================================
# ODDSPAPI
# ============================================================

def safe_message(message):

    message = str(message)

    if ODDSPAPI_KEY:

        message = message.replace(
            ODDSPAPI_KEY,
            "***API_KEY_NASCOSTA***"
        )

    return message


def oddspapi_get(endpoint, params=None):

    if not ODDSPAPI_KEY:

        raise RuntimeError(
            "ODDSPAPI_KEY non configurata su Render."
        )

    final_params = dict(
        params or {}
    )

    final_params["apiKey"] = (
        ODDSPAPI_KEY
    )

    try:

        response = requests.get(
            f"{ODDSPAPI_BASE}/{endpoint}",
            params=final_params,
            timeout=40,
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            "Errore di connessione con OddsPapi: "
            + safe_message(exc)
        )

    # RATE LIMIT
    if response.status_code == 429:

        try:
            body = response.json()
        except Exception:
            body = {}

        retry_ms = 0

        if isinstance(body, dict):

            error = body.get(
                "error",
                {}
            )

            if isinstance(error, dict):

                retry_ms = error.get(
                    "retryMs",
                    0
                )

        raise RuntimeError(
            "OddsPapi rate limit raggiunto. "
            f"Retry indicato: {retry_ms} ms."
        )

    # ALTRI ERRORI HTTP
    if not response.ok:

        body = response.text

        if len(body) > 800:
            body = body[:800] + "..."

        raise RuntimeError(
            f"OddsPapi HTTP {response.status_code}: "
            + safe_message(body)
        )

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            "OddsPapi ha restituito una risposta "
            "che non è JSON valido."
        )

    if isinstance(data, dict):

        error = data.get("error")

        if error:

            raise RuntimeError(
                "OddsPapi: "
                + safe_message(error)
            )

    return data


# ============================================================
# FIXTURES
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def fetch_upcoming_fixtures():

    now_utc = datetime.now(
        timezone.utc
    )

    end_utc = (
        now_utc
        + timedelta(days=7)
    )

    from_iso = (
        now_utc
        .replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        )
    )

    to_iso = (
        end_utc
        .replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        )
    )

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

        for key in (
            "fixtures",
            "response",
            "data",
            "items",
        ):

            if isinstance(
                data.get(key),
                list
            ):

                return data[key]

    return []


# ============================================================
# ODDS
# ============================================================

def normalize_odds_response(data):

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in (
            "fixtures",
            "response",
            "data",
            "items",
            "events",
        ):

            if isinstance(
                data.get(key),
                list
            ):

                return data[key]

        if data.get("fixtureId"):
            return [data]

    return []


@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def fetch_tournament_odds(
    tournament_id
):
    """
    Una richiesta per singolo campionato.

    Questa forma replica l'esempio ufficiale
    OddsPapi:

    /odds-by-tournaments
    ?tournamentIds=17
    &bookmakers=pinnacle
    &language=en
    &verbosity=3
    """

    data = oddspapi_get(
        "odds-by-tournaments",
        {
            "tournamentIds":
                str(tournament_id),

            "bookmakers":
                BOOKMAKER,

            "language":
                "en",

            "verbosity":
                3,
        },
    )

    return normalize_odds_response(
        data
    )


def get_active_price(outcome):

    players = outcome.get(
        "players",
        {}
    )

    if not isinstance(
        players,
        dict
    ):

        return None

    main_line_prices = []
    other_prices = []

    for player in players.values():

        if not isinstance(
            player,
            dict
        ):
            continue

        if (
            player.get("active")
            is False
        ):
            continue

        try:

            price = float(
                player.get("price")
            )

        except (
            TypeError,
            ValueError
        ):
            continue

        if price <= 1.0:
            continue

        if player.get(
            "mainLine"
        ) is True:

            main_line_prices.append(
                price
            )

        else:

            other_prices.append(
                price
            )

    if main_line_prices:

        return float(
            median(
                main_line_prices
            )
        )

    if other_prices:

        return float(
            median(
                other_prices
            )
        )

    return None


def extract_1x2(odds_item):
    """
    OddsPapi:

    Market 101 = Full Time Result

    Outcome:
    101 = 1
    102 = X
    103 = 2
    """

    bookmaker_odds = (
        odds_item.get(
            "bookmakerOdds"
        )
        or {}
    )

    bookmaker_data = (
        bookmaker_odds.get(
            BOOKMAKER
        )
        or {}
    )

    if not bookmaker_data:
        return None

    if bookmaker_data.get(
        "suspended"
    ) is True:

        return None

    markets = (
        bookmaker_data.get(
            "markets"
        )
        or {}
    )

    market = (
        markets.get("101")
        or markets.get(101)
    )

    if not market:
        return None

    if market.get(
        "marketActive"
    ) is False:

        return None

    outcomes = (
        market.get(
            "outcomes"
        )
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

    home_price = get_active_price(
        home_outcome
    )

    draw_price = get_active_price(
        draw_outcome
    )

    away_price = get_active_price(
        away_outcome
    )

    if (
        home_price is None
        or draw_price is None
        or away_price is None
    ):

        return None

    return {
        "home": home_price,
        "draw": draw_price,
        "away": away_price,
    }


# ============================================================
# V8.1
# ============================================================

def get_best_market(
    prediction,
    home_name,
    away_name
):

    markets = prediction[
        "markets"
    ]

    possibilities = [

        (
            "Over 2.5 cartellini",
            markets[
                "match_O2.5"
            ],
        ),

        (
            "Under 3.5 cartellini",
            markets[
                "match_U3.5"
            ],
        ),

        (
            f"{home_name} Over 1.5 cartellini",
            markets[
                "home_team_O1.5"
            ],
        ),

        (
            f"{home_name} Under 1.5 cartellini",
            markets[
                "home_team_U1.5"
            ],
        ),

        (
            f"{away_name} Over 1.5 cartellini",
            markets[
                "away_team_O1.5"
            ],
        ),

        (
            f"{away_name} Under 1.5 cartellini",
            markets[
                "away_team_U1.5"
            ],
        ),
    ]

    market, probability = max(
        possibilities,
        key=lambda item: item[1]
    )

    return (
        market,
        float(probability)
    )


# ============================================================
# DATE
# ============================================================

def parse_datetime(value):

    if not value:
        return None

    try:

        dt = datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00"
            )
        )

        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            TZ_ITALY
        )

    except Exception:

        return None


def kickoff_label(dt):

    if dt is None:
        return "Data N/D"

    return dt.strftime(
        "%d/%m/%Y • %H:%M"
    )


# ============================================================
# AUTO SCANNER
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def build_auto_scanner():

    fixtures = (
        fetch_upcoming_fixtures()
    )

    big5_fixtures = []

    for fixture in fixtures:

        cfg = identify_big5_fixture(
            fixture
        )

        if cfg is None:
            continue

        fixture_copy = dict(
            fixture
        )

        fixture_copy[
            "_model_cfg"
        ] = cfg

        big5_fixtures.append(
            fixture_copy
        )

    # --------------------------------------------
    # Tournament ID effettivamente presenti
    # --------------------------------------------

    tournament_ids = sorted({
        int(
            fixture[
                "tournamentId"
            ]
        )
        for fixture
        in big5_fixtures
        if fixture.get(
            "tournamentId"
        ) is not None
    })

    # --------------------------------------------
    # Quote: una chiamata per campionato
    # --------------------------------------------

    odds_by_fixture = {}

    odds_errors = []

    for tournament_id in tournament_ids:

        try:

            odds_items = (
                fetch_tournament_odds(
                    tournament_id
                )
            )

            for item in odds_items:

                fixture_id = str(
                    item.get(
                        "fixtureId",
                        ""
                    )
                )

                if fixture_id:

                    odds_by_fixture[
                        fixture_id
                    ] = item

        except Exception as exc:

            odds_errors.append({
                "Tournament ID":
                    tournament_id,

                "Errore":
                    safe_message(exc),
            })

    # --------------------------------------------
    # Previsioni
    # --------------------------------------------

    rankings = []
    skipped = []

    for fixture in big5_fixtures:

        cfg = fixture[
            "_model_cfg"
        ]

        fixture_id = str(
            fixture.get(
                "fixtureId",
                ""
            )
        )

        api_home = (
            fixture.get(
                "participant1Name"
            )
            or fixture.get(
                "participant1ShortName"
            )
            or ""
        )

        api_away = (
            fixture.get(
                "participant2Name"
            )
            or fixture.get(
                "participant2ShortName"
            )
            or ""
        )

        if (
            not api_home
            or not api_away
        ):

            skipped.append({
                "Partita":
                    fixture_id,

                "Motivo":
                    "Nome squadre mancante",
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

        if not model_home:

            skipped.append({
                "Partita":
                    f"{api_home} - {api_away}",

                "Motivo":
                    f"Casa non riconosciuta: {api_home}",
            })

            continue

        if not model_away:

            skipped.append({
                "Partita":
                    f"{api_home} - {api_away}",

                "Motivo":
                    f"Ospite non riconosciuta: {api_away}",
            })

            continue

        odds_item = (
            odds_by_fixture.get(
                fixture_id
            )
        )

        if not odds_item:

            skipped.append({
                "Partita":
                    f"{api_home} - {api_away}",

                "Motivo":
                    "Quote Pinnacle non trovate",
            })

            continue

        odds = extract_1x2(
            odds_item
        )

        if odds is None:

            skipped.append({
                "Partita":
                    f"{api_home} - {api_away}",

                "Motivo":
                    "Mercato 1X2 Pinnacle non disponibile",
            })

            continue

        try:

            result = predict_match(

                bundle=bundle,

                league=cfg[
                    "model_id"
                ],

                home=model_home,

                away=model_away,

                oddsH=odds[
                    "home"
                ],

                oddsD=odds[
                    "draw"
                ],

                oddsA=odds[
                    "away"
                ],

                referee=None,

                official=False,
            )

        except Exception as exc:

            skipped.append({
                "Partita":
                    f"{api_home} - {api_away}",

                "Motivo":
                    "Errore modello: "
                    + safe_message(exc),
            })

            continue

        market, probability = (
            get_best_market(
                result,
                api_home,
                api_away,
            )
        )

        kickoff_dt = parse_datetime(
            fixture.get(
                "startTime"
            )
        )

        rankings.append({

            "fixture_id":
                fixture_id,

            "league":
                cfg[
                    "display_name"
                ],

            "home":
                api_home,

            "away":
                api_away,

            "kickoff_dt":
                kickoff_dt,

            "kickoff":
                kickoff_label(
                    kickoff_dt
                ),

            "market":
                market,

            "probability":
                probability,

            "score":
                probability,

            "odds_h":
                odds["home"],

            "odds_d":
                odds["draw"],

            "odds_a":
                odds["away"],

            "bookmaker":
                "Pinnacle",
        })

    rankings.sort(
        key=lambda item:
            item["score"],
        reverse=True,
    )

    return {

        "rankings":
            rankings,

        "skipped":
            skipped,

        "odds_errors":
            odds_errors,

        "fixtures_found":
            len(
                big5_fixtures
            ),

        "tournaments":
            tournament_ids,

        "generated_at":
            datetime.now(
                TZ_ITALY
            ).strftime(
                "%d/%m/%Y %H:%M"
            ),
    }


# ============================================================
# UI
# ============================================================

st.title(
    "🟨 V8.1 Cards Model"
)

st.caption(
    "Automatic Yellow Cards Prediction Engine"
)

st.divider()


st.header(
    "⚡ Auto Scanner Big Five"
)


if not ODDSPAPI_KEY:

    st.error(
        "ODDSPAPI_KEY non configurata su Render."
    )

    st.stop()


col1, col2 = st.columns(2)


with col1:

    days_filter = st.selectbox(

        "Intervallo partite",

        [
            3,
            5,
            7,
        ],

        index=1,

        format_func=lambda x:
            f"Prossimi {x} giorni",
    )


with col2:

    minimum_probability = st.slider(

        "Probabilità minima selezione",

        min_value=0.55,

        max_value=0.90,

        value=0.65,

        step=0.01,

        format="%.2f",
    )


with st.spinner(
    "Recupero partite e quote Pinnacle "
    "e analizzo con V8.1..."
):

    try:

        scanner = (
            build_auto_scanner()
        )

    except Exception as exc:

        st.error(
            "Errore durante l'Auto Scanner."
        )

        st.code(
            safe_message(exc)
        )

        st.stop()


now_italy = datetime.now(
    TZ_ITALY
)

cutoff = (
    now_italy
    + timedelta(
        days=days_filter
    )
)


rankings = [

    item

    for item
    in scanner[
        "rankings"
    ]

    if (
        item[
            "kickoff_dt"
        ]
        is not None

        and

        now_italy
        <= item[
            "kickoff_dt"
        ]
        <= cutoff
    )
]


eligible = [

    item

    for item
    in rankings

    if item[
        "probability"
    ]
    >= minimum_probability
]


# ============================================================
# KPI
# ============================================================

m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "Partite Big Five",
    scanner[
        "fixtures_found"
    ],
)


m2.metric(
    f"Prossimi {days_filter} giorni",
    len(rankings),
)


m3.metric(
    "Sopra soglia",
    len(eligible),
)


m4.metric(
    "Ultimo aggiornamento",
    scanner[
        "generated_at"
    ],
)


st.caption(
    "Dati OddsPapi memorizzati in cache "
    "per limitare il consumo delle richieste API."
)


# ============================================================
# ERRORI ODDS
# ============================================================

if scanner[
    "odds_errors"
]:

    with st.expander(
        "⚠️ Diagnostica OddsPapi"
    ):

        st.dataframe(
            scanner[
                "odds_errors"
            ],
            hide_index=True,
            use_container_width=True,
        )


st.divider()


# ============================================================
# TRIPLA
# ============================================================

st.subheader(
    "🏆 Tripla V8.1"
)


if len(eligible) >= 3:

    top3 = eligible[:3]

    for index, pick in enumerate(
        top3,
        start=1,
    ):

        with st.container(
            border=True
        ):

            st.markdown(
                f"### {index}. "
                f"{pick['home']} – "
                f"{pick['away']}"
            )

            st.caption(
                f"{pick['league']} • "
                f"{pick['kickoff']}"
            )

            a, b = st.columns(2)

            a.metric(
                "Selezione V8.1",
                pick[
                    "market"
                ],
            )

            b.metric(
                "Probabilità modello",
                pct(
                    pick[
                        "probability"
                    ]
                ),
            )

            st.caption(
                "Quote 1X2 usate dal modello: "
                f"1 {pick['odds_h']:.2f} • "
                f"X {pick['odds_d']:.2f} • "
                f"2 {pick['odds_a']:.2f}"
            )

            st.caption(
                "Fonte quote: Pinnacle via OddsPapi"
            )


else:

    st.warning(
        "Non ci sono almeno 3 selezioni "
        "sopra la probabilità minima."
    )


# ============================================================
# QUADRUPLA
# ============================================================

st.subheader(
    "🔥 Quadrupla V8.1"
)


if len(eligible) >= 4:

    top4 = eligible[:4]

    for index, pick in enumerate(
        top4,
        start=1,
    ):

        with st.container(
            border=True
        ):

            st.markdown(
                f"### {index}. "
                f"{pick['home']} – "
                f"{pick['away']}"
            )

            st.caption(
                f"{pick['league']} • "
                f"{pick['kickoff']}"
            )

            st.write(
                f"🎯 **{pick['market']}**"
            )

            st.write(
                "Probabilità V8.1: "
                f"**{pct(pick['probability'])}**"
            )


else:

    st.warning(
        "Non ci sono almeno 4 selezioni "
        "sopra la probabilità minima."
    )


# ============================================================
# CLASSIFICA COMPLETA
# ============================================================

with st.expander(
    "📊 Tutte le partite analizzate"
):

    if rankings:

        table = []

        for item in rankings:

            table.append({

                "Partita":
                    f"{item['home']} - "
                    f"{item['away']}",

                "Campionato":
                    item[
                        "league"
                    ],

                "Data":
                    item[
                        "kickoff"
                    ],

                "Mercato V8.1":
                    item[
                        "market"
                    ],

                "Probabilità":
                    pct(
                        item[
                            "probability"
                        ]
                    ),

                "1":
                    round(
                        item[
                            "odds_h"
                        ],
                        2
                    ),

                "X":
                    round(
                        item[
                            "odds_d"
                        ],
                        2
                    ),

                "2":
                    round(
                        item[
                            "odds_a"
                        ],
                        2
                    ),
            })

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
        )

    else:

        st.info(
            "Nessuna partita analizzata "
            "nell'intervallo selezionato."
        )


# ============================================================
# PARTITE SALTATE
# ============================================================

with st.expander(
    "⚠️ Partite saltate / diagnostica"
):

    if scanner[
        "skipped"
    ]:

        st.dataframe(
            scanner[
                "skipped"
            ],
            hide_index=True,
            use_container_width=True,
        )

    else:

        st.success(
            "Nessuna partita saltata."
        )


# ============================================================
# MODALITÀ MANUALE
# ============================================================

st.divider()


with st.expander(
    "🧪 Analisi manuale"
):

    manual_league_name = (
        st.selectbox(
            "Campionato",
            list(
                BIG5.keys()
            ),
            key="manual_league",
        )
    )

    manual_league = (
        BIG5[
            manual_league_name
        ][
            "model_id"
        ]
    )

    teams = get_model_teams(
        manual_league
    )


    left, right = st.columns(2)


    with left:

        home = st.selectbox(
            "Squadra casa",
            teams,
            key="manual_home",
        )


    away_options = [

        team

        for team
        in teams

        if team != home
    ]


    with right:

        away = st.selectbox(
            "Squadra ospite",
            away_options,
            key="manual_away",
        )


    q1, qx, q2 = st.columns(3)


    with q1:

        odds_h = st.number_input(
            "Quota 1",
            min_value=1.01,
            value=2.00,
            step=0.05,
            key="manual_h",
        )


    with qx:

        odds_d = st.number_input(
            "Quota X",
            min_value=1.01,
            value=3.30,
            step=0.05,
            key="manual_d",
        )


    with q2:

        odds_a = st.number_input(
            "Quota 2",
            min_value=1.01,
            value=3.50,
            step=0.05,
            key="manual_a",
        )


    if st.button(
        "Analizza manualmente",
        use_container_width=True,
    ):

        try:

            result = predict_match(

                bundle=bundle,

                league=
                    manual_league,

                home=home,

                away=away,

                oddsH=
                    odds_h,

                oddsD=
                    odds_d,

                oddsA=
                    odds_a,

                referee=None,

                official=False,
            )

        except Exception as exc:

            st.error(
                "Errore durante la previsione."
            )

            st.code(
                safe_message(exc)
            )

        else:

            markets = (
                result[
                    "markets"
                ]
            )

            st.success(
                f"{home} vs {away}"
            )

            a, b = st.columns(2)


            with a:

                st.metric(
                    "Over 2.5 cartellini",
                    pct(
                        markets[
                            "match_O2.5"
                        ]
                    ),
                )

                st.metric(
                    f"{home} Over 1.5",
                    pct(
                        markets[
                            "home_team_O1.5"
                        ]
                    ),
                )

                st.metric(
                    f"{away} Over 1.5",
                    pct(
                        markets[
                            "away_team_O1.5"
                        ]
                    ),
                )


            with b:

                st.metric(
                    "Under 3.5 cartellini",
                    pct(
                        markets[
                            "match_U3.5"
                        ]
                    ),
                )

                st.metric(
                    f"{home} Under 1.5",
                    pct(
                        markets[
                            "home_team_U1.5"
                        ]
                    ),
                )

                st.metric(
                    f"{away} Under 1.5",
                    pct(
                        markets[
                            "away_team_U1.5"
                        ]
                    ),
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "V8.1 Cards Model • "
    "Fixtures & 1X2: OddsPapi / Pinnacle • "
    "Yellow cards prediction engine"
)
