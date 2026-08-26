import os
import re
import unicodedata
from datetime import datetime, timedelta
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

API_BASE = "https://v3.football.api-sports.io"
API_KEY = os.getenv("API_FOOTBALL_KEY")

TZ = ZoneInfo("Europe/Rome")

BIG5 = {
    "Premier League": {
        "api_id": 39,
        "model_id": "E0",
    },
    "Bundesliga": {
        "api_id": 78,
        "model_id": "D1",
    },
    "La Liga": {
        "api_id": 140,
        "model_id": "SP1",
    },
    "Serie A": {
        "api_id": 135,
        "model_id": "I1",
    },
    "Ligue 1": {
        "api_id": 61,
        "model_id": "F1",
    },
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()


# ============================================================
# HELPERS
# ============================================================

def pct(x):
    return f"{x * 100:.1f}%"


def normalize_name(name):
    if not name:
        return ""

    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = name.replace("&", " and ")
    name = re.sub(r"[^a-z0-9]+", " ", name)

    return " ".join(name.split())


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
    "Real Betis": "Betis",
    "RCD Espanyol": "Espanol",
    "Celta Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alaves": "Alaves",
    "Real Mallorca": "Mallorca",

    # Bundesliga
    "Borussia Monchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "1899 Hoffenheim": "Hoffenheim",
    "1. FC Heidenheim": "Heidenheim",
    "FC St. Pauli": "St Pauli",

    # Serie A
    "Inter": "Inter",
    "Internazionale": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",

    # Ligue 1
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Olympique Marseille": "Marseille",
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
        for league, team in bundle["state"]["team_hist"].keys()
        if league == model_league
    })


def match_model_team(api_name, model_league):
    """
    Traduce il nome API-Football nel nome usato dal bundle V8.1.
    """

    candidates = get_model_teams(model_league)

    if not candidates:
        return None

    q = normalize_name(api_name)

    by_normalized = {
        normalize_name(team): team
        for team in candidates
    }

    # Match esatto
    if q in by_normalized:
        return by_normalized[q]

    # Alias noto
    alias = TEAM_ALIASES.get(q)

    if alias:
        alias_norm = normalize_name(alias)

        if alias_norm in by_normalized:
            return by_normalized[alias_norm]

    # Fuzzy fallback
    best_team = None
    best_score = 0

    for candidate in candidates:
        score = SequenceMatcher(
            None,
            q,
            normalize_name(candidate)
        ).ratio()

        if score > best_score:
            best_score = score
            best_team = candidate

    # Manteniamo una soglia piuttosto prudente
    if best_score >= 0.82:
        return best_team

    return None


# ============================================================
# API-FOOTBALL
# ============================================================

def api_get(endpoint, params=None):
    if not API_KEY:
        raise RuntimeError(
            "API_FOOTBALL_KEY non trovata nelle variabili Environment di Render."
        )

    response = requests.get(
        API_BASE + endpoint,
        headers={
            "x-apisports-key": API_KEY
        },
        params=params or {},
        timeout=25,
    )

    response.raise_for_status()

    data = response.json()

    errors = data.get("errors")

    if errors:
        raise RuntimeError(
            f"API-Football error: {errors}"
        )

    return data


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fixtures(
    league_api_id,
    season,
    date_from,
    date_to
):
    data = api_get(
        "/fixtures",
        {
            "league": league_api_id,
            "season": season,
            "from": date_from,
            "to": date_to,
            "timezone": "Europe/Rome",
        },
    )

    return data.get("response", [])


@st.cache_data(ttl=86400, show_spinner=False)
def get_match_winner_bet_id():
    """
    Recupera dinamicamente l'ID del mercato Match Winner.
    """

    data = api_get(
        "/odds/bets",
        {
            "search": "Match Winner"
        }
    )

    for item in data.get("response", []):
        name = str(item.get("name", "")).lower()

        if "match winner" in name:
            return item.get("id")

    return None


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_odds_for_date(
    league_api_id,
    season,
    date_string,
    bet_id
):
    """
    Una richiesta per campionato/data invece di una richiesta
    per ogni singola partita.
    """

    params = {
        "league": league_api_id,
        "season": season,
        "date": date_string,
    }

    if bet_id is not None:
        params["bet"] = bet_id

    first = api_get(
        "/odds",
        params
    )

    results = list(
        first.get("response", [])
    )

    paging = first.get("paging", {})
    total_pages = int(
        paging.get("total", 1) or 1
    )

    # Generalmente Big Five/date = una pagina,
    # ma gestiamo eventuale paginazione.
    for page in range(2, total_pages + 1):

        page_params = dict(params)
        page_params["page"] = page

        data = api_get(
            "/odds",
            page_params
        )

        results.extend(
            data.get("response", [])
        )

    return results


def extract_1x2(odds_item, bet_id):
    """
    Usa la mediana delle quote disponibili tra i bookmaker.
    È più robusto che dipendere da un singolo bookmaker.
    """

    homes = []
    draws = []
    aways = []

    valid_books = 0

    for bookmaker in odds_item.get(
        "bookmakers",
        []
    ):

        for bet in bookmaker.get(
            "bets",
            []
        ):

            current_id = bet.get("id")
            current_name = str(
                bet.get("name", "")
            ).lower()

            correct_bet = False

            if bet_id is not None:
                correct_bet = current_id == bet_id

            if "match winner" in current_name:
                correct_bet = True

            if not correct_bet:
                continue

            h = None
            d = None
            a = None

            for value in bet.get(
                "values",
                []
            ):

                label = str(
                    value.get("value", "")
                ).strip().lower()

                try:
                    odd = float(
                        value.get("odd")
                    )
                except Exception:
                    continue

                if label in (
                    "home",
                    "1"
                ):
                    h = odd

                elif label in (
                    "draw",
                    "x"
                ):
                    d = odd

                elif label in (
                    "away",
                    "2"
                ):
                    a = odd

            if (
                h is not None
                and d is not None
                and a is not None
            ):
                homes.append(h)
                draws.append(d)
                aways.append(a)

                valid_books += 1

    if not homes:
        return None

    return {
        "home": float(median(homes)),
        "draw": float(median(draws)),
        "away": float(median(aways)),
        "bookmakers": valid_books,
    }


# ============================================================
# V8.1 SCANNER
# ============================================================

def best_market(
    prediction,
    display_home,
    display_away
):
    markets = prediction["markets"]

    possibilities = [
        (
            "Over 2.5 cartellini",
            markets["match_O2.5"]
        ),
        (
            "Under 3.5 cartellini",
            markets["match_U3.5"]
        ),
        (
            f"{display_home} Over 1.5 cartellini",
            markets["home_team_O1.5"]
        ),
        (
            f"{display_home} Under 1.5 cartellini",
            markets["home_team_U1.5"]
        ),
        (
            f"{display_away} Over 1.5 cartellini",
            markets["away_team_O1.5"]
        ),
        (
            f"{display_away} Under 1.5 cartellini",
            markets["away_team_U1.5"]
        ),
    ]

    return max(
        possibilities,
        key=lambda x: x[1]
    )


def historical_coverage(
    model_league,
    home,
    away
):
    state = bundle["state"]

    home_hist = state[
        "team_hist"
    ].get(
        (model_league, home),
        {}
    ).get(
        "cards",
        []
    )

    away_hist = state[
        "team_hist"
    ].get(
        (model_league, away),
        {}
    ).get(
        "cards",
        []
    )

    home_n = min(
        20,
        len(home_hist)
    )

    away_n = min(
        20,
        len(away_hist)
    )

    coverage = (
        home_n + away_n
    ) / 40.0

    return (
        coverage,
        home_n,
        away_n
    )


def parse_kickoff(date_string):
    try:
        dt = datetime.fromisoformat(
            date_string.replace(
                "Z",
                "+00:00"
            )
        )

        dt = dt.astimezone(TZ)

        return dt.strftime(
            "%d/%m • %H:%M"
        )

    except Exception:
        return date_string


def run_scanner(days):
    now = datetime.now(TZ)

    start_date = now.date()

    end_date = (
        start_date
        + timedelta(days=days - 1)
    )

    # Big Five: stagione identificata
    # dall'anno di inizio.
    season = (
        now.year
        if now.month >= 7
        else now.year - 1
    )

    fixtures = []
    skipped = []

    # --------------------------------
    # FIXTURES
    # --------------------------------

    for league_name, cfg in BIG5.items():

        items = fetch_fixtures(
            cfg["api_id"],
            season,
            start_date.isoformat(),
            end_date.isoformat(),
        )

        for item in items:

            fixture = item.get(
                "fixture",
                {}
            )

            status = fixture.get(
                "status",
                {}
            ).get(
                "short"
            )

            # Solo match non iniziati
            if status != "NS":
                continue

            teams = item.get(
                "teams",
                {}
            )

            api_home = teams.get(
                "home",
                {}
            ).get(
                "name"
            )

            api_away = teams.get(
                "away",
                {}
            ).get(
                "name"
            )

            model_home = match_model_team(
                api_home,
                cfg["model_id"]
            )

            model_away = match_model_team(
                api_away,
                cfg["model_id"]
            )

            if (
                not model_home
                or not model_away
            ):

                skipped.append({
                    "match": f"{api_home} - {api_away}",
                    "reason": "Nome squadra non riconosciuto",
                })

                continue

            fixtures.append({
                "fixture_id": fixture.get("id"),
                "league_name": league_name,
                "league_api": cfg["api_id"],
                "league_model": cfg["model_id"],
                "date": str(
                    fixture.get(
                        "date",
                        ""
                    )
                )[:10],
                "kickoff_raw": fixture.get(
                    "date",
                    ""
                ),
                "api_home": api_home,
                "api_away": api_away,
                "model_home": model_home,
                "model_away": model_away,
            })

    # --------------------------------
    # ODDS
    # --------------------------------

    bet_id = get_match_winner_bet_id()

    groups = {}

    for fixture in fixtures:

        key = (
            fixture["league_api"],
            fixture["date"]
        )

        groups.setdefault(
            key,
            []
        ).append(
            fixture
        )

    odds_by_fixture = {}

    for (
        league_api,
        date_string
    ), group in groups.items():

        items = fetch_odds_for_date(
            league_api,
            season,
            date_string,
            bet_id,
        )

        for item in items:

            fixture_info = item.get(
                "fixture",
                {}
            )

            fixture_id = fixture_info.get(
                "id"
            )

            odds = extract_1x2(
                item,
                bet_id
            )

            if odds:
                odds_by_fixture[
                    fixture_id
                ] = odds

    # --------------------------------
    # PREDICTIONS
    # --------------------------------

    rankings = []

    for fixture in fixtures:

        fixture_id = fixture[
            "fixture_id"
        ]

        odds = odds_by_fixture.get(
            fixture_id
        )

        if not odds:

            skipped.append({
                "match":
                    f"{fixture['api_home']} - {fixture['api_away']}",
                "reason":
                    "Quote 1X2 non disponibili",
            })

            continue

        try:

            result = predict_match(
                bundle=bundle,
                league=fixture[
                    "league_model"
                ],
                home=fixture[
                    "model_home"
                ],
                away=fixture[
                    "model_away"
                ],
                oddsH=odds["home"],
                oddsD=odds["draw"],
                oddsA=odds["away"],
                referee=None,
                official=False,
            )

            market_name, probability = (
                best_market(
                    result,
                    fixture[
                        "api_home"
                    ],
                    fixture[
                        "api_away"
                    ],
                )
            )

            coverage, home_n, away_n = (
                historical_coverage(
                    fixture[
                        "league_model"
                    ],
                    fixture[
                        "model_home"
                    ],
                    fixture[
                        "model_away"
                    ],
                )
            )

            # Lo score è dominato dalla probabilità,
            # con una penalizzazione piccola se lo
            # storico disponibile è incompleto.
            ranking_score = (
                probability
                * (
                    0.90
                    + 0.10 * coverage
                )
            )

            rankings.append({
                "fixture_id":
                    fixture_id,
                "league":
                    fixture[
                        "league_name"
                    ],
                "home":
                    fixture[
                        "api_home"
                    ],
                "away":
                    fixture[
                        "api_away"
                    ],
                "kickoff":
                    parse_kickoff(
                        fixture[
                            "kickoff_raw"
                        ]
                    ),
                "market":
                    market_name,
                "probability":
                    float(
                        probability
                    ),
                "score":
                    float(
                        ranking_score
                    ),
                "history":
                    f"{home_n}/20 + {away_n}/20",
                "odds_home":
                    odds["home"],
                "odds_draw":
                    odds["draw"],
                "odds_away":
                    odds["away"],
                "bookmakers":
                    odds["bookmakers"],
            })

        except Exception as exc:

            skipped.append({
                "match":
                    f"{fixture['api_home']} - {fixture['api_away']}",
                "reason":
                    f"Errore modello: {exc}",
            })

    rankings.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return (
        rankings,
        skipped,
        len(fixtures),
        season,
    )


# ============================================================
# UI
# ============================================================

st.title("🟨 V8.1 Cards Model")
st.caption(
    "Automatic Yellow Cards Prediction Engine"
)

st.divider()

st.header("⚡ Auto Scanner Big Five")

c1, c2 = st.columns(2)

with c1:

    scan_days = st.selectbox(
        "Intervallo partite",
        [
            3,
            5,
            7
        ],
        index=1,
        format_func=lambda x:
            f"Prossimi {x} giorni"
    )

with c2:

    minimum_probability = st.slider(
        "Probabilità minima selezione",
        min_value=0.55,
        max_value=0.85,
        value=0.65,
        step=0.01,
        format="%.2f",
    )


if not API_KEY:

    st.error(
        "API_FOOTBALL_KEY non configurata su Render."
    )

    st.stop()


with st.spinner(
    "Analizzo le prossime partite dei Big Five..."
):

    try:

        (
            rankings,
            skipped,
            fixture_count,
            season
        ) = run_scanner(
            scan_days
        )

    except Exception as exc:

        st.error(
            "Errore durante il recupero dei dati."
        )

        st.exception(exc)

        st.stop()


eligible = [
    r
    for r in rankings
    if r["probability"]
    >= minimum_probability
]


m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Partite trovate",
    fixture_count
)

m2.metric(
    "Analizzate V8.1",
    len(rankings)
)

m3.metric(
    "Sopra soglia",
    len(eligible)
)

m4.metric(
    "Stagione",
    f"{season}/{str(season + 1)[-2:]}"
)

st.divider()


# ============================================================
# TRIPLA
# ============================================================

st.subheader("🏆 Tripla V8.1")

if len(eligible) >= 3:

    top3 = eligible[:3]

    for i, pick in enumerate(
        top3,
        1
    ):

        with st.container(
            border=True
        ):

            st.markdown(
                f"### {i}. "
                f"{pick['home']} – "
                f"{pick['away']}"
            )

            st.caption(
                f"{pick['league']} • "
                f"{pick['kickoff']}"
            )

            a, b, c = st.columns(3)

            a.metric(
                "Mercato",
                pick[
                    "market"
                ]
            )

            b.metric(
                "Probabilità V8.1",
                pct(
                    pick[
                        "probability"
                    ]
                )
            )

            c.metric(
                "Storico",
                pick[
                    "history"
                ]
            )

            st.caption(
                "Quote 1X2 usate dal modello: "
                f"1 {pick['odds_home']:.2f} • "
                f"X {pick['odds_draw']:.2f} • "
                f"2 {pick['odds_away']:.2f} "
                f"({pick['bookmakers']} bookmaker)"
            )

else:

    st.warning(
        "Non ci sono almeno 3 selezioni "
        "che superano la soglia impostata."
    )


# ============================================================
# QUADRUPLA
# ============================================================

st.subheader("🔥 Quadrupla V8.1")

if len(eligible) >= 4:

    top4 = eligible[:4]

    for i, pick in enumerate(
        top4,
        1
    ):

        with st.container(
            border=True
        ):

            st.markdown(
                f"**{i}. "
                f"{pick['home']} – "
                f"{pick['away']}**"
            )

            st.write(
                f"**{pick['market']}**"
            )

            st.write(
                "Probabilità modello: "
                f"**{pct(pick['probability'])}**"
            )

            st.caption(
                f"{pick['league']} • "
                f"{pick['kickoff']}"
            )

else:

    st.warning(
        "Non ci sono almeno 4 selezioni "
        "che superano la soglia impostata."
    )


# ============================================================
# ALL ANALYSES
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
                    pct(
                        r[
                            "probability"
                        ]
                    ),
                "Score":
                    round(
                        r[
                            "score"
                        ],
                        3
                    ),
                "Storico":
                    r[
                        "history"
                    ],
            })

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Nessuna partita analizzabile."
        )


with st.expander(
    "⚠️ Partite non analizzate"
):

    if skipped:

        st.dataframe(
            skipped,
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

    manual_league_name = st.selectbox(
        "Campionato",
        list(
            BIG5.keys()
        ),
        key="manual_league",
    )

    manual_league = BIG5[
        manual_league_name
    ]["model_id"]

    teams = get_model_teams(
        manual_league
    )

    col1, col2 = st.columns(2)

    with col1:

        home = st.selectbox(
            "Squadra casa",
            teams,
            key="manual_home",
        )

    with col2:

        away_options = [
            t
            for t in teams
            if t != home
        ]

        away = st.selectbox(
            "Squadra ospite",
            away_options,
            key="manual_away",
        )

    st.write("### Quote 1X2")

    c1, c2, c3 = st.columns(3)

    with c1:

        oddsH = st.number_input(
            "1",
            min_value=1.01,
            value=2.00,
            step=0.05,
            key="manual_odds_h",
        )

    with c2:

        oddsD = st.number_input(
            "X",
            min_value=1.01,
            value=3.30,
            step=0.05,
            key="manual_odds_d",
        )

    with c3:

        oddsA = st.number_input(
            "2",
            min_value=1.01,
            value=3.50,
            step=0.05,
            key="manual_odds_a",
        )

    if st.button(
        "Analizza manualmente",
        use_container_width=True,
    ):

        result = predict_match(
            bundle=bundle,
            league=manual_league,
            home=home,
            away=away,
            oddsH=oddsH,
            oddsD=oddsD,
            oddsA=oddsA,
            referee=None,
            official=False,
        )

        markets = result[
            "markets"
        ]

        st.success(
            f"{home} vs {away}"
        )

        x1, x2 = st.columns(2)

        with x1:

            st.metric(
                "Over 2.5 cartellini",
                pct(
                    markets[
                        "match_O2.5"
                    ]
                )
            )

            st.metric(
                f"{home} Over 1.5",
                pct(
                    markets[
                        "home_team_O1.5"
                    ]
                )
            )

            st.metric(
                f"{away} Over 1.5",
                pct(
                    markets[
                        "away_team_O1.5"
                    ]
                )
            )

        with x2:

            st.metric(
                "Under 3.5 cartellini",
                pct(
                    markets[
                        "match_U3.5"
                    ]
                )
            )

            st.metric(
                f"{home} Under 1.5",
                pct(
                    markets[
                        "home_team_U1.5"
                    ]
                )
            )

            st.metric(
                f"{away} Under 1.5",
                pct(
                    markets[
                        "away_team_U1.5"
                    ]
                )
            )


st.divider()

st.caption(
    "V8.1 Cards Model • "
    "Yellow cards only • "
    "Pre-match probabilities"
)
