import streamlit as st

from predict_v8_1 import load_bundle, predict_match


st.set_page_config(
    page_title="V8.1 Cards Model",
    page_icon="🟨",
    layout="centered",
)


@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()

LEAGUES = {
    "Premier League": "E0",
    "Bundesliga": "D1",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Ligue 1": "F1",
}


def pct(value):
    return f"{value * 100:.1f}%"


st.title("🟨 V8.1 Cards Model")
st.caption("Yellow Cards Prediction Engine")

st.divider()

league_name = st.selectbox(
    "Campionato",
    list(LEAGUES.keys())
)

league = LEAGUES[league_name]

# Squadre presenti nel database del modello
teams = sorted(
    {
        team
        for lg, team in bundle["state"]["team_hist"].keys()
        if lg == league
    }
)

col1, col2 = st.columns(2)

with col1:
    home = st.selectbox(
        "🏠 Squadra casa",
        teams,
        index=0 if teams else None
    )

with col2:
    away_options = [t for t in teams if t != home]

    away = st.selectbox(
        "✈️ Squadra ospite",
        away_options,
        index=0 if away_options else None
    )


st.subheader("Quote 1X2")

c1, c2, c3 = st.columns(3)

with c1:
    oddsH = st.number_input(
        "1",
        min_value=1.01,
        value=2.00,
        step=0.05
    )

with c2:
    oddsD = st.number_input(
        "X",
        min_value=1.01,
        value=3.30,
        step=0.05
    )

with c3:
    oddsA = st.number_input(
        "2",
        min_value=1.01,
        value=3.50,
        step=0.05
    )


# Arbitri conosciuti dal bundle
referees = sorted(bundle["state"].get("ref_hist", {}).keys())

referee_choice = st.selectbox(
    "Arbitro (opzionale)",
    ["Non specificato"] + referees
)

referee = (
    None
    if referee_choice == "Non specificato"
    else referee_choice
)


st.divider()

analyze = st.button(
    "🔎 ANALIZZA PARTITA",
    type="primary",
    use_container_width=True
)


if analyze:

    if home == away:
        st.error("La squadra di casa e quella ospite devono essere diverse.")

    else:

        try:

            result = predict_match(
                bundle=bundle,
                league=league,
                home=home,
                away=away,
                oddsH=oddsH,
                oddsD=oddsD,
                oddsA=oddsA,
                referee=referee,
                official=False,
            )

            markets = result["markets"]

            st.success(
                f"{home} vs {away}"
            )

            st.subheader("Probabilità V8.1")

            m1, m2 = st.columns(2)

            with m1:

                st.metric(
                    "Over 2.5 cartellini",
                    pct(markets["match_O2.5"])
                )

                st.metric(
                    f"{home} Over 1.5",
                    pct(markets["home_team_O1.5"])
                )

                st.metric(
                    f"{away} Over 1.5",
                    pct(markets["away_team_O1.5"])
                )

            with m2:

                st.metric(
                    "Under 3.5 cartellini",
                    pct(markets["match_U3.5"])
                )

                st.metric(
                    f"{home} Under 1.5",
                    pct(markets["home_team_U1.5"])
                )

                st.metric(
                    f"{away} Under 1.5",
                    pct(markets["away_team_U1.5"])
                )


            with st.expander("Distribuzione cartellini per squadra"):

                st.write(f"### {home}")

                home_dist = result[
                    "home_distribution_0_1_2_3_4plus"
                ]

                labels = [
                    "0 cartellini",
                    "1 cartellino",
                    "2 cartellini",
                    "3 cartellini",
                    "4+ cartellini",
                ]

                for label, probability in zip(labels, home_dist):
                    st.write(
                        f"{label}: **{pct(probability)}**"
                    )


                st.write(f"### {away}")

                away_dist = result[
                    "away_distribution_0_1_2_3_4plus"
                ]

                for label, probability in zip(labels, away_dist):
                    st.write(
                        f"{label}: **{pct(probability)}**"
                    )


            with st.expander("Formazioni probabili utilizzate"):

                st.write(f"**{home}**")

                home_xi = result.get(
                    "probable_xi_home_names"
                )

                if home_xi:
                    for player in home_xi:
                        st.write("•", player)
                else:
                    st.write("Dati non disponibili")


                st.write(f"**{away}**")

                away_xi = result.get(
                    "probable_xi_away_names"
                )

                if away_xi:
                    for player in away_xi:
                        st.write("•", player)
                else:
                    st.write("Dati non disponibili")


            st.caption(
                f"Model version: {result['version']} "
                f"• Trained through: {bundle.get('trained_through', 'N/D')}"
            )

        except Exception as e:

            st.error(
                "Errore durante l'elaborazione del modello."
            )

            st.exception(e)


st.divider()

st.caption(
    "V8.1 Cards Model • Yellow cards only"
)
