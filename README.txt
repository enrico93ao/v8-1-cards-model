V8.1 CARDS MODEL — EXECUTABLE BUNDLE
Created: 2026-08-26

WHAT THIS FIXES
The model is now persisted as an actual executable LightGBM artifact instead of only a described/reconstructed methodology.
Future probabilities can be reproduced from the saved bundle.

FILES
- v8_1_cards_bundle.joblib : trained probable-XI + official-XI models, calibrators, history state
- predict_v8_1.py          : feature builder + prediction functions
- v8_1_metadata.json       : frozen specification and validation metrics

DATA / TARGET
- Big Five, 2017/18 through 2025/26
- 16,110 matches / 32,220 team observations
- Yellow cards only
- Team distribution: P(0), P(1), P(2), P(3), P(4+)

KEY FEATURES
- rolling cards received 10/20
- rolling cards forced 10/20
- rolling fouls committed/suffered 10/20
- team card distribution
- home/away
- 1X2 de-vig strength / equilibrium
- league baseline
- referee prior where available (Premier League historical source)
- probable XI player yellow/foul propensity with 1,800-minute shrinkage
- low-history/newcomer count

PROBABLE XI
Historical backtest uses previous-match XI as the day-before proxy.
2025/26 mean overlap with official starters: 8.313/11.

VALIDATION (untouched 2025/26; recency half-life 5 seasons; raw pre-calibration)
Core Team O1.5 AUC: 0.6154
V8.1 probable-XI Team O1.5 AUC: 0.6199
V8.1 probable-XI Match U3.5 AUC: 0.6247
V8.1 probable-XI Match O2.5 AUC: 0.6031

USAGE
Python:
    from predict_v8_1 import load_bundle, predict_match
    b = load_bundle()
    result = predict_match(
        b, "SP1", "Real Madrid", "Sociedad",
        oddsH=1.35, oddsD=5.50, oddsA=8.50
    )
    print(result)

If no XI player IDs are passed, the bundle uses the last official starting XI as the probable-XI proxy.
For a real day-before run, external probable lineups can replace that proxy.

STATE UPDATES
Use update_completed_match(...) to add current-season completed matches in chronological order before predicting future fixtures.
