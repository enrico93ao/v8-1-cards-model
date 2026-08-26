#!/usr/bin/env python3
"""
V8.1 Cards executable prediction helper.
Target: yellow cards only. Team distribution classes = 0,1,2,3,4+.
"""
import math
from collections import deque
from pathlib import Path
import joblib
import numpy as np

COMP_MAP = {'E0':'EPL','D1':'Bundesliga','SP1':'La_Liga','I1':'Serie_A','F1':'Ligue_1'}

def _mean_last(vals,w):
    x=[v for v in list(vals)[-w:] if v is not None]
    return float(np.mean(x)) if x else np.nan

def _std_last(vals,w):
    x=[v for v in list(vals)[-w:] if v is not None]
    return float(np.std(x)) if x else np.nan

def _dist_last(vals,w):
    x=[int(min(4,max(0,v))) for v in list(vals)[-w:] if v is not None]
    if not x: return [np.nan]*5
    return [x.count(k)/len(x) for k in range(5)]

def _logit(p):
    p=float(np.clip(p,1e-5,1-1e-5))
    return math.log(p/(1-p))

def load_bundle(path=None):
    if path is None:
        path=Path(__file__).with_name("v8_1_cards_bundle.joblib")
    return joblib.load(path)

def xi_features_from_ids(bundle, player_ids, league):
    st=bundle['state']
    comp=COMP_MAP[league]
    lm,ly,lf=st['league_player_agg'].get(comp,[0.,0.,0.])
    base_y90=(ly/lm*90.) if lm else 0.20
    base_f90=(lf/lm*90.) if lm else 1.20
    shrink=bundle['hyperparams']['player_shrinkage_minutes']
    ys=[]; fs=[]; mins=[]
    for pid in player_ids or []:
        pm,py,pf=st['player_agg'].get(pid,[0.,0.,0.])
        y90=(py + base_y90*(shrink/90.))/((pm+shrink)/90.)
        f90=(pf + base_f90*(shrink/90.))/((pm+shrink)/90.)
        ys.append(y90); fs.append(f90); mins.append(pm)
    if not ys:
        return {'xi_yellow_sum':np.nan,'xi_fouls_sum':np.nan,'xi_lowhist':np.nan,
                'xi_avg_prior_mins':np.nan,'xi_n':0}
    return {'xi_yellow_sum':float(sum(ys)), 'xi_fouls_sum':float(sum(fs)),
            'xi_lowhist':float(sum(m<450 for m in mins)),
            'xi_avg_prior_mins':float(np.mean(mins)), 'xi_n':len(ys)}

def default_probable_xi_ids(bundle, league, team):
    tid=bundle['state']['team_id_map'].get((league,team))
    if not tid: return []
    return list(bundle['state']['prev_starters'].get(tid,[]))

def probable_xi_names(bundle, league, team):
    ids=default_probable_xi_ids(bundle,league,team)
    nm=bundle['state'].get('player_name_map',{})
    return [nm.get(pid,pid) for pid in ids]

def build_team_feature(bundle, league, team, opp, is_home, oddsH, oddsD, oddsA,
                       xi_ids=None, referee=None, official=False):
    st=bundle['state']
    th=st['team_hist'].get((league,team),{'cards':[],'forced':[],'fouls':[],'fouls_suffer':[]})
    oh=st['team_hist'].get((league,opp),{'cards':[],'forced':[],'fouls':[],'fouls_suffer':[]})
    inv=np.array([1/oddsH,1/oddsD,1/oddsA],dtype=float)
    inv=inv/inv.sum()
    pH,pD,pA=inv.tolist()
    pwin=pH if is_home else pA
    plose=pA if is_home else pH
    ls,ln=st['league_totals'].get(league,[0.,0])
    league_mean=ls/ln if ln else np.nan
    refmean=np.nan; refn=0
    if referee:
        rs,rn=st['ref_hist'].get(referee,[0.,0])
        refn=rn
        base=league_mean if not math.isnan(league_mean) else 4.0
        refmean=(rs+10*base)/(rn+10)
    f={}
    for w in (10,20):
        f[f'cards_mean{w}']=_mean_last(th['cards'],w)
        f[f'cards_std{w}']=_std_last(th['cards'],w)
        f[f'forced_mean{w}']=_mean_last(th['forced'],w)
        f[f'fouls_mean{w}']=_mean_last(th['fouls'],w)
        f[f'fouls_suffer_mean{w}']=_mean_last(th['fouls_suffer'],w)
        f[f'opp_cards_mean{w}']=_mean_last(oh['cards'],w)
        f[f'opp_forced_mean{w}']=_mean_last(oh['forced'],w)
        f[f'opp_fouls_mean{w}']=_mean_last(oh['fouls'],w)
        f[f'opp_fouls_suffer_mean{w}']=_mean_last(oh['fouls_suffer'],w)
        d=_dist_last(th['cards'],w)
        for k,v in enumerate(d):
            f[f'cards_p{k if k<4 else "4p"}_{w}']=v
    f.update({
        'is_home':1 if is_home else 0,
        'hist_n':min(20,len(th['cards'])), 'opp_hist_n':min(20,len(oh['cards'])),
        'pwin':pwin,'pdraw':pD,'plose':plose,'strength_diff':pwin-plose,
        'odds_entropy':-(pH*math.log(pH)+pD*math.log(pD)+pA*math.log(pA)),
        'league_card_mean':league_mean,'ref_card_mean':refmean,'ref_n':refn,
        'league_id':bundle['league_ids'][league],
    })
    if xi_ids is None:
        xi_ids=default_probable_xi_ids(bundle,league,team)
    xf=xi_features_from_ids(bundle,xi_ids,league)
    prefix='off_' if official else 'prob_'
    for k,v in xf.items():
        f[prefix+k]=v
    return f


def update_completed_match(bundle, league, home, away, home_yellow, away_yellow,
                           home_fouls=None, away_fouls=None, referee=None):
    """Update the in-memory history state with one completed match.
    Call matches in chronological order. Does not retrain the model.
    """
    st=bundle['state']
    for team,opp,y,oy,f,of in [
        (home,away,home_yellow,away_yellow,home_fouls,away_fouls),
        (away,home,away_yellow,home_yellow,away_fouls,home_fouls),
    ]:
        key=(league,team)
        h=st['team_hist'].setdefault(key,{'cards':[],'forced':[],'fouls':[],'fouls_suffer':[]})
        h['cards']=(list(h.get('cards',[]))+[int(home_yellow if team==home else away_yellow)])[-50:]
        h['forced']=(list(h.get('forced',[]))+[int(away_yellow if team==home else home_yellow)])[-50:]
        h['fouls']=(list(h.get('fouls',[]))+[f])[-50:]
        h['fouls_suffer']=(list(h.get('fouls_suffer',[]))+[of])[-50:]
    lt=st['league_totals'].setdefault(league,[0.0,0])
    lt[0]+=float(home_yellow)+float(away_yellow); lt[1]+=1
    if referee:
        rh=st['ref_hist'].setdefault(referee,[0.0,0])
        rh[0]+=float(home_yellow)+float(away_yellow); rh[1]+=1
    return bundle

def _matrix(features, names):
    return np.array([[float(features.get(n,np.nan)) if features.get(n,None) is not None else np.nan
                      for n in names]],dtype=float)

def _calibrate(cal,p):
    return float(cal.predict_proba(np.array([[_logit(p)]]))[:,1][0])

def predict_match(bundle, league, home, away, oddsH, oddsD, oddsA,
                  home_xi_ids=None, away_xi_ids=None, referee=None, official=False):
    model=bundle['official_model'] if official else bundle['probable_model']
    names=bundle['official_features'] if official else bundle['probable_features']
    cals=bundle['calibrators_official'] if official else bundle['calibrators_probable']
    hf=build_team_feature(bundle,league,home,away,True,oddsH,oddsD,oddsA,home_xi_ids,referee,official)
    af=build_team_feature(bundle,league,away,home,False,oddsH,oddsD,oddsA,away_xi_ids,referee,official)
    ph=model.predict_proba(_matrix(hf,names))[0]
    pa=model.predict_proba(_matrix(af,names))[0]
    pu35=0.; po25=0.
    for i in range(5):
        for j in range(5):
            pp=ph[i]*pa[j]
            if i<4 and j<4 and i+j<=3: pu35+=pp
            if i==4 or j==4 or i+j>=3: po25+=pp
    h_o15=1-ph[0]-ph[1]; a_o15=1-pa[0]-pa[1]
    return {
        'version':bundle['version'],
        'home_distribution_0_1_2_3_4plus':[float(x) for x in ph],
        'away_distribution_0_1_2_3_4plus':[float(x) for x in pa],
        'markets':{
            'match_U3.5':_calibrate(cals['u35'],pu35),
            'match_O2.5':_calibrate(cals['o25'],po25),
            'home_team_O1.5':_calibrate(cals['team_o15'],h_o15),
            'home_team_U1.5':1-_calibrate(cals['team_o15'],h_o15),
            'away_team_O1.5':_calibrate(cals['team_o15'],a_o15),
            'away_team_U1.5':1-_calibrate(cals['team_o15'],a_o15),
        },
        'probable_xi_home_names':probable_xi_names(bundle,league,home) if home_xi_ids is None else None,
        'probable_xi_away_names':probable_xi_names(bundle,league,away) if away_xi_ids is None else None,
    }

if __name__=="__main__":
    print("Load this module and call predict_match(...). See README.txt.")
