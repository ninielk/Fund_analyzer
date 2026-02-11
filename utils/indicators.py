# ==============================================================================
# INDICATORS - Calcul de tous les indicateurs de performance
# ==============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import TRADING_DAYS_PER_YEAR, OMEGA_THRESHOLD


# ==============================================================================
# TAUX SANS RISQUE (EURIBOR)
# ==============================================================================

def get_euribor_rate(tenor: str = "3M") -> float:
    """
    Récupère le taux Euribor actuel via l'API de la BCE ou Yahoo Finance.
    
    Parameters
    ----------
    tenor : str
        Maturité: "1M", "3M", "6M", "12M"
    
    Returns
    -------
    float
        Taux Euribor annualisé (en décimal, ex: 0.035 pour 3.5%)
    
    Note
    ----
    Si la récupération échoue, retourne le taux par défaut de config.py
    """
    try:
        import yfinance as yf
        
        # Mapping des tickers Yahoo Finance pour Euribor
        euribor_tickers = {
            "1M": "^EURIBOR1M",
            "3M": "^EURIBOR3M",  
            "6M": "^EURIBOR6M",
            "12M": "^EURIBOR1Y",
            "1Y": "^EURIBOR1Y"
        }
        
        ticker = euribor_tickers.get(tenor.upper(), "^EURIBOR3M")
        data = yf.Ticker(ticker)
        hist = data.history(period="5d")
        
        if not hist.empty:
            # Le taux est en pourcentage, convertir en décimal
            rate = hist['Close'].iloc[-1] / 100
            return rate
    except Exception as e:
        print(f"Erreur récupération Euribor: {e}")
    
    # Fallback: taux par défaut
    from config import DEFAULT_RISK_FREE_RATE
    return DEFAULT_RISK_FREE_RATE


def get_risk_free_rate_series(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    tenor: str = "3M"
) -> pd.Series:
    """
    Récupère la série historique des taux Euribor.
    
    Utile pour des calculs plus précis avec taux variable.
    """
    try:
        import yfinance as yf
        
        euribor_tickers = {
            "1M": "^EURIBOR1M",
            "3M": "^EURIBOR3M",
            "6M": "^EURIBOR6M",
            "12M": "^EURIBOR1Y"
        }
        
        ticker = euribor_tickers.get(tenor.upper(), "^EURIBOR3M")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if not data.empty:
            return data['Close'] / 100  # Convertir en décimal
    except Exception as e:
        print(f"Erreur récupération série Euribor: {e}")
    
    return None


# ==============================================================================
# INDICATEURS PAR PERIODE (1Y, 3Y, 5Y)
# ==============================================================================

def get_period_returns(returns: pd.Series, years: int) -> pd.Series:
    """
    Extrait les rendements des N dernières années.
    
    Parameters
    ----------
    returns : pd.Series
        Série complète des rendements
    years : int
        Nombre d'années à extraire
    
    Returns
    -------
    pd.Series
        Rendements sur la période (ou série vide si pas assez de données)
    """
    if len(returns) < 1:
        return pd.Series()
    
    n_days = years * TRADING_DAYS_PER_YEAR
    
    if len(returns) < n_days:
        return pd.Series()  # Pas assez de données
    
    return returns.iloc[-n_days:]


def return_1y(returns: pd.Series) -> float:
    """Rendement sur 1 an (annualisé)."""
    period_ret = get_period_returns(returns, 1)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * 0.8:  # Au moins 80% des jours
        return np.nan
    return annualized_return(period_ret)


def return_3y(returns: pd.Series) -> float:
    """Rendement sur 3 ans (annualisé)."""
    period_ret = get_period_returns(returns, 3)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * 2.5:  # Au moins ~2.5 ans
        return np.nan
    return annualized_return(period_ret)


def return_5y(returns: pd.Series) -> float:
    """Rendement sur 5 ans (annualisé)."""
    period_ret = get_period_returns(returns, 5)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * 4:  # Au moins 4 ans
        return np.nan
    return annualized_return(period_ret)


def volatility_1y(returns: pd.Series) -> float:
    """Volatilité sur 1 an (annualisée)."""
    period_ret = get_period_returns(returns, 1)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * 0.8:
        return np.nan
    return annualized_volatility(period_ret)


def volatility_3y(returns: pd.Series) -> float:
    """Volatilité sur 3 ans (annualisée)."""
    period_ret = get_period_returns(returns, 3)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * 2.5:
        return np.nan
    return annualized_volatility(period_ret)


def volatility_5y(returns: pd.Series) -> float:
    """Volatilité sur 5 ans (annualisée)."""
    period_ret = get_period_returns(returns, 5)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * 4:
        return np.nan
    return annualized_volatility(period_ret)


def sharpe_1y(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Sharpe Ratio sur 1 an."""
    period_ret = get_period_returns(returns, 1)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * 0.8:
        return np.nan
    return sharpe_ratio(period_ret, risk_free_rate)


def sharpe_3y(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Sharpe Ratio sur 3 ans."""
    period_ret = get_period_returns(returns, 3)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * 2.5:
        return np.nan
    return sharpe_ratio(period_ret, risk_free_rate)


def sharpe_5y(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Sharpe Ratio sur 5 ans."""
    period_ret = get_period_returns(returns, 5)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * 4:
        return np.nan
    return sharpe_ratio(period_ret, risk_free_rate)


# ==============================================================================
# INDICATEURS INDIVIDUELS
# ==============================================================================

def annualized_return(returns: pd.Series) -> float:
    """
    Calcule le rendement annualisé.
    
    Formule: (1 + rendement_total)^(252/n_jours) - 1
    
    Explication Grand-Mere:
    "C'est combien tu gagnes en moyenne par an si tu gardes ton placement"
    """
    if len(returns) < 2:
        return np.nan
    
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    
    if n_years <= 0:
        return np.nan
    
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    return ann_return


def annualized_volatility(returns: pd.Series) -> float:
    """
    Calcule la volatilité annualisée (écart-type).
    
    Formule: ecart_type_quotidien * sqrt(252)
    
    Explication Grand-Mere:
    "C'est à quel point ton placement fait les montagnes russes.
    Plus c'est haut, plus c'est agité."
    """
    if len(returns) < 2:
        return np.nan
    
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    Calcule le ratio de Sharpe.
    
    Formule: (rendement_annualisé - taux_sans_risque) / volatilité_annualisée
    
    Explication Grand-Mere:
    "Est-ce que le risque que tu prends en vaut la peine ?
    Plus c'est haut, mieux c'est. Au-dessus de 1 c'est bien,
    au-dessus de 2 c'est très bien."
    """
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    
    if pd.isna(ann_vol) or ann_vol == 0:
        return np.nan
    
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    Calcule le ratio de Sortino.
    
    Formule: (rendement_annualisé - taux_sans_risque) / volatilité_négative_annualisée
    
    Explication Grand-Mere:
    "Comme le Sharpe, mais on punit seulement les baisses, pas les hausses.
    C'est plus juste parce qu'on s'en fiche si ça monte beaucoup !"
    """
    ann_ret = annualized_return(returns)
    
    # Downside deviation: écart-type des rendements négatifs seulement
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) < 2:
        return np.nan
    
    downside_vol = negative_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    if downside_vol == 0:
        return np.nan
    
    return (ann_ret - risk_free_rate) / downside_vol


def semi_variance(returns: pd.Series) -> float:
    """
    Calcule la semi-variance (variance des rendements négatifs).
    
    Formule: variance des rendements en-dessous de la moyenne
    
    Explication Grand-Mere:
    "Ça mesure seulement les mauvaises surprises, pas les bonnes.
    Plus c'est bas, moins tu as de risque de perdre gros."
    """
    mean_return = returns.mean()
    negative_returns = returns[returns < mean_return]
    
    if len(negative_returns) < 2:
        return np.nan
    
    return negative_returns.var()


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calcule le Beta par rapport au benchmark.
    
    Formule: Covariance(fonds, marché) / Variance(marché)
    
    Explication Grand-Mere:
    "Quand le marché bouge de 1%, ton fonds bouge de combien ?
    Beta = 1 : pareil que le marché
    Beta > 1 : plus nerveux que le marché
    Beta < 1 : plus calme que le marché"
    """
    # Aligner les séries
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    
    if len(aligned) < 10:
        return np.nan
    
    fund_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]
    
    covariance = np.cov(fund_ret, bench_ret)[0, 1]
    variance_benchmark = np.var(bench_ret)
    
    if variance_benchmark == 0:
        return np.nan
    
    return covariance / variance_benchmark


def alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    Calcule l'Alpha de Jensen (CAPM).
    
    Formule: Rendement_fonds - (Rf + Beta * (Rm - Rf))
    
    Explication Grand-Mere:
    "C'est le petit plus (ou le petit moins) que le gérant apporte
    par rapport à ce qu'on attendait vu le risque pris.
    Positif = le gérant est bon. Négatif = il fait moins bien que prévu."
    """
    beta_val = beta(returns, benchmark_returns)
    
    if pd.isna(beta_val):
        return np.nan
    
    ann_ret_fund = annualized_return(returns)
    ann_ret_bench = annualized_return(benchmark_returns)
    
    expected_return = risk_free_rate + beta_val * (ann_ret_bench - risk_free_rate)
    
    return ann_ret_fund - expected_return


def max_drawdown(prices: pd.Series) -> float:
    """
    Calcule le Maximum Drawdown.
    
    Formule: (Creux - Pic) / Pic (la pire chute depuis un sommet)
    
    Explication Grand-Mere:
    "C'est la pire dégringolade que tu aurais pu subir si tu avais
    acheté au pire moment et vendu au pire moment.
    -20% veut dire que dans le pire des cas, tu aurais perdu 20%."
    """
    if len(prices) < 2:
        return np.nan
    
    # Calculer le pic cumulé (running maximum)
    running_max = prices.cummax()
    
    # Drawdown à chaque instant
    drawdown = (prices - running_max) / running_max
    
    # Maximum Drawdown = le pire drawdown
    return drawdown.min()


def calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    """
    Calcule le ratio de Calmar.
    
    Formule: Rendement annualisé / |Max Drawdown|
    
    Explication Grand-Mere:
    "C'est le rendement par rapport à la pire chute.
    Plus c'est haut, plus tu es bien récompensé pour le risque de grosse perte."
    """
    ann_ret = annualized_return(returns)
    mdd = max_drawdown(prices)
    
    if pd.isna(mdd) or mdd == 0:
        return np.nan
    
    return ann_ret / abs(mdd)


def omega_ratio(returns: pd.Series, threshold: float = None) -> float:
    """
    Calcule le ratio Omega.
    
    Formule: Somme des gains au-dessus du seuil / Somme des pertes en-dessous
    
    Explication Grand-Mere:
    "Pour chaque euro que tu risques de perdre, combien tu peux gagner ?
    Omega > 1 = tu gagnes plus que tu perds en moyenne.
    Plus c'est haut, mieux c'est."
    """
    if threshold is None:
        threshold = OMEGA_THRESHOLD
    
    # Gains au-dessus du seuil
    gains = returns[returns > threshold] - threshold
    
    # Pertes en-dessous du seuil
    losses = threshold - returns[returns <= threshold]
    
    sum_gains = gains.sum() if len(gains) > 0 else 0
    sum_losses = losses.sum() if len(losses) > 0 else 0
    
    if sum_losses == 0:
        return np.inf if sum_gains > 0 else np.nan
    
    return sum_gains / sum_losses


def pct_beats_benchmark(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calcule le pourcentage de périodes où le fonds bat le benchmark.
    
    Formule: Nb(jours où fonds > benchmark) / Nb(jours total)
    
    Explication Grand-Mere:
    "Sur 100 jours, combien de fois ton fonds a fait mieux que le marché ?
    50% = pareil que le marché en moyenne.
    Plus c'est haut, plus souvent tu gagnes."
    """
    # Aligner les séries
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    
    if len(aligned) < 5:
        return np.nan
    
    fund_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]
    
    beats = (fund_ret > bench_ret).sum()
    total = len(fund_ret)
    
    return beats / total


def cumulative_return(returns: pd.Series) -> float:
    """
    Calcule le rendement cumulé total.
    
    Formule: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    
    Explication Grand-Mere:
    "Si tu avais mis 100 euros au début, combien tu aurais maintenant ?"
    """
    if len(returns) < 1:
        return np.nan
    
    return (1 + returns).prod() - 1


# ==============================================================================
# FONCTION PRINCIPALE - CALCUL DE TOUS LES INDICATEURS
# ==============================================================================

def calculate_all_indicators(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_col: str,
    risk_free_rate: float = 0.03
) -> pd.DataFrame:
    """
    Calcule tous les indicateurs pour tous les fonds.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame des prix
    returns : pd.DataFrame
        DataFrame des rendements
    benchmark_col : str
        Nom de la colonne benchmark
    risk_free_rate : float
        Taux sans risque annualisé
    
    Returns
    -------
    pd.DataFrame
        Tableau récapitulatif de tous les indicateurs par fonds
    """
    results = []
    
    # Identifier le benchmark
    bench_prices = prices[benchmark_col] if benchmark_col in prices.columns else None
    bench_returns = returns[benchmark_col] if benchmark_col in returns.columns else None
    
    # Colonnes des fonds (exclure benchmark)
    fund_cols = [col for col in prices.columns if col.lower() != benchmark_col.lower()]
    
    for fund in fund_cols:
        fund_prices = prices[fund].dropna()
        fund_returns = returns[fund].dropna()
        
        if len(fund_returns) < 10:
            continue
        
        indicators = {
            'Fonds': fund,
            # Indicateurs globaux (sur toute la période)
            'Rendement Annualise (%)': annualized_return(fund_returns) * 100,
            'Volatilite (%)': annualized_volatility(fund_returns) * 100,
            'Sharpe Ratio': sharpe_ratio(fund_returns, risk_free_rate),
            'Sortino Ratio': sortino_ratio(fund_returns, risk_free_rate),
            
            # Rendements par période
            'Rendement 1Y (%)': return_1y(fund_returns) * 100 if not np.isnan(return_1y(fund_returns)) else np.nan,
            'Rendement 3Y (%)': return_3y(fund_returns) * 100 if not np.isnan(return_3y(fund_returns)) else np.nan,
            'Rendement 5Y (%)': return_5y(fund_returns) * 100 if not np.isnan(return_5y(fund_returns)) else np.nan,
            
            # Volatilité par période
            'Volatilite 1Y (%)': volatility_1y(fund_returns) * 100 if not np.isnan(volatility_1y(fund_returns)) else np.nan,
            'Volatilite 3Y (%)': volatility_3y(fund_returns) * 100 if not np.isnan(volatility_3y(fund_returns)) else np.nan,
            'Volatilite 5Y (%)': volatility_5y(fund_returns) * 100 if not np.isnan(volatility_5y(fund_returns)) else np.nan,
            
            # Sharpe par période
            'Sharpe 1Y': sharpe_1y(fund_returns, risk_free_rate),
            'Sharpe 3Y': sharpe_3y(fund_returns, risk_free_rate),
            'Sharpe 5Y': sharpe_5y(fund_returns, risk_free_rate),
            
            # Autres indicateurs
            'Semi-Variance': semi_variance(fund_returns),
            'Max Drawdown (%)': max_drawdown(fund_prices) * 100,
            'Calmar Ratio': calmar_ratio(fund_returns, fund_prices),
            'Omega Ratio': omega_ratio(fund_returns),
            'Rendement Cumule (%)': cumulative_return(fund_returns) * 100,
        }
        
        # Indicateurs relatifs au benchmark
        if bench_returns is not None:
            indicators['Beta'] = beta(fund_returns, bench_returns)
            indicators['Alpha (%)'] = alpha(fund_returns, bench_returns, risk_free_rate) * 100
            indicators['% Bat le Benchmark'] = pct_beats_benchmark(fund_returns, bench_returns) * 100
        
        results.append(indicators)
    
    df_results = pd.DataFrame(results)
    
    # Arrondir pour lisibilité
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns
    df_results[numeric_cols] = df_results[numeric_cols].round(4)
    
    return df_results


def calculate_indicator(
    indicator_name: str,
    returns: pd.Series,
    prices: pd.Series = None,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.03
) -> float:
    """
    Calcule un indicateur spécifique.
    
    Parameters
    ----------
    indicator_name : str
        Nom de l'indicateur
    returns : pd.Series
        Rendements du fonds
    prices : pd.Series, optional
        Prix du fonds (pour max drawdown, calmar)
    benchmark_returns : pd.Series, optional
        Rendements du benchmark (pour beta, alpha)
    risk_free_rate : float
        Taux sans risque
    
    Returns
    -------
    float
        Valeur de l'indicateur
    """
    indicator_map = {
        'rendement': lambda: annualized_return(returns),
        'volatilite': lambda: annualized_volatility(returns),
        'sharpe': lambda: sharpe_ratio(returns, risk_free_rate),
        'sortino': lambda: sortino_ratio(returns, risk_free_rate),
        'semi_variance': lambda: semi_variance(returns),
        'beta': lambda: beta(returns, benchmark_returns) if benchmark_returns is not None else np.nan,
        'alpha': lambda: alpha(returns, benchmark_returns, risk_free_rate) if benchmark_returns is not None else np.nan,
        'max_drawdown': lambda: max_drawdown(prices) if prices is not None else np.nan,
        'calmar': lambda: calmar_ratio(returns, prices) if prices is not None else np.nan,
        'omega': lambda: omega_ratio(returns),
        'pct_beats': lambda: pct_beats_benchmark(returns, benchmark_returns) if benchmark_returns is not None else np.nan,
    }
    
    indicator_name = indicator_name.lower().replace(' ', '_')
    
    if indicator_name in indicator_map:
        return indicator_map[indicator_name]()
    else:
        raise ValueError(f"Indicateur inconnu: {indicator_name}")


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 63,  # ~3 mois
    risk_free_rate: float = 0.03
) -> pd.Series:
    """
    Calcule le Sharpe ratio glissant.
    
    Parameters
    ----------
    returns : pd.Series
        Rendements quotidiens
    window : int
        Taille de la fenêtre (en jours)
    risk_free_rate : float
        Taux sans risque annualisé
    
    Returns
    -------
    pd.Series
        Sharpe ratio glissant
    """
    rf_daily = (1 + risk_free_rate) ** (1/TRADING_DAYS_PER_YEAR) - 1
    
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    rolling_sharpe = (rolling_mean - rf_daily) / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return rolling_sharpe


def calculate_drawdown_series(prices: pd.Series) -> pd.Series:
    """
    Calcule la série temporelle des drawdowns.
    
    Parameters
    ----------
    prices : pd.Series
        Prix du fonds
    
    Returns
    -------
    pd.Series
        Drawdown à chaque date (en %)
    """
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max * 100
    return drawdown
