# ==============================================================================
# INDICATORS - Calcul de tous les indicateurs de performance V2 CORRIGE
# ==============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import TRADING_DAYS_PER_YEAR, OMEGA_THRESHOLD


# ==============================================================================
# INDICATEURS DE BASE
# ==============================================================================

def annualized_return(returns: pd.Series) -> float:
    """Rendement annualise."""
    if len(returns) < 2:
        return np.nan
    
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    
    if n_years <= 0:
        return np.nan
    
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    return ann_return


def annualized_volatility(returns: pd.Series) -> float:
    """Volatilite annualisee (ecart-type)."""
    if len(returns) < 2:
        return np.nan
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Ratio de Sharpe."""
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    
    if pd.isna(ann_vol) or ann_vol == 0:
        return np.nan
    
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Ratio de Sortino."""
    ann_ret = annualized_return(returns)
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) < 2:
        return np.nan
    
    downside_vol = negative_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    if downside_vol == 0:
        return np.nan
    
    return (ann_ret - risk_free_rate) / downside_vol


def semi_variance(returns: pd.Series) -> float:
    """Semi-variance (variance des rendements en-dessous de la moyenne)."""
    if len(returns) < 2:
        return np.nan
    
    mean_return = returns.mean()
    downside_returns = returns[returns < mean_return]
    
    if len(downside_returns) < 2:
        return np.nan
    
    squared_deviations = (downside_returns - mean_return) ** 2
    return squared_deviations.mean()


def max_drawdown(prices: pd.Series) -> float:
    """Maximum Drawdown (pire chute depuis un sommet)."""
    if len(prices) < 2:
        return np.nan
    
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return drawdown.min()


def calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    """Ratio de Calmar = rendement annualise / |max drawdown|."""
    ann_ret = annualized_return(returns)
    mdd = max_drawdown(prices)
    
    if pd.isna(mdd) or mdd == 0:
        return np.nan
    
    return ann_ret / abs(mdd)


def omega_ratio(returns: pd.Series, threshold: float = None) -> float:
    """Ratio Omega."""
    if threshold is None:
        threshold = OMEGA_THRESHOLD
    
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    
    sum_gains = gains.sum() if len(gains) > 0 else 0
    sum_losses = losses.sum() if len(losses) > 0 else 0
    
    if sum_losses == 0:
        return np.inf if sum_gains > 0 else np.nan
    
    return sum_gains / sum_losses


def cumulative_return(returns: pd.Series) -> float:
    """Rendement cumule total."""
    if len(returns) < 1:
        return np.nan
    return (1 + returns).prod() - 1


# ==============================================================================
# INDICATEURS RELATIFS AU BENCHMARK
# ==============================================================================

def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Beta = Cov(fonds, benchmark) / Var(benchmark)."""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    
    if len(aligned) < 10:
        return np.nan
    
    fund_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]
    
    covariance = np.cov(fund_ret, bench_ret)[0, 1]
    variance_benchmark = np.var(bench_ret, ddof=1)
    
    if variance_benchmark == 0:
        return np.nan
    
    return covariance / variance_benchmark


def alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Alpha de Jensen (CAPM)."""
    beta_val = beta(returns, benchmark_returns)
    
    if pd.isna(beta_val):
        return np.nan
    
    ann_ret_fund = annualized_return(returns)
    ann_ret_bench = annualized_return(benchmark_returns)
    
    expected_return = risk_free_rate + beta_val * (ann_ret_bench - risk_free_rate)
    
    return ann_ret_fund - expected_return


def tracking_error(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Tracking Error = ecart-type annualise de (rendement_fonds - rendement_benchmark).
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    
    if len(aligned) < 10:
        return np.nan
    
    fund_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]
    
    diff = fund_ret - bench_ret
    te = diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return te


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    Information Ratio = Alpha / Tracking Error.
    """
    alpha_val = alpha(returns, benchmark_returns, risk_free_rate)
    te = tracking_error(returns, benchmark_returns)
    
    if pd.isna(te) or te == 0:
        return np.nan
    
    return alpha_val / te


def treynor_ratio(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    Treynor Ratio = (Rendement - Rf) / Beta.
    """
    ann_ret = annualized_return(returns)
    beta_val = beta(returns, benchmark_returns)
    
    if pd.isna(beta_val) or beta_val == 0:
        return np.nan
    
    return (ann_ret - risk_free_rate) / beta_val


# ==============================================================================
# NOUVEAU CALCUL : BAT BENCHMARK GLISSANT JOURNALIER (1Y/3Y/5Y)
# ==============================================================================

def bat_benchmark_rolling(
    fund_prices: pd.Series,
    benchmark_prices: pd.Series,
    horizon_years: int = 1
) -> tuple:
    """
    Calcule le nombre de fois ou le fonds bat le benchmark sur un horizon donne.
    METHODE GLISSANTE JOURNALIERE - VERSION OPTIMISEE.
    """
    # Aligner les series
    aligned = pd.concat([fund_prices, benchmark_prices], axis=1).dropna()
    aligned.columns = ['fund', 'bench']
    
    n_days = horizon_years * TRADING_DAYS_PER_YEAR
    
    if len(aligned) < n_days + 10:
        return np.nan, np.nan, np.nan
    
    # Calculer les rendements sur horizon_years en vectorise
    # On utilise shift pour decaler les prix de n_days
    fund_past = aligned['fund'].shift(n_days)
    bench_past = aligned['bench'].shift(n_days)
    
    # Rendement simple
    ret_fund = (aligned['fund'] - fund_past) / fund_past
    ret_bench = (aligned['bench'] - bench_past) / bench_past
    
    # Supprimer les NaN (debut de serie)
    valid_mask = ret_fund.notna() & ret_bench.notna()
    ret_fund = ret_fund[valid_mask]
    ret_bench = ret_bench[valid_mask]
    
    if len(ret_fund) == 0:
        return np.nan, np.nan, np.nan
    
    # Compter les victoires
    nb_beats = (ret_fund > ret_bench).sum()
    total_obs = len(ret_fund)
    ratio_pct = (nb_beats / total_obs) * 100
    
    return int(nb_beats), int(total_obs), ratio_pct


# ==============================================================================
# INDICATEURS PAR PERIODE (1Y, 3Y, 5Y)
# ==============================================================================

def get_period_returns(returns: pd.Series, years: int) -> pd.Series:
    """Extrait les rendements des N dernieres annees."""
    if len(returns) < 1:
        return pd.Series()
    
    n_days = years * TRADING_DAYS_PER_YEAR
    
    if len(returns) < n_days:
        return pd.Series()
    
    return returns.iloc[-n_days:]


def return_period(returns: pd.Series, years: int) -> float:
    """Rendement annualise sur N ans."""
    period_ret = get_period_returns(returns, years)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * years * 0.8:
        return np.nan
    return annualized_return(period_ret)


def volatility_period(returns: pd.Series, years: int) -> float:
    """Volatilite annualisee sur N ans."""
    period_ret = get_period_returns(returns, years)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * years * 0.8:
        return np.nan
    return annualized_volatility(period_ret)


def sharpe_period(returns: pd.Series, years: int, risk_free_rate: float = 0.03) -> float:
    """Sharpe Ratio sur N ans."""
    period_ret = get_period_returns(returns, years)
    if len(period_ret) < TRADING_DAYS_PER_YEAR * years * 0.8:
        return np.nan
    return sharpe_ratio(period_ret, risk_free_rate)


# ==============================================================================
# FONCTION PRINCIPALE - CALCUL POUR UN BENCHMARK DONNE
# ==============================================================================

def calculate_all_indicators(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_col: str,
    risk_free_rate: float = 0.03
) -> pd.DataFrame:
    """
    Calcule tous les indicateurs pour tous les fonds par rapport a UN benchmark.
    """
    results = []
    
    fund_cols = [col for col in prices.columns if col != benchmark_col]
    
    bench_prices = prices[benchmark_col] if benchmark_col in prices.columns else None
    bench_returns = returns[benchmark_col] if benchmark_col in returns.columns else None
    
    for fund in fund_cols:
        fund_prices = prices[fund].dropna()
        fund_returns = returns[fund].dropna()
        
        if len(fund_returns) < 10:
            continue
        
        indicators = {
            'Fonds': fund,
            
            # Rendements
            'Rendement Annualise (%)': annualized_return(fund_returns) * 100,
            'Rendement 1Y (%)': return_period(fund_returns, 1) * 100 if not np.isnan(return_period(fund_returns, 1)) else np.nan,
            'Rendement 3Y (%)': return_period(fund_returns, 3) * 100 if not np.isnan(return_period(fund_returns, 3)) else np.nan,
            'Rendement 5Y (%)': return_period(fund_returns, 5) * 100 if not np.isnan(return_period(fund_returns, 5)) else np.nan,
            'Rendement Cumule (%)': cumulative_return(fund_returns) * 100,
            
            # Volatilite
            'Volatilite (%)': annualized_volatility(fund_returns) * 100,
            'Volatilite 1Y (%)': volatility_period(fund_returns, 1) * 100 if not np.isnan(volatility_period(fund_returns, 1)) else np.nan,
            'Volatilite 3Y (%)': volatility_period(fund_returns, 3) * 100 if not np.isnan(volatility_period(fund_returns, 3)) else np.nan,
            'Volatilite 5Y (%)': volatility_period(fund_returns, 5) * 100 if not np.isnan(volatility_period(fund_returns, 5)) else np.nan,
            
            # Ratios
            'Sharpe Ratio': sharpe_ratio(fund_returns, risk_free_rate),
            'Sharpe 1Y': sharpe_period(fund_returns, 1, risk_free_rate),
            'Sharpe 3Y': sharpe_period(fund_returns, 3, risk_free_rate),
            'Sharpe 5Y': sharpe_period(fund_returns, 5, risk_free_rate),
            'Sortino Ratio': sortino_ratio(fund_returns, risk_free_rate),
            'Calmar Ratio': calmar_ratio(fund_returns, fund_prices),
            'Omega Ratio': omega_ratio(fund_returns),
            
            # Risque
            'Semi-Variance': semi_variance(fund_returns),
            'Max Drawdown (%)': max_drawdown(fund_prices) * 100,
        }
        
        # Indicateurs relatifs au benchmark
        if bench_returns is not None and bench_prices is not None:
            indicators['Beta'] = beta(fund_returns, bench_returns)
            indicators['Alpha (%)'] = alpha(fund_returns, bench_returns, risk_free_rate) * 100
            indicators['Tracking Error (%)'] = tracking_error(fund_returns, bench_returns) * 100
            indicators['Information Ratio'] = information_ratio(fund_returns, bench_returns, risk_free_rate)
            indicators['Treynor Ratio'] = treynor_ratio(fund_returns, bench_returns, risk_free_rate)
            
            # Bat Benchmark - nouveau calcul glissant journalier
            nb_1y, obs_1y, pct_1y = bat_benchmark_rolling(fund_prices, bench_prices, 1)
            nb_3y, obs_3y, pct_3y = bat_benchmark_rolling(fund_prices, bench_prices, 3)
            nb_5y, obs_5y, pct_5y = bat_benchmark_rolling(fund_prices, bench_prices, 5)
            
            indicators['Bat Bench 1Y (%)'] = pct_1y
            indicators['Bat Bench 3Y (%)'] = pct_3y
            indicators['Bat Bench 5Y (%)'] = pct_5y
        
        results.append(indicators)
    
    df_results = pd.DataFrame(results)
    
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns
    df_results[numeric_cols] = df_results[numeric_cols].round(4)
    
    return df_results


# ==============================================================================
# SCORE COMPOSITE AVEC PONDERATION
# ==============================================================================

def calculate_composite_score(
    indicators_df: pd.DataFrame,
    weights: dict = None
) -> pd.DataFrame:
    """
    Calcule le score composite base sur les z-scores ponderes.
    """
    if weights is None:
        weights = {
            'Performance': 0.25,
            'Risque': 0.25,
            'Benchmark': 0.25,
            'ESG': 0.15,
            'Frais': 0.10
        }
    
    category_cols = {
        'Performance': {
            'higher_better': ['Rendement Annualise (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
            'lower_better': []
        },
        'Risque': {
            'higher_better': [],
            'lower_better': ['Volatilite (%)', 'Max Drawdown (%)', 'Semi-Variance']
        },
        'Benchmark': {
            'higher_better': ['Alpha (%)', 'Information Ratio', 'Treynor Ratio', 'Bat Bench 1Y (%)'],
            'lower_better': []
        },
        'ESG': {
            'higher_better': ['ESG Score'],
            'lower_better': []
        },
        'Frais': {
            'higher_better': [],
            'lower_better': ['Frais (%)']
        }
    }
    
    result = indicators_df.copy()
    z_scores_by_cat = {}
    
    for cat, cols in category_cols.items():
        cat_z_scores = []
        
        for col in cols['higher_better']:
            if col in result.columns:
                values = pd.to_numeric(result[col], errors='coerce')
                if values.std() > 0:
                    z = (values - values.mean()) / values.std()
                    z = z.clip(-3, 3)
                    cat_z_scores.append(z)
        
        for col in cols['lower_better']:
            if col in result.columns:
                values = pd.to_numeric(result[col], errors='coerce')
                if values.std() > 0:
                    z = (values - values.mean()) / values.std()
                    z = -z
                    z = z.clip(-3, 3)
                    cat_z_scores.append(z)
        
        if cat_z_scores:
            z_scores_by_cat[cat] = pd.concat(cat_z_scores, axis=1).mean(axis=1)
    
    total_weight = 0
    weighted_z = pd.Series(0, index=result.index)
    
    for cat, weight in weights.items():
        if cat in z_scores_by_cat:
            weighted_z += z_scores_by_cat[cat] * weight
            total_weight += weight
    
    if total_weight > 0:
        weighted_z = weighted_z / total_weight
    
    result['Z-Score Moyen'] = weighted_z
    result['Score Composite (0-100)'] = ((weighted_z + 3) / 6 * 100).round(1).clip(0, 100)
    
    return result
