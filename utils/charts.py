# ==============================================================================
# CHARTS - Fonctions de visualisation avec Plotly (VERSION BAR CHARTS)
# ==============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import COLORS, BENCHMARK_COLOR


def plot_normalized_prices(
    prices: pd.DataFrame,
    benchmark_cols: list = None,
    title: str = "Evolution des prix (Base 100)"
) -> go.Figure:
    """
    Graphique des prix normalisés base 100.
    """
    # Normaliser base 100
    normalized = prices / prices.bfill().iloc[0] * 100
    
    fig = go.Figure()
    
    benchmark_cols_lower = [col.lower() for col in (benchmark_cols or [])]
    
    color_idx = 0
    for col in normalized.columns:
        is_benchmark = col.lower() in benchmark_cols_lower
        
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[col],
            mode='lines',
            name=col,
            line=dict(
                color=BENCHMARK_COLOR if is_benchmark else COLORS[color_idx % len(COLORS)],
                width=3 if is_benchmark else 2,
                dash='dash' if is_benchmark else 'solid'
            ),
            hovertemplate=f"{col}<br>Date: %{{x}}<br>Valeur: %{{y:.2f}}<extra></extra>"
        ))
        
        if not is_benchmark:
            color_idx += 1
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Valeur (Base 100)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=500
    )
    
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
    
    return fig


def plot_drawdown(
    prices: pd.DataFrame,
    benchmark_cols: list = None,
    title: str = "Drawdown"
) -> go.Figure:
    """
    Graphique des drawdowns.
    """
    fig = go.Figure()
    
    benchmark_cols_lower = [col.lower() for col in (benchmark_cols or [])]
    
    color_idx = 0
    for col in prices.columns:
        # Calculer le drawdown
        running_max = prices[col].cummax()
        drawdown = (prices[col] - running_max) / running_max * 100
        
        is_benchmark = col.lower() in benchmark_cols_lower
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode='lines',
            name=col,
            fill='tozeroy',
            line=dict(
                color=BENCHMARK_COLOR if is_benchmark else COLORS[color_idx % len(COLORS)],
                width=1
            ),
            hovertemplate=f"{col}<br>Date: %{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>"
        ))
        
        if not is_benchmark:
            color_idx += 1
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_indicator_bar_chart(
    indicators_df: pd.DataFrame,
    indicator_col: str,
    title: str = None,
    color_scale: str = "RdYlGn",
    reverse_colors: bool = False
) -> go.Figure:
    """
    Bar chart horizontal pour comparer un indicateur entre fonds.
    
    Parameters
    ----------
    indicators_df : pd.DataFrame
        DataFrame avec colonne 'Fonds' et l'indicateur
    indicator_col : str
        Nom de la colonne indicateur
    title : str
        Titre du graphique
    color_scale : str
        Palette de couleurs (RdYlGn, Blues, etc.)
    reverse_colors : bool
        Inverser les couleurs (pour les indicateurs où plus petit = mieux)
    """
    if title is None:
        title = indicator_col
    
    # Trier par valeur
    df_sorted = indicators_df.dropna(subset=[indicator_col]).sort_values(indicator_col, ascending=True)
    
    if len(df_sorted) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Pas de données", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Couleurs basées sur les valeurs
    colors = df_sorted[indicator_col]
    if reverse_colors:
        colors = -colors
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_sorted[indicator_col],
        y=df_sorted['Fonds'],
        orientation='h',
        marker=dict(
            color=df_sorted[indicator_col],
            colorscale=color_scale,
            reversescale=reverse_colors,
            showscale=False
        ),
        text=df_sorted[indicator_col].apply(lambda x: f"{x:.2f}"),
        textposition='outside',
        hovertemplate="%{y}<br>" + indicator_col + ": %{x:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title=indicator_col,
        yaxis_title="",
        template="plotly_white",
        height=max(300, len(df_sorted) * 35),
        margin=dict(l=200)  # Marge pour les noms de fonds
    )
    
    return fig


def plot_multi_indicator_bars(
    indicators_df: pd.DataFrame,
    indicator_cols: list,
    title: str = "Comparaison des indicateurs"
) -> go.Figure:
    """
    Bar chart groupé pour plusieurs indicateurs.
    """
    fig = go.Figure()
    
    for i, col in enumerate(indicator_cols):
        if col not in indicators_df.columns:
            continue
        fig.add_trace(go.Bar(
            name=col,
            x=indicators_df['Fonds'],
            y=indicators_df[col],
            marker_color=COLORS[i % len(COLORS)],
            text=indicators_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
            textposition='outside'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        barmode='group',
        xaxis_title="Fonds",
        yaxis_title="Valeur",
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_tickangle=-45
    )
    
    return fig


def plot_risk_return_scatter(
    indicators_df: pd.DataFrame,
    x_col: str = "Volatilite (%)",
    y_col: str = "Rendement Annualise (%)",
    title: str = "Rendement vs Risque"
) -> go.Figure:
    """
    Scatter plot rendement/risque.
    """
    fig = go.Figure()
    
    for idx, row in indicators_df.iterrows():
        if pd.isna(row[x_col]) or pd.isna(row[y_col]):
            continue
        fig.add_trace(go.Scatter(
            x=[row[x_col]],
            y=[row[y_col]],
            mode='markers+text',
            name=row['Fonds'],
            text=[row['Fonds']],
            textposition='top center',
            marker=dict(
                size=15,
                color=COLORS[idx % len(COLORS)],
                line=dict(width=2, color='white')
            ),
            hovertemplate=(
                f"<b>{row['Fonds']}</b><br>"
                f"{x_col}: %{{x:.2f}}<br>"
                f"{y_col}: %{{y:.2f}}<br>"
                f"<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=False,
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_rolling_sharpe(
    returns: pd.DataFrame,
    window: int = 63,
    risk_free_rate: float = 0.03,
    benchmark_cols: list = None,
    title: str = "Sharpe Ratio Glissant (3 mois)"
) -> go.Figure:
    """
    Graphique du Sharpe ratio glissant.
    """
    from .indicators import calculate_rolling_sharpe
    
    fig = go.Figure()
    
    benchmark_cols_lower = [col.lower() for col in (benchmark_cols or [])]
    
    color_idx = 0
    for col in returns.columns:
        rolling_sharpe = calculate_rolling_sharpe(
            returns[col], 
            window=window, 
            risk_free_rate=risk_free_rate
        )
        
        is_benchmark = col.lower() in benchmark_cols_lower
        
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            mode='lines',
            name=col,
            line=dict(
                color=BENCHMARK_COLOR if is_benchmark else COLORS[color_idx % len(COLORS)],
                width=2
            ),
            hovertemplate=f"{col}<br>Date: %{{x}}<br>Sharpe: %{{y:.2f}}<extra></extra>"
        ))
        
        if not is_benchmark:
            color_idx += 1
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=400
    )
    
    fig.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
    fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.5)
    
    return fig


def plot_benchmark_comparison(
    returns: pd.DataFrame,
    benchmark_col: str,
    title: str = "Surperformance vs Benchmark"
) -> go.Figure:
    """
    Graphique de la surperformance cumulée par rapport au benchmark.
    """
    if benchmark_col not in returns.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    benchmark_ret = returns[benchmark_col]
    
    color_idx = 0
    for col in returns.columns:
        if col.lower() == benchmark_col.lower():
            continue
        
        # Surperformance = rendement fonds - rendement benchmark
        outperformance = returns[col] - benchmark_ret
        cumulative_outperf = (1 + outperformance).cumprod() - 1
        cumulative_outperf = cumulative_outperf * 100
        
        fig.add_trace(go.Scatter(
            x=cumulative_outperf.index,
            y=cumulative_outperf,
            mode='lines',
            name=col,
            fill='tozeroy',
            line=dict(
                color=COLORS[color_idx % len(COLORS)],
                width=2
            ),
            hovertemplate=f"{col}<br>Date: %{{x}}<br>Surperf: %{{y:.2f}}%<extra></extra>"
        ))
        color_idx += 1
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Surperformance cumulée (%)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=400
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
    
    return fig


def plot_correlation_matrix(
    returns: pd.DataFrame,
    title: str = "Matrice de Corrélation"
) -> go.Figure:
    """
    Heatmap des corrélations entre fonds.
    """
    corr_matrix = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Corr: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        template="plotly_white",
        height=500,
        width=700
    )
    
    return fig