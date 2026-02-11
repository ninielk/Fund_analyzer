# ==============================================================================
# CHARTS - Fonctions de visualisation avec Plotly
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
    benchmark_col: str = None,
    title: str = "Evolution des prix (Base 100)"
) -> go.Figure:
    """
    Graphique des prix normalisés base 100.
    
    Permet de comparer visuellement la performance de plusieurs fonds
    partant tous du même point.
    """
    # Normaliser base 100
    normalized = prices / prices.iloc[0] * 100
    
    fig = go.Figure()
    
    color_idx = 0
    for col in normalized.columns:
        is_benchmark = benchmark_col and col.lower() == benchmark_col.lower()
        
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
    
    # Ligne horizontale à 100
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
    
    return fig


def plot_drawdown(
    prices: pd.DataFrame,
    benchmark_col: str = None,
    title: str = "Drawdown"
) -> go.Figure:
    """
    Graphique des drawdowns (pertes depuis le plus haut).
    
    Montre quand et de combien chaque fonds a chuté par rapport
    à son sommet historique.
    """
    fig = go.Figure()
    
    color_idx = 0
    for col in prices.columns:
        # Calculer le drawdown
        running_max = prices[col].cummax()
        drawdown = (prices[col] - running_max) / running_max * 100
        
        is_benchmark = benchmark_col and col.lower() == benchmark_col.lower()
        
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


def plot_risk_return_scatter(
    indicators_df: pd.DataFrame,
    x_col: str = "Volatilite (%)",
    y_col: str = "Rendement Annualise (%)",
    size_col: str = None,
    title: str = "Rendement vs Risque"
) -> go.Figure:
    """
    Scatter plot rendement/risque.
    
    Chaque point est un fonds. Idéalement on veut être en haut à gauche
    (haut rendement, faible risque).
    """
    fig = go.Figure()
    
    for idx, row in indicators_df.iterrows():
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
    
    # Ajouter une ligne de référence (ratio Sharpe = 1 par exemple)
    x_range = indicators_df[x_col].max() - indicators_df[x_col].min()
    x_vals = np.linspace(0, indicators_df[x_col].max() + x_range * 0.1, 50)
    
    return fig


def plot_rolling_sharpe(
    returns: pd.DataFrame,
    window: int = 63,
    risk_free_rate: float = 0.03,
    benchmark_col: str = None,
    title: str = "Sharpe Ratio Glissant (3 mois)"
) -> go.Figure:
    """
    Graphique du Sharpe ratio glissant dans le temps.
    
    Permet de voir si la performance ajustée du risque est stable
    ou fluctuante.
    """
    from .indicators import calculate_rolling_sharpe
    
    fig = go.Figure()
    
    color_idx = 0
    for col in returns.columns:
        rolling_sharpe = calculate_rolling_sharpe(
            returns[col], 
            window=window, 
            risk_free_rate=risk_free_rate
        )
        
        is_benchmark = benchmark_col and col.lower() == benchmark_col.lower()
        
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
    
    # Lignes de référence
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
    
    Montre si le fonds fait mieux ou moins bien que le benchmark
    au fil du temps.
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
        cumulative_outperf = cumulative_outperf * 100  # En %
        
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
    
    # Ligne à 0
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
    
    return fig


def plot_indicator_comparison(
    indicators_df: pd.DataFrame,
    indicator_col: str,
    title: str = None
) -> go.Figure:
    """
    Bar chart comparant un indicateur entre tous les fonds.
    """
    if title is None:
        title = f"Comparaison: {indicator_col}"
    
    # Trier par valeur décroissante
    df_sorted = indicators_df.sort_values(indicator_col, ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_sorted[indicator_col],
        y=df_sorted['Fonds'],
        orientation='h',
        marker=dict(
            color=df_sorted[indicator_col],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title=indicator_col)
        ),
        hovertemplate="%{y}<br>" + indicator_col + ": %{x:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title=indicator_col,
        yaxis_title="",
        template="plotly_white",
        height=max(300, len(df_sorted) * 40)
    )
    
    return fig


def plot_correlation_matrix(
    returns: pd.DataFrame,
    title: str = "Matrice de Corrélation"
) -> go.Figure:
    """
    Heatmap des corrélations entre fonds.
    
    Permet de voir quels fonds bougent ensemble (diversification).
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
        width=600
    )
    
    return fig


def create_summary_table(indicators_df: pd.DataFrame) -> go.Figure:
    """
    Crée un tableau formaté des indicateurs.
    """
    # Formater les colonnes
    df_display = indicators_df.copy()
    
    # Arrondir
    for col in df_display.columns:
        if col != 'Fonds' and df_display[col].dtype in ['float64', 'float32']:
            df_display[col] = df_display[col].round(2)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_display.columns),
            fill_color='#1f77b4',
            font=dict(color='white', size=12),
            align='center',
            height=40
        ),
        cells=dict(
            values=[df_display[col] for col in df_display.columns],
            fill_color=[['#f9f9f9', '#ffffff'] * (len(df_display) // 2 + 1)],
            align='center',
            height=35,
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=dict(text="Tableau Récapitulatif des Indicateurs", x=0.5),
        height=min(600, 100 + len(df_display) * 40)
    )
    
    return fig
