# ==============================================================================
# CHARTS - Fonctions de visualisation V2 FINAL
# ==============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import COLORS, BENCHMARK_COLOR, RADAR_FILL_COLOR, RADAR_LINE_COLOR


def plot_normalized_prices(
    prices: pd.DataFrame,
    benchmark_cols: list = None,
    title: str = "Evolution des prix (Base 100)"
) -> go.Figure:
    """Graphique des prix normalises base 100."""
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
        title=dict(text=title, x=0.5, font=dict(size=18, color="#412761")),
        xaxis_title="Date",
        yaxis_title="Valeur (Base 100)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500
    )
    
    fig.add_hline(y=100, line_dash="dot", line_color="#6E6E6E", opacity=0.5)
    
    return fig


def plot_drawdown(
    prices: pd.DataFrame,
    benchmark_cols: list = None,
    title: str = "Drawdown"
) -> go.Figure:
    """Graphique des drawdowns."""
    fig = go.Figure()
    
    benchmark_cols_lower = [col.lower() for col in (benchmark_cols or [])]
    
    color_idx = 0
    for col in prices.columns:
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
        title=dict(text=title, x=0.5, font=dict(size=18, color="#412761")),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_indicator_bar_chart(
    indicators_df: pd.DataFrame,
    indicator_col: str,
    title: str = None,
    reverse_colors: bool = False
) -> go.Figure:
    """Bar chart horizontal pour comparer un indicateur entre fonds."""
    if title is None:
        title = indicator_col
    
    df_sorted = indicators_df.dropna(subset=[indicator_col]).sort_values(indicator_col, ascending=True)
    
    if len(df_sorted) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Pas de donnees", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    n = len(df_sorted)
    if reverse_colors:
        colors = [COLORS[i % len(COLORS)] for i in range(n)]
    else:
        colors = [COLORS[(n - 1 - i) % len(COLORS)] for i in range(n)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_sorted[indicator_col],
        y=df_sorted['Fonds'],
        orientation='h',
        marker=dict(color=colors),
        text=df_sorted[indicator_col].apply(lambda x: f"{x:.2f}"),
        textposition='outside',
        hovertemplate="%{y}<br>" + indicator_col + ": %{x:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, color="#412761")),
        xaxis_title=indicator_col,
        yaxis_title="",
        template="plotly_white",
        height=max(300, len(df_sorted) * 35),
        margin=dict(l=200)
    )
    
    return fig


def plot_risk_return_scatter(
    indicators_df: pd.DataFrame,
    x_col: str = "Volatilite (%)",
    y_col: str = "Rendement Annualise (%)",
    title: str = "Rendement vs Risque"
) -> go.Figure:
    """Scatter plot rendement/risque."""
    fig = go.Figure()
    
    for idx, row in indicators_df.iterrows():
        if pd.isna(row[x_col]) or pd.isna(row[y_col]):
            continue
        fig.add_trace(go.Scatter(
            x=[row[x_col]],
            y=[row[y_col]],
            mode='markers+text',
            name=row['Fonds'],
            text=[row['Fonds'][:15]],
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
        title=dict(text=title, x=0.5, font=dict(size=18, color="#412761")),
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=False,
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_radar_chart(
    indicators_df: pd.DataFrame,
    fund_name: str,
    metrics: list = None,
    title: str = None
) -> go.Figure:
    """Radar chart pour visualiser le profil d'un seul fonds."""
    if metrics is None:
        metrics = [
            'Rendement Annualise (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Alpha (%)',
            'ESG Score'
        ]
        metrics_lower = ['Volatilite (%)', 'Max Drawdown (%)', 'Frais (%)']
    else:
        metrics_lower = []
    
    if title is None:
        title = f"Profil du fonds: {fund_name}"
    
    fund_row = indicators_df[indicators_df['Fonds'] == fund_name]
    if len(fund_row) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Fonds non trouve", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    fund_row = fund_row.iloc[0]
    
    values = []
    labels = []
    
    all_metrics = metrics + metrics_lower
    
    for metric in all_metrics:
        if metric in indicators_df.columns:
            col_values = pd.to_numeric(indicators_df[metric], errors='coerce')
            val = fund_row[metric]
            
            if pd.notna(val) and col_values.std() > 0:
                min_val = col_values.min()
                max_val = col_values.max()
                
                if metric in metrics_lower:
                    normalized = 100 - ((val - min_val) / (max_val - min_val) * 100) if max_val != min_val else 50
                else:
                    normalized = ((val - min_val) / (max_val - min_val) * 100) if max_val != min_val else 50
                
                values.append(normalized)
                labels.append(metric.replace(' (%)', '').replace(' Ratio', ''))
    
    if len(values) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Pas assez de donnees", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    values.append(values[0])
    labels.append(labels[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        fillcolor=RADAR_FILL_COLOR,
        line=dict(color=RADAR_LINE_COLOR, width=2),
        name=fund_name
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#412761")
            )
        ),
        showlegend=False,
        title=dict(text=title, x=0.5, font=dict(size=16, color="#412761")),
        height=450
    )
    
    return fig


def plot_radar_multi_funds(
    indicators_df: pd.DataFrame,
    fund_names: list,
    metrics: list = None,
    title: str = "Comparaison des fonds"
) -> go.Figure:
    """
    Radar chart comparant plusieurs fonds selectionnes.
    """
    if metrics is None:
        metrics = ['Rendement Annualise (%)', 'Sharpe Ratio', 'Alpha (%)', 'ESG Score', 'Omega vs Benchmark']
    
    # Filtrer les metriques disponibles
    available_metrics = [m for m in metrics if m in indicators_df.columns and indicators_df[m].notna().sum() > 0]
    
    if len(available_metrics) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Pas assez de metriques disponibles", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Metriques ou lower = better
    lower_is_better = ['Volatilite (%)', 'Max Drawdown (%)', 'Frais (%)', 'Semi-Variance', 'Tracking Error (%)']
    
    fig = go.Figure()
    
    for i, fund_name in enumerate(fund_names):
        fund_row = indicators_df[indicators_df['Fonds'] == fund_name]
        if len(fund_row) == 0:
            continue
        fund_row = fund_row.iloc[0]
        
        values = []
        labels = []
        
        for metric in available_metrics:
            col_values = pd.to_numeric(indicators_df[metric], errors='coerce')
            val = fund_row[metric]
            
            if pd.notna(val) and col_values.std() > 0:
                min_val = col_values.min()
                max_val = col_values.max()
                
                # Inverser pour les metriques ou lower = better
                if metric in lower_is_better:
                    normalized = 100 - ((val - min_val) / (max_val - min_val) * 100) if max_val != min_val else 50
                else:
                    normalized = ((val - min_val) / (max_val - min_val) * 100) if max_val != min_val else 50
                
                values.append(normalized)
                # Raccourcir les noms
                label = metric.replace(' (%)', '').replace(' Ratio', '').replace('Annualise', 'Ann.').replace('Benchmark', 'Bench')
                labels.append(label)
        
        if len(values) > 0:
            # Fermer le radar
            values.append(values[0])
            labels.append(labels[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=fund_name[:20],
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                opacity=0.7
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9)
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="#412761")
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        title=dict(text=title, x=0.5, font=dict(size=16, color="#412761")),
        height=550
    )
    
    return fig


def plot_radar_comparison(
    indicators_df: pd.DataFrame,
    fund_names: list,
    metrics: list = None,
    title: str = "Comparaison des fonds"
) -> go.Figure:
    """Alias pour plot_radar_multi_funds."""
    return plot_radar_multi_funds(indicators_df, fund_names, metrics, title)


def plot_benchmark_comparison(
    returns: pd.DataFrame,
    benchmark_col: str,
    title: str = "Surperformance vs Benchmark"
) -> go.Figure:
    """Graphique de la surperformance cumulee."""
    if benchmark_col not in returns.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    benchmark_ret = returns[benchmark_col]
    
    color_idx = 0
    for col in returns.columns:
        if col.lower() == benchmark_col.lower():
            continue
        
        outperformance = returns[col] - benchmark_ret
        cumulative_outperf = (1 + outperformance).cumprod() - 1
        cumulative_outperf = cumulative_outperf * 100
        
        fig.add_trace(go.Scatter(
            x=cumulative_outperf.index,
            y=cumulative_outperf,
            mode='lines',
            name=col,
            fill='tozeroy',
            line=dict(color=COLORS[color_idx % len(COLORS)], width=2),
            hovertemplate=f"{col}<br>Date: %{{x}}<br>Surperf: %{{y:.2f}}%<extra></extra>"
        ))
        color_idx += 1
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, color="#412761")),
        xaxis_title="Date",
        yaxis_title="Surperformance cumulee (%)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=400
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
    
    return fig


def plot_correlation_matrix(
    returns: pd.DataFrame,
    title: str = "Matrice de Correlation"
) -> go.Figure:
    """Heatmap des correlations entre fonds."""
    corr_matrix = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[c[:15] for c in corr_matrix.columns],
        y=[c[:15] for c in corr_matrix.index],
        colorscale=[[0, "#B10967"], [0.5, "#FFFFFF"], [1, "#007078"]],
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 9},
        hovertemplate='%{x} vs %{y}<br>Corr: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, color="#412761")),
        template="plotly_white",
        height=500,
        width=700
    )
    
    return fig


def plot_score_composite_bar(
    indicators_df: pd.DataFrame,
    title: str = "Score Composite (0-100)"
) -> go.Figure:
    """Bar chart du score composite."""
    if 'Score Composite (0-100)' not in indicators_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Score non calcule", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    df_sorted = indicators_df.dropna(subset=['Score Composite (0-100)']).sort_values(
        'Score Composite (0-100)', ascending=True
    )
    
    colors = []
    for score in df_sorted['Score Composite (0-100)']:
        if score >= 70:
            colors.append("#007078")
        elif score >= 50:
            colors.append("#F8AF00")
        else:
            colors.append("#B10967")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_sorted['Score Composite (0-100)'],
        y=df_sorted['Fonds'],
        orientation='h',
        marker=dict(color=colors),
        text=df_sorted['Score Composite (0-100)'].apply(lambda x: f"{x:.1f}"),
        textposition='outside',
        hovertemplate="%{y}<br>Score: %{x:.1f}/100<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, color="#412761")),
        xaxis_title="Score (0-100)",
        xaxis=dict(range=[0, 105]),
        yaxis_title="",
        template="plotly_white",
        height=max(300, len(df_sorted) * 40),
        margin=dict(l=200)
    )
    
    return fig