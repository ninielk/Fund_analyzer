# ==============================================================================
# DASHBOARD ANALYSE FONDS LARGE CAP
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration de la page (DOIT ETRE EN PREMIER)
st.set_page_config(
    page_title="Analyse Fonds Large Cap",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports locaux
from config import TRADING_DAYS_PER_YEAR
from utils.data_loader import (
    load_data, 
    get_valid_date_range,
    filter_data_by_period,
    calculate_returns,
    get_fund_inception_dates
)
from utils.indicators import calculate_all_indicators
from utils.charts import (
    plot_normalized_prices,
    plot_drawdown,
    plot_risk_return_scatter,
    plot_rolling_sharpe,
    plot_benchmark_comparison,
    plot_indicator_bar_chart,
    plot_multi_indicator_bars,
    plot_correlation_matrix
)


# ==============================================================================
# STYLE CSS
# ==============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .inception-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HEADER
# ==============================================================================

st.markdown('<p class="main-header">Analyse de Fonds Large Cap</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Outil d\'analyse de performance et de risque</p>', unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR - CHARGEMENT DES DONNEES
# ==============================================================================

st.sidebar.header("Charger les donnees")

uploaded_file = st.sidebar.file_uploader(
    "Fichier Excel ou CSV",
    type=['xlsx', 'xls', 'csv'],
    help="Format: Date en premiere colonne, puis une colonne par fonds/benchmark/taux"
)

# Charger les données
df_prices = None

if uploaded_file is not None:
    try:
        df_prices = load_data(uploaded_file=uploaded_file)
        st.sidebar.success(f"{uploaded_file.name} charge ({len(df_prices)} observations)")
    except Exception as e:
        st.sidebar.error(f"Erreur: {str(e)}")

if df_prices is None:
    st.warning("""
    ### Bienvenue
    
    Chargez votre fichier Excel via la sidebar pour commencer.
    
    **Format attendu:**
    - Colonne Date
    - Colonnes avec les cours/NAV de chaque fonds
    - Colonne(s) benchmark (ex: MSCI Europe Large Cap)
    - Colonne taux sans risque (ex: EONIA, Euribor)
    """)
    st.stop()


# ==============================================================================
# SIDEBAR - CONFIGURATION
# ==============================================================================

st.sidebar.divider()
st.sidebar.header("Configuration")

all_columns = list(df_prices.columns)

# --- Sélection du taux sans risque ---
st.sidebar.subheader("Taux sans risque")
rf_col = st.sidebar.selectbox(
    "Colonne taux sans risque",
    options=["Aucun (manuel)"] + all_columns,
    index=0,
    help="Selectionne la colonne contenant le taux sans risque (ex: EONIA, Euribor)"
)

if rf_col == "Aucun (manuel)":
    risk_free_rate = st.sidebar.slider(
        "Taux sans risque (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1
    ) / 100
    rf_column_selected = None
else:
    rf_column_selected = rf_col
    # Calculer le taux moyen annualisé à partir de la série
    rf_returns = df_prices[rf_col].pct_change().dropna()
    if len(rf_returns) > 0:
        # Annualiser le rendement moyen quotidien
        avg_daily = rf_returns.mean()
        risk_free_rate = (1 + avg_daily) ** TRADING_DAYS_PER_YEAR - 1
        st.sidebar.info(f"Taux annualise: {risk_free_rate*100:.2f}%")
    else:
        risk_free_rate = 0.03

# --- Sélection des benchmarks ---
st.sidebar.subheader("Benchmark(s)")

# Colonnes restantes après exclusion du taux sans risque
available_for_bench = [c for c in all_columns if c != rf_column_selected]

benchmark_cols = st.sidebar.multiselect(
    "Colonne(s) benchmark",
    options=available_for_bench,
    default=[],
    help="Selectionne une ou plusieurs colonnes benchmark"
)

# --- Sélection des fonds ---
st.sidebar.divider()
st.sidebar.header("Selection des Fonds")

# Colonnes fonds = tout sauf benchmarks et taux sans risque
excluded_cols = benchmark_cols + ([rf_column_selected] if rf_column_selected else [])
fund_columns = [c for c in all_columns if c not in excluded_cols]

selected_funds = st.sidebar.multiselect(
    "Fonds a analyser",
    options=fund_columns,
    default=fund_columns[:5] if len(fund_columns) > 5 else fund_columns
)

if not selected_funds:
    st.warning("Selectionnez au moins un fonds.")
    st.stop()

# --- Afficher les dates d'inception ---
st.sidebar.divider()
st.sidebar.subheader("Dates d'inception")

inception_df = get_fund_inception_dates(df_prices[selected_funds])
for _, row in inception_df.iterrows():
    date_str = row['Date Inception'].strftime('%d/%m/%Y') if pd.notna(row['Date Inception']) else "N/A"
    col_name = row['Colonne'][:30] + "..." if len(row['Colonne']) > 30 else row['Colonne']
    st.sidebar.markdown(
        f"<div class='inception-box'><b>{col_name}</b><br>{date_str}</div>",
        unsafe_allow_html=True
    )

# --- Période d'analyse ---
st.sidebar.divider()
st.sidebar.header("Periode d'Analyse")

date_min, date_max = get_valid_date_range(df_prices, selected_funds)
st.sidebar.caption(f"Disponible: {date_min.strftime('%d/%m/%Y')} - {date_max.strftime('%d/%m/%Y')}")

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Debut", value=date_min, min_value=date_min.date(), max_value=date_max.date())
with col2:
    end_date = st.date_input("Fin", value=date_max, min_value=date_min.date(), max_value=date_max.date())

start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

if start_date >= end_date:
    st.error("Date de debut >= date de fin")
    st.stop()


# ==============================================================================
# FILTRAGE DES DONNEES
# ==============================================================================

# IMPORTANT: Ne garder que les fonds sélectionnés + benchmarks pour les calculs
# Le taux sans risque est utilisé pour le calcul mais PAS inclus dans le tableau des fonds
cols_to_keep = list(selected_funds) + benchmark_cols

df_filtered = filter_data_by_period(df_prices, start_date, end_date, cols_to_keep)
df_returns = calculate_returns(df_filtered)

df_filtered = df_filtered.dropna(how='all')
df_returns = df_returns.dropna(how='all')


# ==============================================================================
# CALCUL DES INDICATEURS
# ==============================================================================

indicators_df = calculate_all_indicators(
    prices=df_filtered,
    returns=df_returns,
    benchmark_cols=benchmark_cols,
    risk_free_rate=risk_free_rate
)


# ==============================================================================
# TABS PRINCIPAUX
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Vue Globale",
    "Performance",
    "Risque",
    "vs Benchmark"
])


# ==============================================================================
# TAB 1: VUE GLOBALE
# ==============================================================================

with tab1:
    st.header("Vue Globale des Indicateurs")
    
    # Résumé rapide
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nb Fonds", len(selected_funds))
    with col2:
        avg_sharpe = indicators_df['Sharpe Ratio'].mean()
        st.metric("Sharpe Moyen", f"{avg_sharpe:.2f}")
    with col3:
        best_return = indicators_df['Rendement Annualise (%)'].max()
        st.metric("Meilleur Rdt", f"{best_return:.1f}%")
    with col4:
        worst_dd = indicators_df['Max Drawdown (%)'].min()
        st.metric("Pire Drawdown", f"{worst_dd:.1f}%")
    
    st.divider()
    
    # Tableau des indicateurs
    st.subheader("Tableau Recapitulatif")
    
    col_categories = {
        "Rendements": ['Rendement Annualise (%)', 'Rendement 1Y (%)', 'Rendement 3Y (%)', 'Rendement 5Y (%)', 'Rendement Cumule (%)'],
        "Volatilite": ['Volatilite (%)', 'Volatilite 1Y (%)', 'Volatilite 3Y (%)', 'Volatilite 5Y (%)'],
        "Ratios": ['Sharpe Ratio', 'Sharpe 1Y', 'Sharpe 3Y', 'Sharpe 5Y', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio'],
        "Risque": ['Max Drawdown (%)', 'Semi-Variance', 'Beta'],
        "Benchmark": ['Alpha (%)', 'Nb Fois Bat Benchmark', 'Total Observations']
    }
    
    selected_categories = st.multiselect(
        "Categories a afficher",
        options=list(col_categories.keys()),
        default=["Rendements", "Ratios", "Benchmark"]
    )
    
    cols_to_show = ['Fonds']
    for cat in selected_categories:
        cols_to_show.extend([c for c in col_categories[cat] if c in indicators_df.columns])
    
    df_display = indicators_df[[c for c in cols_to_show if c in indicators_df.columns]].copy()
    
    # Style
    def highlight_best(s):
        if s.name in ['Sharpe Ratio', 'Sharpe 1Y', 'Sharpe 3Y', 'Sharpe 5Y', 'Sortino Ratio',
                      'Rendement Annualise (%)', 'Rendement 1Y (%)', 'Rendement 3Y (%)', 
                      'Rendement 5Y (%)', 'Rendement Cumule (%)', 'Alpha (%)', 
                      'Calmar Ratio', 'Omega Ratio', 'Nb Fois Bat Benchmark']:
            is_max = s == s.max()
            return ['background-color: #90EE90' if v else '' for v in is_max]
        elif s.name in ['Max Drawdown (%)', 'Volatilite (%)', 'Volatilite 1Y (%)',
                        'Volatilite 3Y (%)', 'Volatilite 5Y (%)', 'Semi-Variance']:
            is_min = s == s.min()
            return ['background-color: #90EE90' if v else '' for v in is_min]
        return ['' for _ in s]
    
    format_dict = {}
    for col in df_display.columns:
        if col == 'Fonds':
            continue
        elif any(x in col for x in ['Sharpe', 'Sortino', 'Calmar', 'Omega', 'Beta']):
            format_dict[col] = '{:.3f}'
        elif 'Semi-Variance' in col:
            format_dict[col] = '{:.6f}'
        elif col in ['Nb Fois Bat Benchmark', 'Total Observations']:
            format_dict[col] = '{:.0f}'
        elif '%' in col:
            format_dict[col] = '{:.2f}'
    
    styled_df = df_display.style.apply(highlight_best).format(format_dict, na_rep="-")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Download
    csv = indicators_df.to_csv(index=False)
    st.download_button("Telecharger CSV", data=csv, 
                       file_name=f"indicateurs_{datetime.now().strftime('%Y%m%d')}.csv", 
                       mime="text/csv")


# ==============================================================================
# TAB 2: PERFORMANCE
# ==============================================================================

with tab2:
    st.header("Analyse de Performance")
    
    # Prix normalisés
    st.subheader("Evolution des Prix (Base 100)")
    cols_for_chart = selected_funds + benchmark_cols
    fig_prices = plot_normalized_prices(df_filtered[cols_for_chart], benchmark_cols=benchmark_cols, title="")
    st.plotly_chart(fig_prices, use_container_width=True)
    
    st.divider()
    
    # Bar charts rendements
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rendement Annualise")
        fig_ret = plot_indicator_bar_chart(indicators_df, "Rendement Annualise (%)", title="")
        st.plotly_chart(fig_ret, use_container_width=True)
    
    with col2:
        st.subheader("Sharpe Ratio")
        fig_sharpe = plot_indicator_bar_chart(indicators_df, "Sharpe Ratio", title="")
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    st.divider()
    
    # Rendement vs Risque scatter
    st.subheader("Rendement vs Risque")
    fig_scatter = plot_risk_return_scatter(indicators_df)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# TAB 3: RISQUE
# ==============================================================================

with tab3:
    st.header("Analyse de Risque")
    
    # Drawdown
    st.subheader("Drawdown")
    fig_dd = plot_drawdown(df_filtered[selected_funds], title="")
    st.plotly_chart(fig_dd, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Max Drawdown")
        fig_mdd = plot_indicator_bar_chart(indicators_df, "Max Drawdown (%)", title="", reverse_colors=True)
        st.plotly_chart(fig_mdd, use_container_width=True)
    
    with col2:
        st.subheader("Volatilite")
        fig_vol = plot_indicator_bar_chart(indicators_df, "Volatilite (%)", title="", reverse_colors=True)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    st.divider()
    
    # Corrélation
    st.subheader("Matrice de Correlation")
    fig_corr = plot_correlation_matrix(df_returns[selected_funds], title="")
    st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# TAB 4: BENCHMARK COMPARISON
# ==============================================================================

with tab4:
    st.header("Comparaison avec le(s) Benchmark(s)")
    
    if not benchmark_cols:
        st.warning("Aucun benchmark selectionne. Retournez dans la sidebar pour en choisir un.")
    else:
        # Sélectionner le benchmark à afficher
        if len(benchmark_cols) > 1:
            selected_bench = st.selectbox("Benchmark a afficher", benchmark_cols)
        else:
            selected_bench = benchmark_cols[0]
        
        st.caption(f"Benchmark: {selected_bench}")
        
        # Surperformance
        st.subheader("Surperformance Cumulee")
        fig_outperf = plot_benchmark_comparison(df_returns, benchmark_col=selected_bench, title="")
        st.plotly_chart(fig_outperf, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Alpha")
            if 'Alpha (%)' in indicators_df.columns:
                fig_alpha = plot_indicator_bar_chart(indicators_df, "Alpha (%)", title="")
                st.plotly_chart(fig_alpha, use_container_width=True)
        
        with col2:
            st.subheader("Nb Fois Bat le Benchmark")
            if 'Nb Fois Bat Benchmark' in indicators_df.columns:
                fig_beats = plot_indicator_bar_chart(indicators_df, "Nb Fois Bat Benchmark", title="")
                st.plotly_chart(fig_beats, use_container_width=True)
        
        st.divider()
        
        # Beta
        st.subheader("Beta")
        if 'Beta' in indicators_df.columns:
            fig_beta = plot_indicator_bar_chart(indicators_df, "Beta", title="")
            st.plotly_chart(fig_beta, use_container_width=True)


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.caption(f"""
Dashboard Analyse Fonds Large Cap | 
Periode: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')} | 
Taux sans risque: {risk_free_rate*100:.2f}% | 
Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}
""")



