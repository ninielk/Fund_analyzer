# ==============================================================================
# DASHBOARD ANALYSE FONDS LARGE CAP - V2 FINAL
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Analyse Fonds Large Cap",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import TRADING_DAYS_PER_YEAR, COLORS
from utils.data_loader import (
    load_data,
    load_fund_metadata,
    match_fund_metadata,
    get_valid_date_range,
    filter_data_by_period,
    calculate_returns,
    get_fund_inception_dates
)
from utils.indicators import calculate_all_indicators
from utils.charts import (
    plot_normalized_prices,
    plot_drawdown,
    plot_indicator_bar_chart,
    plot_risk_return_scatter,
    plot_radar_chart,
    plot_radar_multi_funds,
    plot_benchmark_comparison,
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
        color: #412761;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6E6E6E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .inception-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #B10967;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HEADER
# ==============================================================================

st.markdown('<p class="main-header">Analyse de Fonds Large Cap</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Outil d\'analyse de performance et de risque - V2</p>', unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR - CHARGEMENT DES DONNEES
# ==============================================================================

st.sidebar.header("Charger les donnees")

uploaded_file = st.sidebar.file_uploader(
    "Fichier Excel (2 feuilles)",
    type=['xlsx', 'xls', 'csv'],
    help="Feuille 1: Prix/NAV | Feuille 2: ESG et Frais"
)

df_prices = None
metadata_df = None

if uploaded_file is not None:
    try:
        df_prices = load_data(uploaded_file=uploaded_file)
        st.sidebar.success(f"{uploaded_file.name} charge ({len(df_prices)} obs)")
        
        metadata_df = load_fund_metadata(uploaded_file, sheet_name=1)
        if metadata_df is not None:
            st.sidebar.info(f"Metadonnees chargees")
            
    except Exception as e:
        st.sidebar.error(f"Erreur: {str(e)}")

if df_prices is None:
    st.warning("""
    ### Bienvenue
    
    Chargez votre fichier Excel via la sidebar.
    
    **Format attendu:**
    - **Feuille 1**: Date + colonnes prix/NAV des fonds et benchmarks
    - **Feuille 2**: Metriques (ESG, Frais) par fonds
    """)
    st.stop()


# ==============================================================================
# SIDEBAR - CONFIGURATION
# ==============================================================================

st.sidebar.divider()
st.sidebar.header("Configuration")

all_columns = list(df_prices.columns)

# Taux sans risque
st.sidebar.subheader("Taux sans risque")
rf_col = st.sidebar.selectbox(
    "Colonne taux sans risque",
    options=["Manuel"] + all_columns,
    index=0
)

if rf_col == "Manuel":
    risk_free_rate = st.sidebar.slider("Taux (%)", 0.0, 10.0, 3.0, 0.1) / 100
    rf_column_selected = None
else:
    rf_column_selected = rf_col
    rf_returns = df_prices[rf_col].pct_change().dropna()
    if len(rf_returns) > 0:
        avg_daily = rf_returns.mean()
        risk_free_rate = (1 + avg_daily) ** TRADING_DAYS_PER_YEAR - 1
        st.sidebar.info(f"Taux annualise: {risk_free_rate*100:.2f}%")
    else:
        risk_free_rate = 0.03

# Benchmarks
st.sidebar.subheader("Benchmark(s)")
available_for_bench = [c for c in all_columns if c != rf_column_selected]
benchmark_cols = st.sidebar.multiselect("Colonne(s) benchmark", options=available_for_bench, default=[])

# Fonds
st.sidebar.divider()
st.sidebar.header("Selection des Fonds")

excluded_cols = benchmark_cols + ([rf_column_selected] if rf_column_selected else [])
fund_columns = [c for c in all_columns if c not in excluded_cols]

selected_funds = st.sidebar.multiselect(
    "Fonds a analyser",
    options=fund_columns,
    default=fund_columns[:6] if len(fund_columns) > 6 else fund_columns
)

if not selected_funds:
    st.warning("Selectionnez au moins un fonds.")
    st.stop()

# Dates d'inception
st.sidebar.divider()
st.sidebar.subheader("Dates de démarrage")

inception_df = get_fund_inception_dates(df_prices[selected_funds])
for _, row in inception_df.iterrows():
    date_str = row['Date Inception'].strftime('%d/%m/%Y') if pd.notna(row['Date Inception']) else "N/A"
    col_name = row['Colonne'][:25] + "..." if len(row['Colonne']) > 25 else row['Colonne']
    st.sidebar.markdown(f"<div class='inception-box'><b>{col_name}</b><br>{date_str}</div>", unsafe_allow_html=True)

# Periode
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
# FILTRAGE ET CALCULS
# ==============================================================================

cols_to_keep = list(selected_funds) + benchmark_cols
df_filtered = filter_data_by_period(df_prices, start_date, end_date, cols_to_keep)
df_returns = calculate_returns(df_filtered)

df_filtered = df_filtered.dropna(how='all')
df_returns = df_returns.dropna(how='all')

# Stocker dans session_state
st.session_state['df_filtered'] = df_filtered
st.session_state['df_returns'] = df_returns
st.session_state['benchmark_cols'] = benchmark_cols
st.session_state['selected_funds'] = selected_funds
st.session_state['risk_free_rate'] = risk_free_rate
st.session_state['metadata_df'] = metadata_df


# ==============================================================================
# TABS
# ==============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Vue Globale",
    "Performance",
    "Risque",
    "vs Benchmark",
    "Radar"
])


# ==============================================================================
# TAB 1: VUE GLOBALE
# ==============================================================================

with tab1:
    st.header("Vue Globale des Indicateurs")
    
    # Choix du benchmark pour le calcul
    if len(benchmark_cols) > 1:
        selected_bench_calc = st.selectbox("Benchmark pour les calculs", benchmark_cols, key="bench_tab1")
    elif len(benchmark_cols) == 1:
        selected_bench_calc = benchmark_cols[0]
    else:
        selected_bench_calc = None
        st.warning("Aucun benchmark selectionne - certains indicateurs ne seront pas calcules.")
    
    # Calculer les indicateurs
    if selected_bench_calc:
        cols_for_calc = selected_funds + [selected_bench_calc]
        df_calc = df_filtered[cols_for_calc]
        ret_calc = df_returns[cols_for_calc]
        indicators_df = calculate_all_indicators(df_calc, ret_calc, selected_bench_calc, risk_free_rate)
    else:
        indicators_df = calculate_all_indicators(df_filtered[selected_funds], df_returns[selected_funds], "", risk_free_rate)
    
    # Ajouter ESG et Frais
    if metadata_df is not None:
        indicators_df = match_fund_metadata(indicators_df, metadata_df)
    
    # Stocker pour les autres pages
    st.session_state['indicators_df'] = indicators_df
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nb Fonds", len(selected_funds))
    with col2:
        st.metric("Sharpe Moyen", f"{indicators_df['Sharpe Ratio'].mean():.2f}")
    with col3:
        st.metric("Meilleur Rdt", f"{indicators_df['Rendement Annualise (%)'].max():.1f}%")
    with col4:
        st.metric("Pire Drawdown", f"{indicators_df['Max Drawdown (%)'].min():.1f}%")
    
    st.divider()
    
    # Tableau
    st.subheader("Tableau Recapitulatif")
    if selected_bench_calc:
        st.caption(f"Indicateurs calcules par rapport a: **{selected_bench_calc}**")
    
    col_categories = {
        "Rendements": ['Rendement Annualise (%)', 'Rendement 1Y (%)', 'Rendement 3Y (%)', 'Rendement 5Y (%)'],
        "Volatilite": ['Volatilite (%)', 'Volatilite 1Y (%)', 'Volatilite 3Y (%)', 'Volatilite 5Y (%)'],
        "Ratios": ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio', 'Information Ratio', 'Treynor Ratio'],
        "Risque": ['Max Drawdown (%)', 'Semi-Variance', 'Beta', 'Tracking Error (%)'],
        "Benchmark": ['Alpha (%)', 'Omega vs Benchmark', 'Bat Bench 1Y (%)', 'Bat Bench 3Y (%)', 'Bat Bench 5Y (%)'],
        "ESG & Frais": ['ESG Score', 'Frais (%)']
    }
    
    selected_categories = st.multiselect(
        "Categories a afficher",
        options=list(col_categories.keys()),
        default=["Rendements", "Ratios", "Benchmark", "ESG & Frais"]
    )
    
    cols_to_show = ['Fonds']
    for cat in selected_categories:
        cols_to_show.extend([c for c in col_categories[cat] if c in indicators_df.columns])
    
    df_display = indicators_df[[c for c in cols_to_show if c in indicators_df.columns]].copy()
    
    def highlight_best(s):
        higher_better = ['Sharpe Ratio', 'Sortino Ratio', 'Rendement Annualise (%)', 'Rendement 1Y (%)',
                        'Rendement 3Y (%)', 'Rendement 5Y (%)', 'Alpha (%)', 'Calmar Ratio', 
                        'Omega Ratio', 'Information Ratio', 'Treynor Ratio', 'ESG Score',
                        'Bat Bench 1Y (%)', 'Bat Bench 3Y (%)', 'Bat Bench 5Y (%)', 'Omega vs Benchmark']
        lower_better = ['Volatilite (%)', 'Volatilite 1Y (%)', 'Volatilite 3Y (%)', 'Volatilite 5Y (%)',
                       'Max Drawdown (%)', 'Semi-Variance', 'Frais (%)']
        
        if s.name in higher_better:
            is_max = s == s.max()
            return ['background-color: #D3E8CA' if v else '' for v in is_max]
        elif s.name in lower_better:
            is_min = s == s.min()
            return ['background-color: #D3E8CA' if v else '' for v in is_min]
        return ['' for _ in s]
    
    format_dict = {}
    for col in df_display.columns:
        if col == 'Fonds':
            continue
        elif any(x in col for x in ['Sharpe', 'Sortino', 'Calmar', 'Omega', 'Beta', 'Information', 'Treynor']):
            format_dict[col] = '{:.3f}'
        elif 'Semi-Variance' in col:
            format_dict[col] = '{:.6f}'
        elif '%' in col or 'Score' in col:
            format_dict[col] = '{:.2f}'
    
    styled_df = df_display.style.apply(highlight_best).format(format_dict, na_rep="-")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    csv = indicators_df.to_csv(index=False)
    st.download_button("Telecharger CSV", data=csv, file_name=f"indicateurs_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")


# ==============================================================================
# TAB 2: PERFORMANCE
# ==============================================================================

with tab2:
    st.header("Analyse de Performance")
    
    # Choix du benchmark
    if len(benchmark_cols) > 1:
        bench_perf = st.selectbox("Benchmark pour affichage", benchmark_cols, key="bench_tab2")
    elif len(benchmark_cols) == 1:
        bench_perf = benchmark_cols[0]
    else:
        bench_perf = None
    
    # Recalculer si benchmark different
    if bench_perf and 'indicators_df' in st.session_state:
        cols_for_calc = selected_funds + [bench_perf]
        df_calc = df_filtered[cols_for_calc]
        ret_calc = df_returns[cols_for_calc]
        indicators_perf = calculate_all_indicators(df_calc, ret_calc, bench_perf, risk_free_rate)
        if metadata_df is not None:
            indicators_perf = match_fund_metadata(indicators_perf, metadata_df)
    else:
        indicators_perf = st.session_state.get('indicators_df', pd.DataFrame())
    
    st.subheader("Evolution des Prix (Base 100)")
    cols_for_chart = selected_funds + benchmark_cols
    fig_prices = plot_normalized_prices(df_filtered[cols_for_chart], benchmark_cols=benchmark_cols, title="")
    st.plotly_chart(fig_prices, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rendement Annualise")
        fig_ret = plot_indicator_bar_chart(indicators_perf, "Rendement Annualise (%)", title="")
        st.plotly_chart(fig_ret, use_container_width=True)
    
    with col2:
        st.subheader("Sharpe Ratio")
        fig_sharpe = plot_indicator_bar_chart(indicators_perf, "Sharpe Ratio", title="")
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    st.divider()
    
    # Bat Benchmark visuel
    st.subheader("Bat le Benchmark (% des jours)")
    if bench_perf:
        st.caption(f"Pourcentage de jours ou le fonds surperforme {bench_perf} sur differents horizons")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'Bat Bench 1Y (%)' in indicators_perf.columns:
            fig_bat1y = plot_indicator_bar_chart(indicators_perf, "Bat Bench 1Y (%)", title="Horizon 1 an")
            st.plotly_chart(fig_bat1y, use_container_width=True)
    
    with col2:
        if 'Bat Bench 3Y (%)' in indicators_perf.columns:
            fig_bat3y = plot_indicator_bar_chart(indicators_perf, "Bat Bench 3Y (%)", title="Horizon 3 ans")
            st.plotly_chart(fig_bat3y, use_container_width=True)
    
    with col3:
        if 'Bat Bench 5Y (%)' in indicators_perf.columns:
            fig_bat5y = plot_indicator_bar_chart(indicators_perf, "Bat Bench 5Y (%)", title="Horizon 5 ans")
            st.plotly_chart(fig_bat5y, use_container_width=True)
    
    st.divider()
    
    st.subheader("Rendement vs Risque")
    fig_scatter = plot_risk_return_scatter(indicators_perf)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# TAB 3: RISQUE
# ==============================================================================

with tab3:
    st.header("Analyse de Risque")
    
    indicators_risk = st.session_state.get('indicators_df', pd.DataFrame())
    
    st.subheader("Drawdown")
    fig_dd = plot_drawdown(df_filtered[selected_funds], title="")
    st.plotly_chart(fig_dd, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Max Drawdown")
        fig_mdd = plot_indicator_bar_chart(indicators_risk, "Max Drawdown (%)", title="", reverse_colors=True)
        st.plotly_chart(fig_mdd, use_container_width=True)
    
    with col2:
        st.subheader("Volatilite")
        fig_vol = plot_indicator_bar_chart(indicators_risk, "Volatilite (%)", title="", reverse_colors=True)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    st.divider()
    
    st.subheader("Matrice de Correlation")
    fig_corr = plot_correlation_matrix(df_returns[selected_funds], title="")
    st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# TAB 4: BENCHMARK
# ==============================================================================

with tab4:
    st.header("Comparaison avec le(s) Benchmark(s)")
    
    if not benchmark_cols:
        st.warning("Aucun benchmark selectionne.")
    else:
        # Choix du benchmark
        if len(benchmark_cols) > 1:
            selected_bench = st.selectbox("Benchmark a afficher", benchmark_cols, key="bench_tab4")
        else:
            selected_bench = benchmark_cols[0]
        
        # Recalculer pour ce benchmark
        cols_for_calc = selected_funds + [selected_bench]
        df_calc = df_filtered[cols_for_calc]
        ret_calc = df_returns[cols_for_calc]
        indicators_bench = calculate_all_indicators(df_calc, ret_calc, selected_bench, risk_free_rate)
        if metadata_df is not None:
            indicators_bench = match_fund_metadata(indicators_bench, metadata_df)
        
        st.caption(f"Indicateurs par rapport a: **{selected_bench}**")
        
        st.subheader("Surperformance Cumulee")
        fig_outperf = plot_benchmark_comparison(df_returns, benchmark_col=selected_bench, title="")
        st.plotly_chart(fig_outperf, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Alpha")
            if 'Alpha (%)' in indicators_bench.columns:
                fig_alpha = plot_indicator_bar_chart(indicators_bench, "Alpha (%)", title="")
                st.plotly_chart(fig_alpha, use_container_width=True)
        
        with col2:
            st.subheader("Tracking Error")
            if 'Tracking Error (%)' in indicators_bench.columns:
                fig_te = plot_indicator_bar_chart(indicators_bench, "Tracking Error (%)", title="")
                st.plotly_chart(fig_te, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Information Ratio")
            if 'Information Ratio' in indicators_bench.columns:
                fig_ir = plot_indicator_bar_chart(indicators_bench, "Information Ratio", title="")
                st.plotly_chart(fig_ir, use_container_width=True)
        
        with col2:
            st.subheader("Omega vs Benchmark")
            if 'Omega vs Benchmark' in indicators_bench.columns:
                fig_omega_bench = plot_indicator_bar_chart(indicators_bench, "Omega vs Benchmark", title="")
                st.plotly_chart(fig_omega_bench, use_container_width=True)
        
        st.divider()
        
        st.subheader("Treynor Ratio")
        if 'Treynor Ratio' in indicators_bench.columns:
            fig_tr = plot_indicator_bar_chart(indicators_bench, "Treynor Ratio", title="")
            st.plotly_chart(fig_tr, use_container_width=True)


# ==============================================================================
# TAB 5: RADAR
# ==============================================================================

with tab5:
    st.header("Comparaison Radar des Fonds")
    
    indicators_radar = st.session_state.get('indicators_df', pd.DataFrame())
    
    if len(indicators_radar) > 0:
        # Multi-selection de fonds
        all_funds = indicators_radar['Fonds'].tolist()
        selected_funds_radar = st.multiselect(
            "Selectionner les fonds a comparer",
            options=all_funds,
            default=all_funds[:3] if len(all_funds) >= 3 else all_funds,
            key="radar_funds"
        )
        
        if len(selected_funds_radar) == 0:
            st.warning("Selectionnez au moins un fonds.")
        else:
            # Choix des metriques
            metric_options = ['Rendement Annualise (%)', 'Sharpe Ratio', 'Sortino Ratio', 
                           'Alpha (%)', 'Information Ratio', 'Treynor Ratio', 'Omega vs Benchmark',
                           'ESG Score', 'Volatilite (%)', 'Max Drawdown (%)', 'Frais (%)']
            
            available_metrics = []
            for col in metric_options:
                if col in indicators_radar.columns and indicators_radar[col].notna().sum() > 0:
                    available_metrics.append(col)
            
            selected_metrics = st.multiselect(
                "Metriques a afficher",
                options=available_metrics,
                default=available_metrics[:5] if len(available_metrics) >= 5 else available_metrics,
                key="radar_metrics"
            )
            
            if len(selected_metrics) < 3:
                st.warning("Selectionnez au moins 3 metriques.")
            else:
                fig_radar = plot_radar_multi_funds(
                    indicators_radar, 
                    selected_funds_radar, 
                    metrics=selected_metrics,
                    title=""
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        
        st.divider()
        
        # Explication
        with st.expander("Comment lire le radar ?"):
            st.markdown("""
            **Interpretation:**
            - Chaque axe represente une metrique normalisee sur 0-100
            - Plus la surface est grande, meilleur est le fonds sur ces criteres
            - Pour Volatilite, Drawdown et Frais: l'echelle est inversee (plus grand = meilleur = moins de risque/frais)
            
            **Omega vs Benchmark:**
            - Mesure combien tu gagnes vs le benchmark pour chaque euro perdu vs le benchmark
            - > 1 = tu surperformes plus souvent/fort que tu sous-performes
            - C'est un ratio gains/pertes RELATIF au benchmark
            """)


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.caption(f"""
Dashboard Analyse Fonds Large Cap V2 | 
Periode: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')} | 
Taux sans risque: {risk_free_rate*100:.2f}% | 
{datetime.now().strftime('%d/%m/%Y %H:%M')}
""")