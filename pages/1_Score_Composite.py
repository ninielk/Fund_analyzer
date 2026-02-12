# ==============================================================================
# PAGE SCORE COMPOSITE - Ponderation personnalisee
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Score Composite",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.indicators import calculate_composite_score
from utils.charts import plot_score_composite_bar, plot_radar_comparison, COLORS


st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #412761;
        margin-bottom: 1rem;
    }
    .weight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #007078;
    }
    .score-card {
        background-color: #D3E8CA;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .score-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #007078;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-header">Score Composite avec Ponderation</p>', unsafe_allow_html=True)

# Verifier que les donnees sont disponibles
if 'indicators_df' not in st.session_state:
    st.warning("Veuillez d'abord charger des donnees sur la page principale.")
    st.stop()

indicators_df = st.session_state['indicators_df'].copy()


# ==============================================================================
# SIDEBAR - PONDERATIONS
# ==============================================================================

st.sidebar.header("Ponderations")
st.sidebar.markdown("Ajustez les poids de chaque categorie (total = 100%)")

# Sliders pour les ponderations
w_perf = st.sidebar.slider("Performance (Rendement, Sharpe...)", 0, 100, 25, 5)
w_risk = st.sidebar.slider("Risque (Vol, Drawdown...)", 0, 100, 25, 5)
w_bench = st.sidebar.slider("vs Benchmark (Alpha, IR...)", 0, 100, 25, 5)
w_esg = st.sidebar.slider("ESG Score", 0, 100, 15, 5)
w_frais = st.sidebar.slider("Frais (inverse)", 0, 100, 10, 5)

total_weight = w_perf + w_risk + w_bench + w_esg + w_frais

if total_weight != 100:
    st.sidebar.warning(f"Total: {total_weight}% (devrait etre 100%)")
else:
    st.sidebar.success(f"Total: {total_weight}%")

# Normaliser les poids
weights = {
    'Performance': w_perf / 100,
    'Risque': w_risk / 100,
    'Benchmark': w_bench / 100,
    'ESG': w_esg / 100,
    'Frais': w_frais / 100
}


# ==============================================================================
# CALCUL DU SCORE COMPOSITE
# ==============================================================================

# Calculer le score avec les ponderations choisies
df_scored = calculate_composite_score(indicators_df, weights=weights)


# ==============================================================================
# AFFICHAGE
# ==============================================================================

st.header("Resultats")

# Top 3
st.subheader("Top 3 Fonds")

top3 = df_scored.nlargest(3, 'Score Composite (0-100)')

cols = st.columns(3)
medals = ["1er", "2eme", "3eme"]
medal_colors = ["#F8AF00", "#6E6E6E", "#B10967"]

for i, (_, row) in enumerate(top3.iterrows()):
    with cols[i]:
        st.markdown(f"""
        <div style='background-color: {medal_colors[i]}20; padding: 1rem; border-radius: 0.5rem; text-align: center; border: 2px solid {medal_colors[i]};'>
            <div style='font-size: 1.5rem; font-weight: bold; color: {medal_colors[i]};'>{medals[i]}</div>
            <div style='font-size: 1rem; margin: 0.5rem 0;'>{row['Fonds'][:20]}...</div>
            <div style='font-size: 2rem; font-weight: bold; color: #412761;'>{row['Score Composite (0-100)']:.1f}</div>
            <div style='font-size: 0.8rem; color: #6E6E6E;'>/ 100</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Bar chart
st.subheader("Classement complet")
fig_score = plot_score_composite_bar(df_scored)
st.plotly_chart(fig_score, use_container_width=True)

st.divider()

# Tableau detaille
st.subheader("Detail des scores")

cols_to_show = ['Fonds', 'Score Composite (0-100)', 'Z-Score Moyen']

# Ajouter les colonnes disponibles
for col in ['Rendement Annualise (%)', 'Sharpe Ratio', 'Volatilite (%)', 'Max Drawdown (%)', 
            'Alpha (%)', 'Information Ratio', 'ESG Score', 'Frais (%)']:
    if col in df_scored.columns:
        cols_to_show.append(col)

df_display = df_scored[cols_to_show].sort_values('Score Composite (0-100)', ascending=False)

st.dataframe(df_display.style.format({
    'Score Composite (0-100)': '{:.1f}',
    'Z-Score Moyen': '{:.3f}',
    'Rendement Annualise (%)': '{:.2f}',
    'Sharpe Ratio': '{:.3f}',
    'Volatilite (%)': '{:.2f}',
    'Max Drawdown (%)': '{:.2f}',
    'Alpha (%)': '{:.2f}',
    'Information Ratio': '{:.3f}',
    'ESG Score': '{:.1f}',
    'Frais (%)': '{:.2f}'
}, na_rep="-"), use_container_width=True, hide_index=True)

st.divider()

# Radar comparatif des top 3
st.subheader("Comparaison Radar - Top 3")

top3_names = top3['Fonds'].tolist()
metrics_for_radar = []
for col in ['Rendement Annualise (%)', 'Sharpe Ratio', 'Alpha (%)', 'ESG Score']:
    if col in df_scored.columns and df_scored[col].notna().sum() > 0:
        metrics_for_radar.append(col)

if len(metrics_for_radar) >= 3:
    fig_radar = plot_radar_comparison(df_scored, top3_names, metrics=metrics_for_radar)
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# Explication de la methodologie
with st.expander("Methodologie du Score Composite"):
    st.markdown("""
    ### Comment est calcule le score ?
    
    1. **Z-Score par indicateur**: Pour chaque indicateur, on calcule le z-score:
       ```
       z = (valeur - moyenne) / ecart-type
       ```
    
    2. **Winsorisation**: Les z-scores sont bornes entre -3 et +3 pour eviter les valeurs extremes.
    
    3. **Inversion**: Pour les indicateurs ou "plus bas = mieux" (volatilite, frais, drawdown), 
       le z-score est inverse.
    
    4. **Moyenne ponderee**: Les z-scores sont moyennes selon vos ponderations par categorie.
    
    5. **Rebasage 0-100**: Le score final est calcule:
       ```
       Score = (z_moyen + 3) / 6 × 100
       ```
       
    ### Interpretation
    - **Score > 70**: Excellent fonds sur les criteres choisis
    - **Score 50-70**: Fonds dans la moyenne
    - **Score < 50**: Fonds en-dessous de la moyenne
    
    ### Categories
    - **Performance**: Rendement annualise, Sharpe, Sortino, Calmar
    - **Risque**: Volatilite, Max Drawdown, Semi-variance (inverse)
    - **Benchmark**: Alpha, Information Ratio, Treynor
    - **ESG**: Score ESG (0-100)
    - **Frais**: Frais de gestion (inverse)
    """)


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.caption(f"Ponderations: Perf={w_perf}% | Risque={w_risk}% | Bench={w_bench}% | ESG={w_esg}% | Frais={w_frais}%")
