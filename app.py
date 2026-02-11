# ==============================================================================
# DASHBOARD ANALYSE FONDS LARGE CAP
# Application Streamlit
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration de la page (DOIT ETRE EN PREMIER)
st.set_page_config(
    page_title="Analyse Fonds Large Cap",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports locaux
from config import (
    DATA_PATH, 
    BENCHMARK_COLUMN, 
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR
)
from utils.data_loader import (
    load_data, 
    get_fund_columns, 
    get_valid_date_range,
    filter_data_by_period,
    calculate_returns,
    get_fund_metadata
)
from utils.indicators import (
    calculate_all_indicators,
    calculate_rolling_sharpe,
    calculate_drawdown_series
)
from utils.charts import (
    plot_normalized_prices,
    plot_drawdown,
    plot_risk_return_scatter,
    plot_rolling_sharpe,
    plot_benchmark_comparison,
    plot_indicator_comparison,
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
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .indicator-explanation {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HEADER
# ==============================================================================

st.markdown('<p class="main-header">📊 Analyse de Fonds Large Cap</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Outil d\'analyse de performance et de risque</p>', unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR - CHARGEMENT DES DONNEES ET FILTRES
# ==============================================================================

st.sidebar.header("📁 Données")

# Option 1: Upload de fichier
uploaded_file = st.sidebar.file_uploader(
    "Charger un fichier de données",
    type=['xlsx', 'xls', 'csv'],
    help="Format attendu: Date en première colonne, puis une colonne par fonds"
)

# Option 2: Utiliser le fichier par défaut
use_default = st.sidebar.checkbox(
    "Utiliser le fichier par défaut",
    value=uploaded_file is None,
    help=f"Cherche le fichier dans {DATA_PATH}"
)

# Charger les données
df_prices = None
error_message = None

try:
    if uploaded_file is not None:
        df_prices = load_data(uploaded_file=uploaded_file)
        st.sidebar.success(f"✅ Fichier chargé: {uploaded_file.name}")
    elif use_default:
        df_prices = load_data(file_path=DATA_PATH)
        st.sidebar.success(f"✅ Fichier par défaut chargé")
except Exception as e:
    error_message = str(e)
    st.sidebar.error(f"❌ Erreur: {error_message}")

# Si pas de données, afficher un message et arrêter
if df_prices is None:
    st.warning("""
    ### 👋 Bienvenue !
    
    Pour commencer, veuillez charger vos données de fonds via la sidebar.
    
    **Format attendu:**
    - Fichier Excel (.xlsx) ou CSV
    - Première colonne: Date
    - Colonnes suivantes: Prix/NAV de chaque fonds
    - Une colonne "Benchmark" pour l'indice de référence
    
    **Exemple:**
    | Date | Fonds_A | Fonds_B | Benchmark |
    |------|---------|---------|-----------|
    | 2020-01-02 | 100.5 | 98.2 | 3500 |
    | 2020-01-03 | 101.2 | 99.1 | 3520 |
    """)
    st.stop()


# ==============================================================================
# SIDEBAR - CONFIGURATION
# ==============================================================================

st.sidebar.divider()
st.sidebar.header("⚙️ Configuration")

# Identifier la colonne benchmark
all_columns = list(df_prices.columns)
benchmark_candidates = [col for col in all_columns if 'bench' in col.lower() or 'indice' in col.lower() or 'index' in col.lower()]

if benchmark_candidates:
    default_bench = benchmark_candidates[0]
else:
    default_bench = all_columns[-1] if all_columns else None

benchmark_col = st.sidebar.selectbox(
    "Colonne Benchmark",
    options=all_columns,
    index=all_columns.index(default_bench) if default_bench in all_columns else 0,
    help="L'indice de référence pour comparer les fonds"
)

# Taux sans risque
st.sidebar.subheader("Taux sans risque")

rf_option = st.sidebar.radio(
    "Source du taux",
    options=["Manuel", "Euribor 3M (auto)"],
    horizontal=True,
    help="Euribor récupéré automatiquement via Yahoo Finance"
)

if rf_option == "Euribor 3M (auto)":
    try:
        from utils.indicators import get_euribor_rate
        risk_free_rate = get_euribor_rate("3M")
        st.sidebar.success(f"✅ Euribor 3M: {risk_free_rate*100:.2f}%")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erreur Euribor, taux par défaut utilisé")
        risk_free_rate = DEFAULT_RISK_FREE_RATE
else:
    risk_free_rate = st.sidebar.slider(
        "Taux sans risque (%)",
        min_value=0.0,
        max_value=10.0,
        value=DEFAULT_RISK_FREE_RATE * 100,
        step=0.1,
        help="Taux sans risque annualisé pour le calcul du Sharpe/Sortino"
    ) / 100

st.sidebar.divider()
st.sidebar.header("🎯 Sélection des Fonds")

# Liste des fonds (exclure benchmark)
fund_columns = get_fund_columns(df_prices, benchmark_col)

# Sélection multiple des fonds
selected_funds = st.sidebar.multiselect(
    "Fonds à analyser",
    options=fund_columns,
    default=fund_columns[:5] if len(fund_columns) > 5 else fund_columns,
    help="Sélectionnez les fonds que vous souhaitez comparer"
)

if not selected_funds:
    st.warning("⚠️ Veuillez sélectionner au moins un fonds dans la sidebar.")
    st.stop()

st.sidebar.divider()
st.sidebar.header("📅 Période d'Analyse")

# Calculer la plage de dates valide
date_min, date_max = get_valid_date_range(df_prices, selected_funds)

st.sidebar.caption(f"Période disponible: {date_min.strftime('%d/%m/%Y')} - {date_max.strftime('%d/%m/%Y')}")

# Sélection de la période
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Date début",
        value=date_min,
        min_value=date_min.date(),
        max_value=date_max.date()
    )
with col2:
    end_date = st.date_input(
        "Date fin",
        value=date_max,
        min_value=date_min.date(),
        max_value=date_max.date()
    )

# Convertir en Timestamp
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

# Vérifier la validité
if start_date >= end_date:
    st.error("❌ La date de début doit être antérieure à la date de fin.")
    st.stop()


# ==============================================================================
# FILTRAGE DES DONNEES
# ==============================================================================

# Colonnes à garder (fonds sélectionnés + benchmark)
cols_to_keep = list(selected_funds) + [benchmark_col]
df_filtered = filter_data_by_period(df_prices[cols_to_keep], start_date, end_date)

# Calculer les rendements
df_returns = calculate_returns(df_filtered)

# Supprimer les lignes avec trop de NaN
df_filtered = df_filtered.dropna(how='all')
df_returns = df_returns.dropna(how='all')


# ==============================================================================
# CALCUL DES INDICATEURS
# ==============================================================================

indicators_df = calculate_all_indicators(
    prices=df_filtered,
    returns=df_returns,
    benchmark_col=benchmark_col,
    risk_free_rate=risk_free_rate
)


# ==============================================================================
# TABS PRINCIPAUX
# ==============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue Globale",
    "📈 Performance",
    "⚠️ Risque",
    "🏆 vs Benchmark",
    "📚 Explications"
])


# ==============================================================================
# TAB 1: VUE GLOBALE
# ==============================================================================

with tab1:
    st.header("Vue Globale des Indicateurs")
    
    # Résumé rapide
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Nb Fonds Analysés",
            len(selected_funds),
            help="Nombre de fonds sélectionnés"
        )
    
    with col2:
        avg_sharpe = indicators_df['Sharpe Ratio'].mean()
        st.metric(
            "Sharpe Moyen",
            f"{avg_sharpe:.2f}",
            help="Moyenne des ratios de Sharpe"
        )
    
    with col3:
        best_return = indicators_df['Rendement Annualise (%)'].max()
        st.metric(
            "Meilleur Rendement",
            f"{best_return:.1f}%",
            help="Meilleur rendement annualisé"
        )
    
    with col4:
        worst_dd = indicators_df['Max Drawdown (%)'].min()
        st.metric(
            "Pire Drawdown",
            f"{worst_dd:.1f}%",
            delta_color="inverse",
            help="Pire drawdown parmi les fonds"
        )
    
    st.divider()
    
    # Tableau des indicateurs
    st.subheader("Tableau Récapitulatif")
    
    # Sélection des colonnes à afficher
    col_categories = {
        "Rendements": ['Rendement Annualise (%)', 'Rendement 1Y (%)', 'Rendement 3Y (%)', 'Rendement 5Y (%)', 'Rendement Cumule (%)'],
        "Volatilité": ['Volatilite (%)', 'Volatilite 1Y (%)', 'Volatilite 3Y (%)', 'Volatilite 5Y (%)'],
        "Ratios": ['Sharpe Ratio', 'Sharpe 1Y', 'Sharpe 3Y', 'Sharpe 5Y', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio'],
        "Risque": ['Max Drawdown (%)', 'Semi-Variance', 'Beta'],
        "Benchmark": ['Alpha (%)', '% Bat le Benchmark']
    }
    
    selected_categories = st.multiselect(
        "Catégories d'indicateurs à afficher",
        options=list(col_categories.keys()),
        default=["Rendements", "Volatilité", "Ratios"],
        help="Sélectionnez les catégories d'indicateurs à afficher dans le tableau"
    )
    
    # Construire la liste des colonnes à afficher
    cols_to_show = ['Fonds']
    for cat in selected_categories:
        cols_to_show.extend([c for c in col_categories[cat] if c in indicators_df.columns])
    
    # Formatter le dataframe pour l'affichage
    df_display = indicators_df[cols_to_show].copy()
    
    # Style conditionnel
    def highlight_best(s):
        if s.name in ['Sharpe Ratio', 'Sharpe 1Y', 'Sharpe 3Y', 'Sharpe 5Y', 
                      'Sortino Ratio', 'Rendement Annualise (%)', 'Rendement 1Y (%)',
                      'Rendement 3Y (%)', 'Rendement 5Y (%)', 'Rendement Cumule (%)',
                      'Alpha (%)', 'Calmar Ratio', 'Omega Ratio', '% Bat le Benchmark']:
            is_max = s == s.max()
            return ['background-color: #90EE90' if v else '' for v in is_max]
        elif s.name in ['Max Drawdown (%)', 'Volatilite (%)', 'Volatilite 1Y (%)',
                        'Volatilite 3Y (%)', 'Volatilite 5Y (%)', 'Semi-Variance']:
            is_min = s == s.min()
            return ['background-color: #90EE90' if v else '' for v in is_min]
        return ['' for _ in s]
    
    # Créer le format dict dynamiquement
    format_dict = {}
    for col in df_display.columns:
        if col == 'Fonds':
            continue
        elif 'Sharpe' in col or 'Sortino' in col or 'Calmar' in col or 'Omega' in col or 'Beta' in col:
            format_dict[col] = '{:.3f}'
        elif 'Semi-Variance' in col:
            format_dict[col] = '{:.6f}'
        elif '%' in col:
            format_dict[col] = '{:.2f}%'
    
    styled_df = df_display.style.apply(highlight_best).format(format_dict, na_rep="-")
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = indicators_df.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger les indicateurs (CSV)",
        data=csv,
        file_name=f"indicateurs_fonds_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


# ==============================================================================
# TAB 2: PERFORMANCE
# ==============================================================================

with tab2:
    st.header("Analyse de Performance")
    
    # Graphique des prix normalisés
    st.subheader("Evolution des Prix (Base 100)")
    fig_prices = plot_normalized_prices(
        df_filtered,
        benchmark_col=benchmark_col,
        title=""
    )
    st.plotly_chart(fig_prices, use_container_width=True)
    
    st.divider()
    
    # Scatter Rendement vs Volatilité
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rendement vs Risque")
        fig_scatter = plot_risk_return_scatter(
            indicators_df,
            x_col="Volatilite (%)",
            y_col="Rendement Annualise (%)",
            title=""
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.subheader("Comparaison des Rendements")
        fig_return = plot_indicator_comparison(
            indicators_df,
            "Rendement Annualise (%)",
            title=""
        )
        st.plotly_chart(fig_return, use_container_width=True)
    
    st.divider()
    
    # Sharpe glissant
    st.subheader("Sharpe Ratio Glissant (3 mois)")
    fig_rolling = plot_rolling_sharpe(
        df_returns[selected_funds],
        window=63,
        risk_free_rate=risk_free_rate,
        title=""
    )
    st.plotly_chart(fig_rolling, use_container_width=True)


# ==============================================================================
# TAB 3: RISQUE
# ==============================================================================

with tab3:
    st.header("Analyse de Risque")
    
    # Drawdown
    st.subheader("Drawdown dans le Temps")
    fig_dd = plot_drawdown(
        df_filtered[selected_funds],
        title=""
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Comparaison Max Drawdown")
        fig_mdd = plot_indicator_comparison(
            indicators_df,
            "Max Drawdown (%)",
            title=""
        )
        st.plotly_chart(fig_mdd, use_container_width=True)
    
    with col2:
        st.subheader("Comparaison Volatilité")
        fig_vol = plot_indicator_comparison(
            indicators_df,
            "Volatilite (%)",
            title=""
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    st.divider()
    
    # Matrice de corrélation
    st.subheader("Matrice de Corrélation")
    st.caption("Permet de voir quels fonds évoluent ensemble (diversification)")
    
    fig_corr = plot_correlation_matrix(
        df_returns[selected_funds],
        title=""
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# TAB 4: BENCHMARK COMPARISON
# ==============================================================================

with tab4:
    st.header("Comparaison avec le Benchmark")
    st.caption(f"Benchmark utilisé: {benchmark_col}")
    
    # Surperformance cumulée
    st.subheader("Surperformance Cumulée")
    fig_outperf = plot_benchmark_comparison(
        df_returns,
        benchmark_col=benchmark_col,
        title=""
    )
    st.plotly_chart(fig_outperf, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Alpha (surperformance ajustée)")
        if 'Alpha (%)' in indicators_df.columns:
            fig_alpha = plot_indicator_comparison(
                indicators_df,
                "Alpha (%)",
                title=""
            )
            st.plotly_chart(fig_alpha, use_container_width=True)
    
    with col2:
        st.subheader("% Périodes Battant le Benchmark")
        if '% Bat le Benchmark' in indicators_df.columns:
            fig_beats = plot_indicator_comparison(
                indicators_df,
                "% Bat le Benchmark",
                title=""
            )
            st.plotly_chart(fig_beats, use_container_width=True)
    
    st.divider()
    
    # Beta
    st.subheader("Beta (sensibilité au marché)")
    if 'Beta' in indicators_df.columns:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig_beta = plot_indicator_comparison(
                indicators_df,
                "Beta",
                title=""
            )
            st.plotly_chart(fig_beta, use_container_width=True)


# ==============================================================================
# TAB 5: EXPLICATIONS
# ==============================================================================

with tab5:
    st.header("📚 Guide des Indicateurs")
    st.caption("Explications simples pour comprendre chaque indicateur")
    
    explanations = {
        "Rendement Annualisé": {
            "formule": "(1 + rendement_total)^(252/n_jours) - 1",
            "explication": "Combien tu gagnes en moyenne par an si tu gardes ton placement. C'est le premier indicateur à regarder, mais attention, il ne dit rien sur le risque pris pour l'obtenir.",
            "interpretation": "Plus c'est haut, mieux c'est. Un rendement de 8% signifie que 100€ investis deviennent 108€ en un an."
        },
        "Rendement 1Y / 3Y / 5Y": {
            "formule": "Même formule mais sur les 1, 3 ou 5 dernières années",
            "explication": "Permet de voir si le fonds performe bien sur différents horizons de temps. Un fonds peut être bon sur 1 an mais mauvais sur 5 ans (ou l'inverse).",
            "interpretation": "Compare les rendements sur différentes périodes. Si 1Y >> 5Y, le fonds s'est amélioré récemment. Si 1Y << 5Y, il a peut-être des difficultés."
        },
        "Volatilité": {
            "formule": "écart-type des rendements × √252",
            "explication": "Est-ce que le fonds fait les montagnes russes ou c'est tranquille ? La volatilité mesure l'amplitude des variations quotidiennes.",
            "interpretation": "Plus c'est bas, plus c'est stable. Une vol de 15% signifie que le fonds peut facilement varier de +15% ou -15% sur l'année."
        },
        "Volatilité 1Y / 3Y / 5Y": {
            "formule": "Même formule mais sur les 1, 3 ou 5 dernières années",
            "explication": "Permet de voir si le fonds est devenu plus ou moins risqué au fil du temps.",
            "interpretation": "Si Vol 1Y < Vol 5Y, le fonds s'est calmé récemment. Si Vol 1Y > Vol 5Y, il est devenu plus nerveux."
        },
        "Sharpe Ratio": {
            "formule": "(rendement - taux sans risque) / volatilité",
            "explication": "Est-ce que le risque pris en valait la peine ? C'est le rendement que tu obtiens pour chaque unité de risque. C'est l'indicateur roi en gestion de portefeuille.",
            "interpretation": "< 0 : mauvais | 0-1 : moyen | 1-2 : bon | > 2 : excellent"
        },
        "Sortino Ratio": {
            "formule": "(rendement - taux sans risque) / volatilité négative",
            "explication": "Comme le Sharpe mais on punit seulement les baisses, pas les hausses. C'est plus juste parce qu'on s'en fiche si le fonds monte beaucoup !",
            "interpretation": "Mêmes seuils que le Sharpe. Souvent plus élevé car il ignore la 'bonne' volatilité."
        },
        "Max Drawdown": {
            "formule": "(creux - pic) / pic",
            "explication": "La pire dégringolade qu'on aurait pu subir si on avait acheté au pire moment. C'est le scénario catastrophe.",
            "interpretation": "-20% signifie que dans le pire des cas, tu aurais perdu 20% de ton investissement avant que ça remonte."
        },
        "Beta": {
            "formule": "Cov(fonds, marché) / Var(marché)",
            "explication": "Quand le marché bouge de 1%, le fonds bouge de combien ?",
            "interpretation": "Beta = 1 : pareil que le marché | Beta > 1 : plus nerveux | Beta < 1 : plus calme"
        },
        "Alpha": {
            "formule": "Rendement - (Rf + Beta × (Rm - Rf))",
            "explication": "Le petit plus (ou moins) que le gérant apporte par rapport à ce qu'on attendait vu le risque pris.",
            "interpretation": "Positif = le gérant crée de la valeur. Négatif = autant acheter l'indice."
        },
        "Calmar Ratio": {
            "formule": "rendement annualisé / |max drawdown|",
            "explication": "Le rendement par rapport à la pire chute subie. Combine performance et risque extrême.",
            "interpretation": "> 1 signifie que ton rendement annuel est supérieur à ta pire perte. Plus c'est haut, mieux c'est."
        },
        "Omega Ratio": {
            "formule": "Σ gains au-dessus du seuil / Σ pertes en-dessous",
            "explication": "Pour chaque euro que tu risques de perdre, combien tu peux gagner ? Prend en compte toute la distribution des rendements.",
            "interpretation": "Omega > 1 = tu gagnes plus que tu perds en moyenne. Plus c'est haut, mieux c'est."
        },
        "% Bat le Benchmark": {
            "formule": "Nb(jours où fonds > benchmark) / Nb(jours total)",
            "explication": "Sur 100 jours, combien de fois le fonds a fait mieux que le marché ?",
            "interpretation": "50% = pareil que le marché. > 50% = le fonds surperforme régulièrement."
        },
        "Taux Sans Risque (Euribor)": {
            "formule": "Taux interbancaire européen",
            "explication": "C'est le rendement qu'on peut obtenir 'sans risque' (en théorie). On l'utilise comme référence : si un fonds fait moins bien que l'Euribor, autant laisser son argent à la banque !",
            "interpretation": "L'Euribor 3M est le taux auquel les banques se prêtent entre elles sur 3 mois. Il varie selon la politique de la BCE."
        }
    }
    
    for indicator, details in explanations.items():
        with st.expander(f"📌 {indicator}"):
            st.markdown(f"**Formule:** `{details['formule']}`")
            st.markdown(f"**Explication:** {details['explication']}")
            st.info(f"**Interprétation:** {details['interpretation']}")


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.caption(f"""
📊 Dashboard Analyse Fonds Large Cap | 
Période: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')} | 
Taux sans risque: {risk_free_rate*100:.1f}% | 
Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}
""")
