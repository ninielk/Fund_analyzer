# Dashboard Analyse Fonds Large Cap - V2

## Nouveautes V2

- **Tracking Error** avec flag (OK si entre 2% et 5%)
- **Information Ratio** et **Treynor Ratio**
- **Score Composite** avec ponderation personnalisee
- **Lecture ESG et Frais** depuis feuille 2 du fichier Excel
- **Radar Chart** pour visualiser le profil des fonds
- **Warning ESG** si meilleur ou pire score (eliminatoire)
- **Nouveau calcul "Bat Benchmark"** par horizon 1Y/3Y/5Y
- **Nouvelles couleurs** charte graphique

## Structure du projet

```
fund_analyzer_v2/
├── app.py                      # Application principale
├── config.py                   # Configuration et couleurs
├── requirements.txt            # Dependances
├── pages/
│   └── 1_Score_Composite.py    # Page score avec ponderation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # Chargement donnees
│   ├── indicators.py           # Calcul indicateurs
│   └── charts.py               # Visualisations
├── .streamlit/
│   └── config.toml             # Config Streamlit
└── data/
    └── (vos fichiers Excel)
```

## Format du fichier Excel

### Feuille 1 - Prix/NAV
| Dates | Benchmark | Taux_RF | Fonds_A | Fonds_B | ... |
|-------|-----------|---------|---------|---------|-----|
| 2015-01-01 | 100 | 99.5 | 100 | 100 | ... |

### Feuille 2 - Metadonnees
| Metrique | Fonds_A | Fonds_B | ... |
|----------|---------|---------|-----|
| ESG Score | 75.5 | 82.3 | ... |
| Frais (%) | 1.5 | 0.9 | ... |

## Installation

```bash
cd fund_analyzer_v2
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Indicateurs calcules

### Performance
- Rendement annualise (global + 1Y/3Y/5Y)
- Sharpe Ratio (global + 1Y/3Y/5Y)
- Sortino Ratio
- Calmar Ratio
- Omega Ratio

### Risque
- Volatilite (global + 1Y/3Y/5Y)
- Max Drawdown
- Semi-Variance
- Beta

### vs Benchmark
- Alpha
- Tracking Error + Flag
- Information Ratio
- Treynor Ratio
- Bat Benchmark (nb/obs) par horizon 1Y/3Y/5Y

### Autres
- ESG Score (depuis feuille 2)
- Frais (depuis feuille 2)
- Score Composite (0-100) avec ponderation

## Formules cles

### Tracking Error
```
TE = std(R_fonds - R_benchmark) × sqrt(252)
Flag OK si 2% < TE < 5%
```

### Information Ratio
```
IR = Alpha / (Tracking_Error / 100)
```

### Treynor Ratio
```
Treynor = (Rendement - Rf) / Beta
```

### Bat Benchmark (1Y/3Y/5Y)
Pour chaque paire de dates espacees de N ans:
1. Rendement simple = (P2 - P1) / P1
2. Binaire = 1 si fonds > benchmark, sinon 0
3. Score = sum(binaires) / total_observations

### Score Composite
1. Z-score = (valeur - moyenne) / std
2. Winsoriser entre -3 et 3
3. Inverser si "lower is better"
4. Moyenne ponderee par categorie
5. Rebaser: Score = (z + 3) / 6 × 100
