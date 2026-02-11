# Dashboard Analyse Fonds Large Cap

## Structure du projet

```
fund_analyzer/
├── app.py                 # Point d'entrée Streamlit (lance ça)
├── pyproject.toml         # Configuration Poetry
├── requirements.txt       # Dépendances (si pas Poetry)
├── config.py              # Configuration (taux sans risque, etc.)
├── data/
│   └── fonds_data.xlsx    # TON FICHIER DE DONNÉES ICI
├── utils/
│   ├── __init__.py
│   ├── data_loader.py     # Chargement et nettoyage des données
│   ├── indicators.py      # Calcul de tous les indicateurs
│   └── charts.py          # Fonctions de visualisation
└── README.md
```

## Installation avec Poetry (recommandé)

```bash
# 1. Aller dans le dossier
cd fund_analyzer

# 2. Installer les dépendances avec Poetry
poetry install

# 3. Activer l'environnement virtuel
poetry shell

# 4. Mettre ton fichier de données dans data/fonds_data.xlsx

# 5. Lancer le dashboard
streamlit run app.py
```

## Installation avec venv classique

```bash
# 1. Créer le venv
python -m venv .venv

# 2. Activer (Windows)
.venv\Scripts\activate

# 2. Activer (Mac/Linux)
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Mettre ton fichier de données dans data/fonds_data.xlsx

# 5. Lancer le dashboard
streamlit run app.py
```

## Format des données attendu

Ton fichier Excel/CSV doit avoir ce format :

| Date | Fonds_A | Fonds_B | Fonds_C | Benchmark |
|------|---------|---------|---------|-----------|
| 2020-01-02 | 100.5 | NaN | 98.2 | 3500 |
| 2020-01-03 | 101.2 | NaN | 99.1 | 3520 |

- **Colonne Date** : format YYYY-MM-DD ou DD/MM/YYYY
- **Colonnes Fonds** : cours/NAV quotidiens
- **Colonne Benchmark** : cours de l'indice de référence (ex: MSCI Europe Large Cap)
- **NaN** : OK pour les fonds qui n'existaient pas encore

## Indicateurs calculés

### Globaux (sur toute la période)
- Rendement annualisé
- Volatilité
- Sharpe Ratio
- Sortino Ratio
- Semi-Variance
- Max Drawdown
- Calmar Ratio
- Omega Ratio
- Beta / Alpha
- % bat le benchmark

### Par période
- Rendement 1Y, 3Y, 5Y
- Volatilité 1Y, 3Y, 5Y
- Sharpe 1Y, 3Y, 5Y

## Taux sans risque

Deux options :
1. **Manuel** : tu rentres le taux toi-même
2. **Euribor 3M (auto)** : récupéré automatiquement via Yahoo Finance

## Déploiement Streamlit Cloud

1. Push ton code sur GitHub
2. Va sur https://share.streamlit.io
3. Connecte ton repo
4. Sélectionne `app.py` comme fichier principal
5. Deploy !
