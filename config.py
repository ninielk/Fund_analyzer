# ==============================================================================
# CONFIGURATION DU DASHBOARD
# ==============================================================================

# Chemin vers le fichier de données
DATA_PATH = "data/fonds_data.xlsx"

# Nom de la colonne benchmark dans ton fichier
BENCHMARK_COLUMN = "Benchmark"

# Nom de la colonne date
DATE_COLUMN = "Date"

# Taux sans risque par défaut (annualisé, en décimal)
# 3% = 0.03
DEFAULT_RISK_FREE_RATE = 0.03

# Nombre de jours de trading par an (pour annualisation)
TRADING_DAYS_PER_YEAR = 252

# Seuil pour le calcul de l'Omega Ratio (rendement minimum acceptable)
# 0 = on veut au moins ne pas perdre d'argent
OMEGA_THRESHOLD = 0.0

# Couleurs pour les graphiques
COLORS = [
    "#1f77b4",  # bleu
    "#ff7f0e",  # orange
    "#2ca02c",  # vert
    "#d62728",  # rouge
    "#9467bd",  # violet
    "#8c564b",  # marron
    "#e377c2",  # rose
    "#7f7f7f",  # gris
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# Couleur du benchmark (pour le distinguer des fonds)
BENCHMARK_COLOR = "#000000"  # noir
