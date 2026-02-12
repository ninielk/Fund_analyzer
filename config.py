# ==============================================================================
# CONFIGURATION DU DASHBOARD - V2
# ==============================================================================

# Nombre de jours de trading par an (pour annualisation)
TRADING_DAYS_PER_YEAR = 252

# Seuil pour le calcul de l'Omega Ratio
OMEGA_THRESHOLD = 0.0

# Tracking Error - seuils acceptables
TRACKING_ERROR_MIN = 2.0  # %
TRACKING_ERROR_MAX = 5.0  # %

# Couleurs charte graphique
COLORS = [
    "#B10967",  # rose/magenta
    "#412761",  # violet fonce
    "#007078",  # bleu-vert
    "#F8AF00",  # orange/jaune
    "#99DBF2",  # bleu clair
    "#D3E8CA",  # vert clair
    "#6E6E6E",  # gris
    "#243D7C",  # bleu fonce
]

# Couleur du benchmark
BENCHMARK_COLOR = "#6E6E6E"

# Degradés pour les graphiques
COLOR_GRADIENT_POSITIVE = ["#D3E8CA", "#007078"]  # vert clair -> bleu-vert
COLOR_GRADIENT_NEGATIVE = ["#F8AF00", "#B10967"]  # jaune -> rose
COLOR_GRADIENT_NEUTRAL = ["#99DBF2", "#243D7C"]   # bleu clair -> bleu fonce

# Couleurs pour le radar chart
RADAR_FILL_COLOR = "rgba(177, 9, 103, 0.3)"   # rose transparent
RADAR_LINE_COLOR = "#B10967"                    # rose
