# ==============================================================================
# DATA LOADER - Chargement et préparation des données
# ==============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PATH, DATE_COLUMN, BENCHMARK_COLUMN


@st.cache_data
def load_data(file_path: str = None, uploaded_file=None) -> pd.DataFrame:
    """
    Charge les données depuis un fichier Excel ou CSV.
    
    Parameters
    ----------
    file_path : str, optional
        Chemin vers le fichier de données
    uploaded_file : UploadedFile, optional
        Fichier uploadé via Streamlit
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec Date en index, colonnes = fonds + benchmark
    """
    # Déterminer la source des données
    if uploaded_file is not None:
        source = uploaded_file
        file_name = uploaded_file.name
    elif file_path is not None:
        source = file_path
        file_name = file_path
    else:
        source = DATA_PATH
        file_name = DATA_PATH
    
    # Charger selon le format
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        df = pd.read_excel(source)
    elif file_name.endswith('.csv'):
        # Essayer plusieurs séparateurs
        try:
            df = pd.read_csv(source, sep=';')
            if len(df.columns) == 1:
                df = pd.read_csv(source, sep=',')
        except:
            df = pd.read_csv(source)
    else:
        raise ValueError(f"Format non supporté: {file_name}")
    
    # Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip()
    
    # Identifier et parser la colonne date
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'dates', 'dt', 'datetime']:
            date_col = col
            break
    
    if date_col is None:
        date_col = df.columns[0]  # Première colonne par défaut
    
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.set_index(date_col)
    df.index.name = 'Date'
    
    # Trier par date
    df = df.sort_index()
    
    # Convertir en numérique (gérer les virgules comme séparateur décimal)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def get_fund_columns(df: pd.DataFrame, benchmark_col: str = None) -> list:
    """
    Retourne la liste des colonnes de fonds (exclut le benchmark).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des données
    benchmark_col : str, optional
        Nom de la colonne benchmark
    
    Returns
    -------
    list
        Liste des noms de colonnes des fonds
    """
    if benchmark_col is None:
        benchmark_col = BENCHMARK_COLUMN
    
    # Chercher la colonne benchmark (insensible à la casse)
    benchmark_found = None
    for col in df.columns:
        if col.lower() == benchmark_col.lower():
            benchmark_found = col
            break
    
    if benchmark_found:
        return [col for col in df.columns if col != benchmark_found]
    else:
        return list(df.columns)


def get_valid_date_range(df: pd.DataFrame, selected_funds: list) -> tuple:
    """
    Retourne la plage de dates valide pour les fonds sélectionnés.
    
    La date de début est la plus récente des premières dates valides
    de tous les fonds sélectionnés.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des données
    selected_funds : list
        Liste des fonds sélectionnés
    
    Returns
    -------
    tuple
        (date_min, date_max)
    """
    if not selected_funds:
        return df.index.min(), df.index.max()
    
    # Trouver la première date valide pour chaque fonds
    first_valid_dates = []
    for fund in selected_funds:
        if fund in df.columns:
            first_valid = df[fund].first_valid_index()
            if first_valid is not None:
                first_valid_dates.append(first_valid)
    
    if not first_valid_dates:
        return df.index.min(), df.index.max()
    
    # La date de début est le MAX des premières dates (le plus récent)
    date_min = max(first_valid_dates)
    date_max = df.index.max()
    
    return date_min, date_max


def filter_data_by_period(
    df: pd.DataFrame, 
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp,
    selected_funds: list = None
) -> pd.DataFrame:
    """
    Filtre les données par période et fonds sélectionnés.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des données
    start_date : pd.Timestamp
        Date de début
    end_date : pd.Timestamp
        Date de fin
    selected_funds : list, optional
        Liste des fonds à garder (si None, garde tout)
    
    Returns
    -------
    pd.DataFrame
        DataFrame filtré
    """
    # Filtrer par date
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask].copy()
    
    # Filtrer par colonnes si spécifié
    if selected_funds:
        # Garder aussi le benchmark s'il existe
        cols_to_keep = list(selected_funds)
        for col in df.columns:
            if col.lower() == BENCHMARK_COLUMN.lower() and col not in cols_to_keep:
                cols_to_keep.append(col)
        df_filtered = df_filtered[cols_to_keep]
    
    return df_filtered


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements quotidiens à partir des prix.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des prix
    
    Returns
    -------
    pd.DataFrame
        DataFrame des rendements (en décimal, pas en %)
    """
    returns = df.pct_change().dropna()
    return returns


def get_fund_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les métadonnées de chaque fonds.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des prix
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec première date, dernière date, nb observations par fonds
    """
    metadata = []
    
    for col in df.columns:
        first_valid = df[col].first_valid_index()
        last_valid = df[col].last_valid_index()
        n_obs = df[col].notna().sum()
        
        metadata.append({
            'Fonds': col,
            'Premiere_Date': first_valid,
            'Derniere_Date': last_valid,
            'Nb_Observations': n_obs
        })
    
    return pd.DataFrame(metadata)
