# ==============================================================================
# DATA LOADER - Chargement et preparation des donnees V2
# ==============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st


@st.cache_data
def load_data(uploaded_file=None) -> pd.DataFrame:
    """
    Charge les donnees depuis un fichier Excel ou CSV uploade (feuille 1).
    """
    if uploaded_file is None:
        raise ValueError("Aucun fichier fourni")
    
    file_name = uploaded_file.name
    
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        df = pd.read_excel(uploaded_file, sheet_name=0)
    elif file_name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, sep=';')
            if len(df.columns) == 1:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
    else:
        raise ValueError(f"Format non supporte: {file_name}")
    
    df.columns = df.columns.str.strip()
    
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'dates', 'dt', 'datetime']:
            date_col = col
            break
    
    if date_col is None:
        date_col = df.columns[0]
    
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.set_index(date_col)
    df.index.name = 'Date'
    df = df.sort_index()
    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


@st.cache_data
def load_fund_metadata(uploaded_file, sheet_name=1) -> pd.DataFrame:
    """
    Charge les metadonnees des fonds depuis la 2eme feuille Excel.
    
    Format attendu de la feuille 2:
    - Premiere colonne : nom de la metrique (ex: "ESG Score", "Frais", etc.)
    - Colonnes suivantes : une colonne par fonds avec le code du fonds en header
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec les metriques en index et les fonds en colonnes
    """
    if uploaded_file is None:
        return None
    
    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        df.columns = df.columns.str.strip()
        
        metric_col = df.columns[0]
        df = df.set_index(metric_col)
        df.index.name = 'Metrique'
        
        # Nettoyer les noms d'index
        df.index = df.index.astype(str).str.strip()
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        return None


def match_fund_metadata(indicators_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les indicateurs calcules avec les metadonnees (ESG, Frais).
    Match les fonds par nom exact.
    """
    if metadata_df is None or metadata_df.empty:
        return indicators_df
    
    result = indicators_df.copy()
    
    # Mapping des noms de metriques
    metric_mapping = {
        'HB Fund Overall ESG Pillar Score': 'ESG Score',
        'Fund Manager Stated Fee': 'Frais (%)',
        'ESG': 'ESG Score',
        'Frais': 'Frais (%)',
    }
    
    for idx, row in result.iterrows():
        fund_name = row['Fonds']
        
        if fund_name in metadata_df.columns:
            for metric in metadata_df.index:
                value = metadata_df.loc[metric, fund_name]
                if pd.notna(value):
                    # Utiliser le nom mappe si disponible
                    metric_name = metric_mapping.get(metric, metric)
                    result.loc[idx, metric_name] = value
    
    return result


def get_valid_date_range(df: pd.DataFrame, selected_funds: list) -> tuple:
    """
    Retourne la plage de dates valide pour les fonds selectionnes.
    """
    if not selected_funds:
        return df.index.min(), df.index.max()
    
    first_valid_dates = []
    for fund in selected_funds:
        if fund in df.columns:
            first_valid = df[fund].first_valid_index()
            if first_valid is not None:
                first_valid_dates.append(first_valid)
    
    if not first_valid_dates:
        return df.index.min(), df.index.max()
    
    date_min = max(first_valid_dates)
    date_max = df.index.max()
    
    return date_min, date_max


def filter_data_by_period(
    df: pd.DataFrame, 
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp,
    selected_columns: list = None
) -> pd.DataFrame:
    """
    Filtre les donnees par periode et colonnes selectionnees.
    """
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask].copy()
    
    if selected_columns:
        cols_to_keep = [c for c in selected_columns if c in df.columns]
        df_filtered = df_filtered[cols_to_keep]
    
    return df_filtered


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements quotidiens a partir des prix.
    """
    returns = df.pct_change().dropna()
    return returns


def get_fund_inception_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne la date d'inception (premiere date valide) de chaque colonne.
    """
    inception_dates = []
    
    for col in df.columns:
        first_valid = df[col].first_valid_index()
        n_obs = df[col].notna().sum()
        
        inception_dates.append({
            'Colonne': col,
            'Date Inception': first_valid,
            'Nb Observations': n_obs
        })
    
    return pd.DataFrame(inception_dates)
