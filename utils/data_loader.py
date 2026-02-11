# ==============================================================================
# DATA LOADER - Chargement et préparation des données
# ==============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PATH, DATE_COLUMN


@st.cache_data
def load_data(uploaded_file=None) -> pd.DataFrame:
    """
    Charge les données depuis un fichier Excel ou CSV uploadé.
    """
    if uploaded_file is None:
        raise ValueError("Aucun fichier fourni")
    
    file_name = uploaded_file.name
    
    # Charger selon le format
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        df = pd.read_excel(uploaded_file)
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
        date_col = df.columns[0]
    
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.set_index(date_col)
    df.index.name = 'Date'
    
    # Trier par date
    df = df.sort_index()
    
    # Convertir en numérique
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def get_valid_date_range(df: pd.DataFrame, selected_funds: list) -> tuple:
    """
    Retourne la plage de dates valide pour les fonds sélectionnés.
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
    Filtre les données par période et colonnes sélectionnées.
    """
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask].copy()
    
    if selected_columns:
        cols_to_keep = [c for c in selected_columns if c in df.columns]
        df_filtered = df_filtered[cols_to_keep]
    
    return df_filtered


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements quotidiens à partir des prix.
    """
    returns = df.pct_change().dropna()
    return returns


def get_fund_inception_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne la date d'inception (première date valide) de chaque colonne.
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