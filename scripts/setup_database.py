"""
Script d'initialisation de la base de données
"""
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime

# Configuration du chemin - MODIFIEZ CETTE LIGNE avec le vrai chemin de votre CSV
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Chercher automatiquement le CSV
def find_csv_file():
    """Cherche le fichier CSV dans plusieurs emplacements possibles."""
    possible_names = [
        'data.csv',
        'enriched_supply_chain_data.csv',
        'supply_chain_data.csv'
    ]
    
    possible_dirs = [
        PROJECT_ROOT / 'data',
        PROJECT_ROOT,
        SCRIPT_DIR,
    ]
    
    for directory in possible_dirs:
        for name in possible_names:
            csv_path = directory / name
            if csv_path.exists():
                print(f"✅ CSV trouvé: {csv_path}")
                return str(csv_path)
    
    # Si rien n'est trouvé, retourner le chemin par défaut
    return str(PROJECT_ROOT / 'data' / 'data.csv')

# Chemin par défaut
CSV_FILE_PATH = find_csv_file()


def setup_database(csv_path=None):
    """
    Initialise et valide la base de données supply chain.
    
    Args:
        csv_path: Chemin vers le fichier CSV (None = chemin par défaut)
        
    Returns:
        pd.DataFrame: Données chargées et validées
    """
    if csv_path is None:
        csv_path = CSV_FILE_PATH
    
    try:
        # Charger les données
        print(f"📂 Chargement de {csv_path}...")
        
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"❌ Fichier CSV non trouvé: {csv_path}\n"
                f"   Emplacements vérifiés:\n"
                f"   - {PROJECT_ROOT / 'data' / 'data.csv'}\n"
                f"   - {PROJECT_ROOT / 'enriched_supply_chain_data.csv'}\n"
                f"   Placez votre CSV dans l'un de ces emplacements."
            )
        
        data = pd.read_csv(csv_path)
        
        # Convertir la colonne date
        print("📅 Conversion des dates...")
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Valider les colonnes essentielles
        required_cols = ['Product type', 'date', 'current_stock_level', 'daily_sold_units']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        # Ajouter is_stockout si manquant
        if 'is_stockout' not in data.columns:
            print("➕ Ajout de la colonne 'is_stockout'...")
            data['is_stockout'] = (data['current_stock_level'] == 0).astype(int)
        
        # Gérer les valeurs manquantes
        print("🔧 Nettoyage des données...")
        
        # Remplir les NaN dans daily_sold_units avec 0
        if data['daily_sold_units'].isna().any():
            data['daily_sold_units'] = data['daily_sold_units'].fillna(0)
        
        # Remplir les NaN dans daily_production_units avec 0
        if 'daily_production_units' in data.columns:
            data['daily_production_units'] = data['daily_production_units'].fillna(0)
        
        # S'assurer que les valeurs sont positives
        data['current_stock_level'] = data['current_stock_level'].clip(lower=0)
        data['daily_sold_units'] = data['daily_sold_units'].clip(lower=0)
        
        print(f"✅ Base de données initialisée avec succès")
        print(f"   - {len(data):,} enregistrements")
        print(f"   - {data['Product type'].nunique()} produits: {data['Product type'].unique()[:5].tolist()}")
        print(f"   - Période: {data['date'].min()} à {data['date'].max()}")
        print(f"   - Colonnes: {len(data.columns)}")
        
        return data
        
    except FileNotFoundError as e:
        print(str(e))
        raise
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        raise


if __name__ == "__main__":
    # Test du setup
    print("="*70)
    print("🧪 TEST DE SETUP_DATABASE")
    print("="*70)
    
    try:
        data = setup_database()
        print(f"\n📊 Aperçu des données:")
        print(data.head())
        print(f"\n✅ Test réussi!")
    except Exception as e:
        print(f"\n❌ Test échoué: {e}")