"""
Script d'initialisation de la base de données
"""
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime


CSV_FILE_PATH = 'C:/Users/Pc-Marie/Documents/MASTER_APE/S3/Advancing_programming/projet_supply_chain/Projet/data/data.csv'


def setup_database(csv_path=CSV_FILE_PATH):
    """
    Initialise et valide la base de données supply chain.
    
    Args:
        csv_path: Chemin vers le fichier CSV
        
    Returns:
        pd.DataFrame: Données chargées et validées
    """
    try:
        # Charger les données
        print(f"📂 Chargement de {csv_path}...")
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
        
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        raise

if __name__ == "__main__":
    # Test du setup
    data = setup_database()
    print(f"\n📊 Aperçu des données:")
    print(data.head())


class DatabaseManager:
    """Gestionnaire de base de données pour la supply chain."""
    
    def __init__(self, data):
        """
        Initialise le gestionnaire avec les données.
        
        Args:
            data: DataFrame pandas contenant les données
        """
        self.data = data.copy()
        
        # Assurer que la colonne 'is_stockout' existe
        if 'is_stockout' not in self.data.columns:
            self.data['is_stockout'] = (self.data['current_stock_level'] == 0).astype(int)
        
        # Convertir la date si ce n'est pas déjà fait
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        # S'assurer que les données sont triées par date
        self.data = self.data.sort_values('date').reset_index(drop=True)
    
    def get_inventory_data(self, product=None, period_days=90):
        """
        Extrait les données historiques de stock et ventes.
        
        Args:
            product: Nom du produit (None pour tous)
            period_days: Nombre de jours à extraire
            
        Returns:
            pd.DataFrame: Données filtrées
        """
        df = self.data.copy()
        
        # Filtrer par produit si spécifié
        if product:
            df = df[df['Product type'] == product]
        
        # Filtrer par période
        if len(df) > 0:
            end_date = df['date'].max()
            start_date = end_date - timedelta(days=period_days)
            df = df[df['date'] >= start_date]
        
        return df.reset_index(drop=True)
    
    def get_product_stats(self, product, period_days=30):
        """
        Calcule les statistiques pour un produit.
        
        Args:
            product: Nom du produit
            period_days: Période d'analyse
            
        Returns:
            dict: Statistiques du produit
        """
        df = self.get_inventory_data(product, period_days)
        
        if len(df) == 0:
            return None
        
        stats = {
            'total_sales': df['daily_sold_units'].sum(),
            'avg_daily_sales': df['daily_sold_units'].mean(),
            'current_stock': df['current_stock_level'].iloc[-1],
            'min_stock': df['current_stock_level'].min(),
            'max_stock': df['current_stock_level'].max(),
            'stockout_days': (df['is_stockout'] == 1).sum() if 'is_stockout' in df.columns else 0
        }
        
        # Ajouter des stats supplémentaires si disponibles
        if 'Revenue generated' in df.columns:
            stats['total_revenue'] = df['Revenue generated'].sum()
        
        if 'Price' in df.columns:
            stats['avg_price'] = df['Price'].mean()
        
        if 'daily_production_units' in df.columns:
            stats['total_production'] = df['daily_production_units'].sum()
        
        return stats
    
    def get_all_products(self):
        """
        Retourne la liste de tous les produits.
        
        Returns:
            list: Liste des noms de produits
        """
        return self.data['Product type'].unique().tolist()
    
    def get_date_range(self):
        """
        Retourne la période couverte par les données.
        
        Returns:
            dict: Informations sur la période
        """
        return {
            'start': self.data['date'].min(),
            'end': self.data['date'].max(),
            'days': (self.data['date'].max() - self.data['date'].min()).days
        }
    
    def __repr__(self):
        """Représentation textuelle du gestionnaire."""
        return (f"DatabaseManager(records={len(self.data)}, "
                f"products={self.data['Product type'].nunique()}, "
                f"period={self.data['date'].min()} to {self.data['date'].max()})")
    
    def __len__(self):
        """Retourne le nombre d'enregistrements."""
        return len(self.data)