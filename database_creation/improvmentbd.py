import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def build_enriched_dataset(input_csv):
    df_orig = pd.read_csv(input_csv)
    n_days = 90
    start_date = datetime(2023, 10, 1)
    date_list = [start_date + timedelta(days=x) for x in range(n_days)]
    weather_options = ['Sunny', 'Rainy', 'Cloudy', 'Stormy']
    
    all_data = []
    idx = 1
    
    for _, row in df_orig.iterrows():
        base = row.to_dict()
        avg_sales = row['Number of products sold'] / n_days
        curr_stock = row['Stock levels']
        
        for d in date_list:
            # Simulation météo et ventes
            weather = np.random.choice(weather_options, p=[0.5, 0.2, 0.2, 0.1])
            temp = np.random.uniform(5, 30)
            
            # Simulation demande avec un léger aléatoire
            sold = np.random.poisson(avg_sales)
            
            # Logique de réapprovisionnement automatique
            prod = row['Production volumes'] if curr_stock < (row['Stock levels'] * 0.4) else 0
            
            actual_sold = min(sold, curr_stock + prod)
            curr_stock = (curr_stock + prod) - actual_sold
            
            # Fusion des données originales + nouvelles variables
            new_vars = {
                'inventory_id': idx,
                'date': d.strftime('%Y-%m-%d'),
                'current_stock_level': int(curr_stock),
                'daily_sold_units': int(actual_sold),
                'daily_production_units': int(prod),
                'temp_celsius': round(temp, 1),
                'weather_condition': weather,
                'daily_holding_cost': round(curr_stock * (row['Price'] * 0.001), 2),
                'is_stockout': 1 if curr_stock == 0 else 0,
                'promotion_active': np.random.choice([0, 1], p=[0.9, 0.1])
            }
            all_data.append({**base, **new_vars})
            idx += 1
            
    df_final = pd.DataFrame(all_data)
    df_final.to_csv('enriched_supply_chain_data.csv', index=False)
    print("Fichier 'enriched_supply_chain_data.csv' créé avec succès !")

# Lancement (assurez-vous que supply_chain_data.csv est dans le dossier)
build_enriched_dataset('supply_chain_data.csv')