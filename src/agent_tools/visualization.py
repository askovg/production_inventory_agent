"""
Outils de visualisation des données
"""
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

# Imports optionnels
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
    # Configuration du style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (16, 8)
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️ matplotlib/seaborn non disponibles. Installer avec: pip install matplotlib seaborn")

class Visualizer:
    """Gestionnaire de visualisations pour la supply chain."""
    
    def __init__(self, db_manager, analysis_engine):
        """
        Initialise le visualiseur.
        
        Args:
            db_manager: Instance de DatabaseManager
            analysis_engine: Instance d'AnalysisEngine
        """
        self.db = db_manager
        self.analysis = analysis_engine
        
        if not PLOT_AVAILABLE:
            print("⚠️ Visualisations désactivées (matplotlib non disponible)")
    
    def plot_inventory_levels(self, product, days=30, save_path=None):
        """
        Visualise l'évolution des stocks.
        
        Args:
            product: Nom du produit
            days: Nombre de jours à afficher
            save_path: Chemin pour sauvegarder le graphique
        """
        if not PLOT_AVAILABLE:
            print("❌ Visualisation non disponible (matplotlib manquant)")
            return
        
        df = self.db.get_inventory_data(product, period_days=days)
        
        if len(df) == 0:
            print(f"❌ Pas de données pour {product}")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Graphique des stocks
        ax1.plot(df['date'], df['current_stock_level'], 
                label='Niveau de stock', linewidth=2, color='#3b82f6')
        ax1.fill_between(df['date'], df['current_stock_level'], 
                         alpha=0.3, color='#3b82f6')
        ax1.set_title(f'Évolution du Stock - {product} 🤗', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Stock (unités)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Graphique des ventes
        ax2.bar(df['date'], df['daily_sold_units'], 
               label='Ventes quotidiennes', color='#10b981', alpha=0.7)
        ax2.set_title('Ventes Quotidiennes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Unités vendues', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_weekly_demand_forecast(self, product, horizon=14, method='prophet', save_path=None):
        """
        Graphique agrégé par semaine pour une meilleure lisibilité.
        SUPER SIMPLIFIÉ ET CLAIR.
        
        Args:
            product: Nom du produit
            horizon: Nombre de jours à prévoir
            method: Méthode de prévision
            save_path: Chemin pour sauvegarder
        """
        if not PLOT_AVAILABLE:
            print("❌ Visualisation non disponible")
            return
        
        print("\n🔄 Génération du graphique hebdomadaire...")
        
        # Données historiques sur 8 semaines
        historical = self.db.get_inventory_data(product, period_days=56)
        
        # Prévisions
        forecast = self.analysis.forecast_demand(product, horizon, method)
        
        if forecast is None or len(historical) == 0:
            print("❌ Pas assez de données")
            return
        
        # ========== AGRÉGATION PAR SEMAINE ==========
        print("📊 Agrégation des données par semaine...")
        
        # Historique
        hist_copy = historical.copy()
        hist_copy['week_start'] = hist_copy['date'] - pd.to_timedelta(hist_copy['date'].dt.dayofweek, unit='d')
        weekly_hist = hist_copy.groupby('week_start').agg({
            'daily_sold_units': 'sum'
        }).reset_index()
        weekly_hist.columns = ['date', 'units']
        weekly_hist['type'] = 'Historique'
        
        # Prévisions
        fore_copy = forecast.copy()
        fore_copy['week_start'] = fore_copy['date'] - pd.to_timedelta(fore_copy['date'].dt.dayofweek, unit='d')
        weekly_fore = fore_copy.groupby('week_start').agg({
            'predicted_demand': 'sum'
        }).reset_index()
        weekly_fore.columns = ['date', 'units']
        weekly_fore['type'] = 'Prévision'
        
        # ========== GRAPHIQUE ==========
        fig, ax = plt.subplots(figsize=(18, 9))
        
        # Barres historiques (bleu)
        x_hist = np.arange(len(weekly_hist))
        bars_hist = ax.bar(x_hist, weekly_hist['units'], 
                          width=0.7, alpha=0.8, color='#3b82f6', 
                          label='📊 Ventes hebdomadaires (historique)',
                          edgecolor='black', linewidth=1.5)
        
        # Barres prévisions (violet)
        x_fore = np.arange(len(weekly_hist), len(weekly_hist) + len(weekly_fore))
        bars_fore = ax.bar(x_fore, weekly_fore['units'], 
                          width=0.7, alpha=0.8, color='#8b5cf6', 
                          label='🔮 Prévisions hebdomadaires',
                          edgecolor='black', linewidth=1.5)
        
        # Valeurs au-dessus des barres
        for bars in [bars_hist, bars_fore]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Ligne de tendance historique
        if len(weekly_hist) > 2:
            z = np.polyfit(x_hist, weekly_hist['units'], 1)
            p = np.poly1d(z)
            ax.plot(x_hist, p(x_hist), 
                   color='#1e40af', linewidth=3, linestyle='--', alpha=0.7,
                   label=f'Tendance historique')
        
        # Moyenne historique
        avg_hist = weekly_hist['units'].mean()
        ax.axhline(y=avg_hist, color='#059669', linestyle='--', linewidth=2.5,
                  label=f'Moyenne historique: {avg_hist:.0f} unités/semaine', alpha=0.7)
        
        # Séparation visuelle
        separation = len(weekly_hist) - 0.5
        ax.axvline(x=separation, color='#dc2626', linestyle=':', linewidth=3, alpha=0.7,
                  label='➤ Aujourd\'hui')
        
        # Zones colorées
        ax.axvspan(-0.5, separation, alpha=0.08, color='#3b82f6', zorder=0)
        ax.axvspan(separation, len(weekly_hist) + len(weekly_fore) - 0.5, 
                  alpha=0.08, color='#8b5cf6', zorder=0)
        
        # ========== LABELS ET TITRE ==========
        # Créer les labels de semaines
        all_weeks = pd.concat([weekly_hist[['date']], weekly_fore[['date']]])
        week_labels = [f"S{i+1}\n{date.strftime('%d/%m')}" 
                      for i, date in enumerate(all_weeks['date'])]
        
        ax.set_xticks(range(len(all_weeks)))
        ax.set_xticklabels(week_labels, fontsize=11)
        
        ax.set_title(f'📊 Demande Hebdomadaire - {product}', 
                    fontsize=20, fontweight='bold', pad=25)
        ax.set_xlabel('Semaine (date de début)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Unités vendues (total semaine)', fontsize=14, fontweight='bold')
        
        # Légende
        legend = ax.legend(fontsize=13, loc='upper left', framealpha=0.95,
                          shadow=True, fancybox=True)
        legend.get_frame().set_facecolor('#f8f9fa')
        
        # Grille
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
        ax.set_ylim(bottom=0, top=max(all_weeks['units'] if 'units' not in all_weeks.columns 
                                     else pd.concat([weekly_hist['units'], weekly_fore['units']])) * 1.15)
        
        # Statistiques
        total_hist = weekly_hist['units'].sum()
        total_fore = weekly_fore['units'].sum()
        variation = ((weekly_fore['units'].mean() - avg_hist) / avg_hist * 100) if avg_hist > 0 else 0
        
        stats_text = (
            f"📊 Historique: {total_hist:.0f} unités ({len(weekly_hist)} semaines) | "
            f"🔮 Prévisions: {total_fore:.0f} unités ({len(weekly_fore)} semaines) | "
            f"📈 Variation attendue: {variation:+.1f}%"
        )
        
        fig.text(0.5, 0.01, stats_text, ha='center', fontsize=12, 
                style='italic', weight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#e0e7ff', 
                         alpha=0.8, edgecolor='#6366f1', linewidth=2))
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Graphique sauvegardé: {save_path}")
        
        print("✅ Graphique généré avec succès!")
        plt.show()
    
    def plot_demand_forecast(self, product, horizon=14, method='hf_enhanced', save_path=None):
        """
        Compare prévisions et ventes réelles.
        
        Args:
            product: Nom du produit
            horizon: Nombre de jours à prévoir
            method: Méthode de prévision
            save_path: Chemin pour sauvegarder le graphique
        """
        historical = self.db.get_inventory_data(product, period_days=30)
        forecast = self.analysis.forecast_demand(product, horizon, method)
        
        if forecast is None:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Ventes réelles
        ax.plot(historical['date'], historical['daily_sold_units'], 
               label='Ventes réelles', linewidth=2, color='#3b82f6', marker='o')
        
        # Prévisions
        ax.plot(forecast['date'], forecast['predicted_demand'], 
               label=f'Prévisions ({forecast["method"].iloc[0]})', linewidth=2, 
               color='#8b5cf6', linestyle='--', marker='s')
        
        # Intervalle de confiance
        ax.fill_between(forecast['date'], 
                       forecast['lower_bound'], 
                       forecast['upper_bound'],
                       alpha=0.2, color='#8b5cf6', label='Intervalle de confiance')
        
        ax.set_title(f'Prévisions de Demande - {product} 🤗', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Unités', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_anomalies(self, product=None, save_path=None):
        """
        Met en évidence les anomalies détectées.
        
        Args:
            product: Produit à analyser (None pour tous)
            save_path: Chemin pour sauvegarder le graphique
        """
        df = self.db.get_inventory_data(product, period_days=60)
        anomalies = self.analysis.detect_stock_anomalies(product)
        
        if anomalies is None or len(anomalies) == 0:
            print("Aucune anomalie à afficher")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Ligne de stock
        ax.plot(df['date'], df['current_stock_level'], 
               linewidth=2, color='#3b82f6', label='Stock')
        
        # Marqueurs d'anomalies
        severity_colors = {
            'critical': '#ef4444',
            'danger': '#f97316',
            'warning': '#eab308'
        }
        
        for severity, color in severity_colors.items():
            anom = anomalies[anomalies['severity'] == severity]
            if len(anom) > 0:
                ax.scatter(anom['date'], anom['stock_level'], 
                          s=150, color=color, marker='X', 
                          label=f'{severity.capitalize()}', zorder=5, 
                          edgecolors='black', linewidths=1.5)
        
        ax.set_title('Détection des Anomalies de Stock 🤗', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Niveau de stock', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_product_comparison(self, products=None, metric='daily_sold_units', save_path=None):
        """
        Compare plusieurs produits.
        
        Args:
            products: Liste de produits (None pour top 5)
            metric: Métrique à comparer
            save_path: Chemin pour sauvegarder
        """
        if products is None:
            all_products = self.db.get_all_products()
            products = all_products[:5]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for product in products:
            df = self.db.get_inventory_data(product, period_days=30)
            if len(df) > 0:
                ax.plot(df['date'], df[metric], label=product, linewidth=2, marker='o')
        
        ax.set_title(f'Comparaison des Produits - {metric}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_restock_urgency(self, restock_plan, save_path=None):
        """
        Visualise les urgences de réapprovisionnement.
        
        Args:
            restock_plan: DataFrame du plan de réappro
            save_path: Chemin pour sauvegarder
        """
        urgency_colors = {
            'urgent': '#ef4444',
            'high': '#f97316',
            'normal': '#10b981',
            'low': '#3b82f6'
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Trier par jours de stock
        plot_data = restock_plan.sort_values('days_of_stock').head(15)
        
        colors = [urgency_colors.get(u, '#gray') for u in plot_data['urgency']]
        
        ax.barh(plot_data['product'], plot_data['days_of_stock'], color=colors)
        ax.set_xlabel('Jours de stock restants', fontsize=12)
        ax.set_title('Urgence de Réapprovisionnement par Produit', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=urg.capitalize()) 
                          for urg, color in urgency_colors.items()]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Graphique sauvegardé: {save_path}")
        
        plt.show()