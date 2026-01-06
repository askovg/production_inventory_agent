"""
Script principal - Agent Supply Chain avec Hugging Face
"""
import sys
import warnings
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from transformers import pipeline
from src.agent_tools import DatabaseManager, AnalysisEngine, Visualizer, ReportGenerator
from scripts.setup_database import setup_database


class SupplyChainAgentHF:
    """
    Agent d'analyse de supply chain avec Hugging Face pour l'analyse avancée,
    prévision, détection d'anomalies et optimisation des stocks.
    """
    
    def __init__(self, csv_file='enriched_supply_chain_data.csv'):
        """
        Initialise l'agent avec le fichier de données et les modèles HF.
        
        Args:
            csv_file: Chemin vers le fichier CSV
        """
        print("🤗 Initialisation de l'Agent Supply Chain avec Hugging Face\n")
        
        # Charger les données
        data = setup_database(csv_file)
        
        # Initialiser les composants
        self.db = DatabaseManager(data)
        
        # Initialiser les modèles HF
        hf_models = self._initialize_hf_models()
        
        # Initialiser les modules
        self.analysis = AnalysisEngine(self.db, hf_models)
        self.viz = Visualizer(self.db, self.analysis)
        self.reports = ReportGenerator(self.db, self.analysis)
        
        print("\n✅ Agent initialisé avec succès!\n")
    
    def _initialize_hf_models(self):
        """Initialise les modèles Hugging Face."""
        print("\n🤗 Initialisation des modèles Hugging Face...")
        hf_models = {}
        
        try:
            # 1. Modèle d'analyse de sentiment
            print("  • Chargement du modèle d'analyse de sentiment...")
            hf_models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # 2. Modèle de génération de texte
            print("  • Chargement du modèle de génération de texte...")
            hf_models['generator'] = pipeline(
                "text-generation",
                model="gpt2",
                max_length=100
            )
            
            # 3. Modèle de classification
            print("  • Chargement du modèle de classification...")
            hf_models['classifier'] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            print("✅ Modèles Hugging Face chargés avec succès!")
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement des modèles HF: {e}")
            print("L'agent continuera avec les méthodes classiques.")
        
        return hf_models
    
    # ==================== MÉTHODES PRINCIPALES ====================
    
    def run_complete_analysis(self, product=None):
        """
        Lance une analyse complète.
        
        Args:
            product: Produit à analyser (None pour le premier produit)
        """
        if product is None:
            product = self.db.get_all_products()[0]
        
        print("\n" + "="*60)
        print(f"🔍 ANALYSE COMPLÈTE POUR: {product}")
        print("="*60)
        
        # 1. Analyse de sentiment
        print("\n1️⃣ ANALYSE DE SENTIMENT DU MARCHÉ")
        self.analysis.analyze_market_sentiment(product)
        
        # 2. Visualiser les stocks
        print("\n2️⃣ VISUALISATION DES STOCKS")
        self.viz.plot_inventory_levels(product, days=30)
        
        # 3. Prévisions
        print("\n3️⃣ PRÉVISIONS DE DEMANDE")
        self.viz.plot_demand_forecast(product, horizon=14, method='hf_enhanced')
        
        # 4. Anomalies
        print("\n4️⃣ DÉTECTION DES ANOMALIES")
        self.viz.plot_anomalies(product)
        
        # 5. Plan de réappro
        print("\n5️⃣ PLAN DE RÉAPPROVISIONNEMENT")
        restock = self.analysis.suggest_restock_plan()
        print("\n📊 Top 10 produits par urgence:")
        print(restock.head(10).to_string(index=False))
        
        # 6. Rapport
        print("\n6️⃣ GÉNÉRATION DU RAPPORT")
        self.reports.generate_inventory_report()
        
        print("\n✅ Analyse complète terminée!")
    
    def quick_status(self):
        """Affiche un résumé rapide du statut."""
        print("\n" + "="*60)
        print("📊 STATUT RAPIDE DE LA SUPPLY CHAIN")
        print("="*60)
        
        summary = self.reports.generate_summary_stats()
        
        print(f"\n📦 Produits: {summary['total_products']}")
        print(f"📈 Ventes totales: {summary['total_sales']:.0f} unités")
        print(f"📊 Ventes moy/jour: {summary['avg_daily_sales']:.2f} unités")
        print(f"🏪 Stock total: {summary['total_stock']:.0f} unités")
        print(f"⚠️ Ruptures de stock: {summary['stockout_incidents']}")
        
        # Plan de réappro urgent
        restock = self.analysis.suggest_restock_plan()
        urgent = restock[restock['urgency'] == 'urgent']
        
        if len(urgent) > 0:
            print(f"\n🚨 {len(urgent)} produits en urgence:")
            for _, item in urgent.head(3).iterrows():
                print(f"  • {item['product']}: {item['days_of_stock']:.1f} jours de stock")
        else:
            print("\n✅ Aucun produit en situation urgente")
        
        print("="*60)
    
    def analyze_product(self, product):
        """
        Analyse détaillée d'un produit.
        
        Args:
            product: Nom du produit
        """
        print(f"\n📦 Analyse de {product}")
        print("-" * 60)
        
        # Stats
        stats = self.db.get_product_stats(product, period_days=30)
        if stats:
            print(f"Ventes (30j): {stats['total_sales']:.0f} unités")
            print(f"Stock actuel: {stats['current_stock']:.0f} unités")
            print(f"Jours de stock: {stats['current_stock']/stats['avg_daily_sales']:.1f}")
        
        # Sentiment
        sentiment = self.analysis.analyze_market_sentiment(product)
        
        # Prévisions
        forecast = self.analysis.forecast_demand(product, horizon=7)
        if forecast is not None:
            print(f"\nPrévisions 7j: {forecast['predicted_demand'].sum():.0f} unités")
        
        # Rapport détaillé
        self.reports.generate_product_report(product, f"report_{product}.txt")


def main():
    """Fonction principale."""
    print("="*60)
    print("🤗 AGENT SUPPLY CHAIN AVEC HUGGING FACE")
    print("="*60)
    
    # Initialiser l'agent
    agent = SupplyChainAgentHF('enriched_supply_chain_data.csv')
    
    # Menu interactif
    while True:
        print("\n" + "="*60)
        print("MENU PRINCIPAL")
        print("="*60)
        print("1. Analyse complète (produit)")
        print("2. Statut rapide")
        print("3. Analyser un produit spécifique")
        print("4. Visualiser les stocks")
        print("5. Plan de réapprovisionnement")
        print("6. Générer rapport complet")
        print("7. Démo automatique")
        print("0. Quitter")
        
        choice = input("\nVotre choix: ").strip()
        
        if choice == '1':
            products = agent.db.get_all_products()
            print("\nProduits disponibles:")
            for i, p in enumerate(products[:10], 1):
                print(f"{i}. {p}")
            
            idx = input("\nNuméro du produit (ou Entrée pour le premier): ").strip()
            product = products[int(idx)-1] if idx.isdigit() and int(idx) <= len(products) else products[0]
            
            agent.run_complete_analysis(product)
        
        elif choice == '2':
            agent.quick_status()
        
        elif choice == '3':
            product = input("Nom du produit: ").strip()
            agent.analyze_product(product)
        
        elif choice == '4':
            product = input("Nom du produit: ").strip()
            agent.viz.plot_inventory_levels(product, days=30)
        
        elif choice == '5':
            restock = agent.analysis.suggest_restock_plan()
            print("\n📋 PLAN DE RÉAPPROVISIONNEMENT")
            print("="*60)
            print(restock.to_string(index=False))
            
            # Visualiser
            agent.viz.plot_restock_urgency(restock)
        
        elif choice == '6':
            agent.reports.generate_inventory_report('supply_chain_report_hf.txt')
        
        elif choice == '7':
            print("\n🎬 DÉMO AUTOMATIQUE")
            example_product = agent.db.get_all_products()[0]
            agent.run_complete_analysis(example_product)
        
        elif choice == '0':
            print("\n👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide")


if __name__ == "__main__":
    main()