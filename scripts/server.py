"""
Script principal - Agent Supply Chain avec Hugging Face
"""
import sys
import warnings
from pathlib import Path

# ==================== CONFIGURATION DES CHEMINS ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AGENT_TOOLS_DIR = PROJECT_ROOT / "src" / "agent_tools"

# Ajouter tous les chemins nécessaires
for path in [PROJECT_ROOT, AGENT_TOOLS_DIR, SCRIPT_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

warnings.filterwarnings('ignore')

# ==================== IMPORTS ====================
# Imports locaux
from database import DatabaseManager
from analysis import AnalysisEngine
from visualization import Visualizer
from reports import ReportGenerator
from setup_database import setup_database

# Imports Hugging Face (optionnels)
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers non installé - Mode classique activé")
    HF_AVAILABLE = False
    pipeline = None


class SupplyChainAgentHF:
    """
    Agent d'analyse de supply chain avec Hugging Face pour l'analyse avancée,
    prévision, détection d'anomalies et optimisation des stocks.
    """
    
    def __init__(self, csv_file=None):
        """
        Initialise l'agent avec le fichier de données et les modèles HF.
        
        Args:
            csv_file: Chemin vers le fichier CSV (None = chemin par défaut)
        """
        print("🤗 Initialisation de l'Agent Supply Chain avec Hugging Face\n")
        
        # Charger les données
        if csv_file:
            data = setup_database(csv_file)
        else:
            data = setup_database()  # Utilise le chemin par défaut
        
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
        if not HF_AVAILABLE:
            print("⚠️ Hugging Face non disponible - Mode classique\n")
            return {}
        
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
            hf_models = {}
        
        return hf_models
    
    # ==================== MÉTHODES PRINCIPALES ====================
    
    def run_complete_analysis(self, product=None):
        """
        Lance une analyse complète.
        
        Args:
            product: Produit à analyser (None pour le premier produit)
        """
        if product is None:
            products = self.db.get_all_products()
            if len(products) == 0:
                print("❌ Aucun produit disponible")
                return
            product = products[0]
        
        print("\n" + "="*60)
        print(f"🔍 ANALYSE COMPLÈTE POUR: {product}")
        print("="*60)
        
        # 1. Analyse de sentiment
        print("\n1️⃣ ANALYSE DE SENTIMENT DU MARCHÉ")
        try:
            self.analysis.analyze_market_sentiment(product)
        except Exception as e:
            print(f"⚠️ Sentiment analysis non disponible: {e}")
        
        # 2. Visualiser les stocks
        print("\n2️⃣ VISUALISATION DES STOCKS")
        try:
            self.viz.plot_inventory_levels(product, days=30)
        except Exception as e:
            print(f"❌ Erreur visualisation: {e}")
        
        # 3. Prévisions
        print("\n3️⃣ PRÉVISIONS DE DEMANDE")
        try:
            self.viz.plot_demand_forecast(product, horizon=14, method='prophet')
        except Exception as e:
            print(f"❌ Erreur prévisions: {e}")
        
        # 4. Anomalies
        print("\n4️⃣ DÉTECTION DES ANOMALIES")
        try:
            self.viz.plot_anomalies(product)
        except Exception as e:
            print(f"❌ Erreur détection anomalies: {e}")
        
        # 5. Plan de réappro
        print("\n5️⃣ PLAN DE RÉAPPROVISIONNEMENT")
        try:
            restock = self.analysis.suggest_restock_plan()
            print("\n📊 Top 10 produits par urgence:")
            print(restock.head(10).to_string(index=False))
        except Exception as e:
            print(f"❌ Erreur plan réappro: {e}")
        
        # 6. Rapport
        print("\n6️⃣ GÉNÉRATION DU RAPPORT")
        try:
            self.reports.generate_inventory_report()
        except Exception as e:
            print(f"❌ Erreur génération rapport: {e}")
        
        print("\n✅ Analyse complète terminée!")
    
    def quick_status(self):
        """Affiche un résumé rapide du statut."""
        print("\n" + "="*60)
        print("📊 STATUT RAPIDE DE LA SUPPLY CHAIN")
        print("="*60)
        
        try:
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
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
        
        print("="*60)
    
    def analyze_product(self, product):
        """
        Analyse détaillée d'un produit.
        
        Args:
            product: Nom du produit
        """
        print(f"\n📦 Analyse de {product}")
        print("-" * 60)
        
        try:
            # Stats
            stats = self.db.get_product_stats(product, period_days=30)
            if stats:
                print(f"Ventes (30j): {stats['total_sales']:.0f} unités")
                print(f"Stock actuel: {stats['current_stock']:.0f} unités")
                if stats['avg_daily_sales'] > 0:
                    print(f"Jours de stock: {stats['current_stock']/stats['avg_daily_sales']:.1f}")
            
            # Sentiment
            try:
                self.analysis.analyze_market_sentiment(product)
            except:
                pass
            
            # Prévisions
            try:
                forecast = self.analysis.forecast_demand(product, horizon=7)
                if forecast is not None:
                    print(f"\nPrévisions 7j: {forecast['predicted_demand'].sum():.0f} unités")
            except:
                pass
            
            # Rapport détaillé
            self.reports.generate_product_report(product, f"report_{product}.txt")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")


def main():
    """Fonction principale."""
    print("="*60)
    print("🤗 AGENT SUPPLY CHAIN AVEC HUGGING FACE")
    print("="*60)
    
    # Initialiser l'agent
    try:
        agent = SupplyChainAgentHF()
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        print("\n📝 Vérifiez:")
        print("  1. Le fichier CSV existe")
        print("  2. Les colonnes requises sont présentes")
        print("  3. Les dépendances sont installées")
        return
    
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
            
            choice_input = input("\nNuméro du produit (ou Entrée pour le premier): ").strip()
            
            # Gérer la sélection
            if choice_input.isdigit():
                idx = int(choice_input)
                if 1 <= idx <= len(products):
                    product = products[idx - 1]
                else:
                    product = products[0]
            elif choice_input in products:
                # Si l'utilisateur tape le nom du produit
                product = choice_input
            else:
                product = products[0]
            
            agent.run_complete_analysis(product)
        
        elif choice == '2':
            agent.quick_status()
        
        elif choice == '3':
            products = agent.db.get_all_products()
            print("\nProduits disponibles:")
            for i, p in enumerate(products, 1):
                print(f"{i}. {p}")
            
            product = input("\nNom du produit: ").strip()
            if product in products:
                agent.analyze_product(product)
            else:
                print(f"❌ Produit '{product}' non trouvé")
        
        elif choice == '4':
            products = agent.db.get_all_products()
            print("\nProduits disponibles:", ", ".join(products))
            product = input("\nNom du produit: ").strip()
            
            if product in products:
                try:
                    agent.viz.plot_inventory_levels(product, days=30)
                except Exception as e:
                    print(f"❌ Erreur: {e}")
            else:
                print(f"❌ Produit '{product}' non trouvé")
        
        elif choice == '5':
            try:
                restock = agent.analysis.suggest_restock_plan()
                print("\n📋 PLAN DE RÉAPPROVISIONNEMENT")
                print("="*60)
                print(restock.to_string(index=False))
                
                # Visualiser
                visualize = input("\nVisualiser graphiquement? (o/n): ").strip().lower()
                if visualize == 'o':
                    agent.viz.plot_restock_urgency(restock)
            except Exception as e:
                print(f"❌ Erreur: {e}")
        
        elif choice == '6':
            try:
                output_file = input("\nNom du fichier (ou Entrée pour 'supply_chain_report_hf.txt'): ").strip()
                if not output_file:
                    output_file = 'supply_chain_report_hf.txt'
                agent.reports.generate_inventory_report(output_file)
            except Exception as e:
                print(f"❌ Erreur: {e}")
        
        elif choice == '7':
            print("\n🎬 DÉMO AUTOMATIQUE")
            try:
                example_product = agent.db.get_all_products()[0]
                agent.run_complete_analysis(example_product)
            except Exception as e:
                print(f"❌ Erreur: {e}")
        
        elif choice == '0':
            print("\n👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interruption par l'utilisateur. Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()