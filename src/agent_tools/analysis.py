"""
Outils d'analyse et de prévision avec Hugging Face
"""
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import warnings
warnings.filterwarnings('ignore')

# Imports optionnels
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

class AnalysisEngine:
    """Moteur d'analyse avec prévisions et détection d'anomalies."""
    
    def __init__(self, db_manager, hf_models=None):
        """
        Initialise le moteur d'analyse.
        
        Args:
            db_manager: Instance de DatabaseManager
            hf_models: Dictionnaire des modèles Hugging Face
        """
        self.db = db_manager
        self.hf_models = hf_models or {}
    
    def forecast_demand(self, product, horizon=14, method='prophet'):
        """
        Prédit la demande future.
        
        Args:
            product: Nom du produit
            horizon: Nombre de jours à prévoir
            method: Méthode ('arima', 'prophet', 'hf_enhanced')
            
        Returns:
            pd.DataFrame: Prévisions
        """
        df = self.db.get_inventory_data(product, period_days=90)
        
        if len(df) < 20:
            print(f"⚠️ Pas assez de données pour {product}")
            return None
        
        if method == 'arima':
            return self._forecast_arima(df, horizon)
        elif method == 'prophet':
            return self._forecast_prophet(df, horizon)
        else:  # hf_enhanced
            return self._forecast_hf_enhanced(df, horizon, product)
    
    def _forecast_arima(self, df, horizon):
        """Prévision avec modèle ARIMA."""
        if not ARIMA_AVAILABLE:
            print("❌ statsmodels non installé. Installer avec: pip install statsmodels")
            return None
        
        try:
            # Préparer les données avec fréquence explicite
            sales_df = df[['date', 'daily_sold_units']].copy()
            sales_df = sales_df.set_index('date')
            sales_df = sales_df.fillna(0)
            
            # Définir la fréquence explicitement (quotidienne)
            sales_df.index = pd.DatetimeIndex(sales_df.index)
            sales_df = sales_df.asfreq('D', fill_value=0)
            
            sales = sales_df['daily_sold_units']
            
            # Modèle ARIMA avec paramètres plus stables
            import warnings
            warnings.filterwarnings('ignore')
            
            model = ARIMA(sales, order=(2, 1, 2))  # Paramètres plus simples
            fitted = model.fit(method='statespace')
            
            # Prévisions
            forecast = fitted.forecast(steps=horizon)
            forecast_obj = fitted.get_forecast(steps=horizon)
            conf_int = forecast_obj.conf_int()
            
            # Créer les dates futures
            last_date = df['date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                        periods=horizon, freq='D')
            
            results = pd.DataFrame({
                'date': future_dates,
                'predicted_demand': np.maximum(0, forecast.values),
                'lower_bound': np.maximum(0, conf_int.iloc[:, 0].values),
                'upper_bound': np.maximum(0, conf_int.iloc[:, 1].values),
                'method': 'ARIMA'
            })
            
            print(f"✅ Prévision ARIMA: moyenne {results['predicted_demand'].mean():.1f} unités/jour")
            return results
            
        except Exception as e:
            print(f"❌ Erreur ARIMA: {e}")
            print("   Passage à la méthode simple...")
            return self._forecast_simple(df, horizon)
    
    def _forecast_prophet(self, df, horizon):
        """Prévision avec Prophet."""
        if not PROPHET_AVAILABLE:
            print("❌ Prophet non installé. Installer avec: pip install prophet")
            return None
        
        try:
            prophet_df = df[['date', 'daily_sold_units']].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.fillna(0)
            
            # Configuration Prophet plus robuste
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            if 'temp_celsius' in df.columns and df['temp_celsius'].notna().any():
                prophet_df['temp'] = df['temp_celsius'].fillna(df['temp_celsius'].mean())
                model.add_regressor('temp')
            
            # Supprimer les logs de Prophet
            import logging
            logging.getLogger('prophet').setLevel(logging.ERROR)
            
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=horizon, freq='D')
            
            if 'temp' in prophet_df.columns:
                future['temp'] = prophet_df['temp'].mean()
            
            forecast = model.predict(future)
            forecast = forecast.tail(horizon)
            
            results = pd.DataFrame({
                'date': forecast['ds'].values,
                'predicted_demand': forecast['yhat'].clip(lower=0).values,
                'lower_bound': forecast['yhat_lower'].clip(lower=0).values,
                'upper_bound': forecast['yhat_upper'].clip(lower=0).values,
                'method': 'Prophet'
            })
            
            return results
            
        except AttributeError as e:
            if 'stan_backend' in str(e):
                print("⚠️ Erreur Prophet (problème de version). Passage à ARIMA...")
                return self._forecast_arima(df, horizon)
            else:
                print(f"❌ Erreur Prophet: {e}")
                return self._forecast_arima(df, horizon)
        except Exception as e:
            print(f"❌ Erreur Prophet: {e}")
            print("   Tentative avec ARIMA...")
            return self._forecast_arima(df, horizon)
    
    def _forecast_hf_enhanced(self, df, horizon, product):
        """Prévision améliorée avec analyse HF."""
        base_forecast = self._forecast_prophet(df, horizon)
        
        if base_forecast is None:
            return None
        
        # Analyser le sentiment si modèle disponible
        if 'sentiment' in self.hf_models:
            sentiment = self.analyze_market_sentiment(product)
            
            if sentiment and sentiment['label'] == 'POSITIVE':
                adjustment = 1 + (sentiment['score'] - 0.5) * 0.3
                base_forecast['predicted_demand'] *= adjustment
                base_forecast['upper_bound'] *= adjustment
                base_forecast['method'] = 'HF-Enhanced Prophet'
                print(f"  ✨ Prévisions ajustées avec sentiment positif (+{(adjustment-1)*100:.1f}%)")
        
        return base_forecast
    
    def analyze_market_sentiment(self, product):
        """Analyse le sentiment du marché avec HF."""
        if 'sentiment' not in self.hf_models:
            return None
        
        try:
            df = self.db.get_inventory_data(product, period_days=30)
            
            avg_sales = df['daily_sold_units'].mean()
            recent_sales = df['daily_sold_units'].tail(7).mean()
            
            if recent_sales > avg_sales * 1.2:
                trend_text = f"Sales for {product} are increasing significantly with strong demand"
            elif recent_sales < avg_sales * 0.8:
                trend_text = f"Sales for {product} are declining with weak demand"
            else:
                trend_text = f"Sales for {product} are stable with steady demand"
            
            sentiment = self.hf_models['sentiment'](trend_text)[0]
            
            print(f"\n📊 Analyse de sentiment pour {product}:")
            print(f"  • Tendance: {trend_text}")
            print(f"  • Sentiment: {sentiment['label']} (confiance: {sentiment['score']:.2%})")
            
            return sentiment
            
        except Exception as e:
            print(f"❌ Erreur d'analyse de sentiment: {e}")
            return None
    
    def detect_stock_anomalies(self, product=None, threshold_std=2.5):
        """
        Détecte les ruptures de stock et surstocks.
        
        Args:
            product: Produit à analyser (None pour tous)
            threshold_std: Seuil de détection en écarts-types
            
        Returns:
            pd.DataFrame: Anomalies détectées
        """
        df = self.db.get_inventory_data(product, period_days=60)
        
        if len(df) < 10:
            print("⚠️ Pas assez de données pour détecter les anomalies")
            return None
        
        anomalies = []
        products = [product] if product else df['Product type'].unique()
        
        for prod in products:
            prod_df = df[df['Product type'] == prod].copy()
            mean_stock = prod_df['current_stock_level'].mean()
            std_stock = prod_df['current_stock_level'].std()
            prod_df['z_score'] = (prod_df['current_stock_level'] - mean_stock) / std_stock
            
            for idx, row in prod_df.iterrows():
                anomaly = None
                
                if row['is_stockout'] == 1 or row['current_stock_level'] == 0:
                    anomaly = {
                        'date': row['date'],
                        'product': prod,
                        'type': 'Rupture de stock',
                        'severity': 'critical',
                        'stock_level': row['current_stock_level'],
                        'z_score': row['z_score'],
                        'message': 'Stock épuisé'
                    }
                elif abs(row['z_score']) > threshold_std:
                    if row['current_stock_level'] > mean_stock:
                        anomaly = {
                            'date': row['date'],
                            'product': prod,
                            'type': 'Surstock',
                            'severity': 'warning',
                            'stock_level': row['current_stock_level'],
                            'z_score': row['z_score'],
                            'message': f"Stock anormalement élevé ({row['z_score']:.2f}σ)"
                        }
                    else:
                        anomaly = {
                            'date': row['date'],
                            'product': prod,
                            'type': 'Stock critique',
                            'severity': 'danger',
                            'stock_level': row['current_stock_level'],
                            'z_score': row['z_score'],
                            'message': f"Stock anormalement bas ({row['z_score']:.2f}σ)"
                        }
                
                if anomaly:
                    # Classification HF si disponible
                    if 'classifier' in self.hf_models:
                        desc = f"{anomaly['type']} for {prod} with stock at {anomaly['stock_level']}"
                        hf_class = self._classify_anomaly_hf(desc)
                        if hf_class:
                            anomaly['hf_category'] = hf_class['category']
                            anomaly['hf_confidence'] = hf_class['confidence']
                    
                    anomalies.append(anomaly)
        
        if anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            print(f"🔍 {len(anomalies)} anomalies détectées")
            return anomalies_df
        else:
            print("✅ Aucune anomalie détectée")
            return pd.DataFrame()
    
    def _classify_anomaly_hf(self, description):
        """Classifie une anomalie avec zero-shot classification."""
        if 'classifier' not in self.hf_models:
            return None
        
        try:
            candidate_labels = [
                "critical stock shortage",
                "overstock situation",
                "seasonal fluctuation",
                "supply chain disruption",
                "demand spike"
            ]
            
            result = self.hf_models['classifier'](description, candidate_labels)
            
            return {
                'category': result['labels'][0],
                'confidence': result['scores'][0]
            }
            
        except Exception as e:
            print(f"⚠️ Erreur de classification: {e}")
            return None
    
    def suggest_restock_plan(self):
        """Calcule les quantités recommandées pour chaque produit."""
        products = self.db.get_all_products()
        restock_plan = []
        
        for product in products:
            df = self.db.get_inventory_data(product, period_days=30)
            
            if len(df) == 0:
                continue
            
            avg_daily_sales = df['daily_sold_units'].mean()
            current_stock = df['current_stock_level'].iloc[-1]
            
            # Gérer différents noms de colonnes pour le lead time
            if 'Lead time' in df.columns:
                avg_lead_time = df['Lead time'].mean()
            elif 'Lead times' in df.columns:
                avg_lead_time = df['Lead times'].mean()
            else:
                avg_lead_time = 7  # Valeur par défaut
            
            safety_stock = avg_daily_sales * 7
            reorder_point = (avg_daily_sales * avg_lead_time) + safety_stock
            optimal_order_qty = avg_daily_sales * 30
            days_of_stock = current_stock / avg_daily_sales if avg_daily_sales > 0 else 999
            
            if current_stock <= reorder_point:
                action = 'COMMANDER'
                urgency = 'urgent' if current_stock < safety_stock else 'high'
            elif current_stock > optimal_order_qty * 2:
                action = 'SURSTOCK'
                urgency = 'low'
            else:
                action = 'OK'
                urgency = 'normal'
            
            plan_item = {
                'product': product,
                'current_stock': int(current_stock),
                'avg_daily_sales': round(avg_daily_sales, 2),
                'days_of_stock': round(days_of_stock, 1),
                'reorder_point': int(reorder_point),
                'safety_stock': int(safety_stock),
                'suggested_order_qty': int(optimal_order_qty),
                'action': action,
                'urgency': urgency
            }
            
            # Recommandation IA si disponible
            if 'generator' in self.hf_models and urgency in ['urgent', 'high']:
                plan_item['ai_recommendation'] = self._generate_ai_recommendation(product, plan_item)
            
            restock_plan.append(plan_item)
        
        restock_df = pd.DataFrame(restock_plan)
        urgency_order = {'urgent': 0, 'high': 1, 'normal': 2, 'low': 3}
        restock_df['urgency_rank'] = restock_df['urgency'].map(urgency_order)
        restock_df = restock_df.sort_values('urgency_rank').drop('urgency_rank', axis=1)
        
        print(f"📦 Plan de réapprovisionnement créé pour {len(restock_df)} produits")
        return restock_df
    
    def _generate_ai_recommendation(self, product, plan):
        """Génère une recommandation IA."""
        if 'generator' not in self.hf_models:
            return "Recommandations classiques disponibles."
        
        try:
            prompt = f"For product {product} with {plan['current_stock']} units in stock, "
            prompt += f"selling {plan['avg_daily_sales']:.1f} units per day, recommend:"
            
            recs = self.hf_models['generator'](
                prompt,
                max_length=80,
                num_return_sequences=1,
                temperature=0.7
            )
            
            return recs[0]['generated_text']
            
        except Exception as e:
            return "Analyse basée sur les données historiques."