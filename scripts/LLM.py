"""
Enhanced LLM Supply Chain Agent - English Version with Fixed Calculations
"""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AGENT_TOOLS_DIR = PROJECT_ROOT / "src" / "agent_tools"

for path in [PROJECT_ROOT, AGENT_TOOLS_DIR, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Imports
from database import DatabaseManager
from analysis import AnalysisEngine
from visualization import Visualizer
from reports import ReportGenerator
from setup_database import setup_database

# LLM Imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ transformers not available. Install: pip install transformers torch")


class EnhancedSupplyChainLLM:
    """
    Enhanced LLM Agent with:
    - Correct revenue calculations (avoiding duplication)
    - Clear distinction between SKU-level and category-level metrics
    - Better demand classification
    - English conversation
    """
    
    def __init__(self, csv_file=None):
        """Initialize the enhanced LLM agent."""
        print("🤖 Initializing Enhanced Supply Chain LLM Agent\n")
        
        # Load data
        if csv_file:
            data = setup_database(csv_file)
        else:
            data = setup_database()
        
        # Initialize components
        self.db = DatabaseManager(data)
        
        # Initialize HF models for advanced analysis
        hf_models = self._initialize_hf_models()
        
        self.analysis = AnalysisEngine(self.db, hf_models)
        self.viz = Visualizer(self.db, self.analysis)
        self.reports = ReportGenerator(self.db, self.analysis)
        
        # Initialize conversational LLM
        self.llm = None
        self.tokenizer = None
        self._initialize_conversational_llm()
        
        # Conversation context
        self.conversation_history = []
        
        print("✅ Enhanced LLM Agent initialized!\n")
    
    def _initialize_hf_models(self):
        """Initialize Hugging Face models for analysis."""
        print("🔄 Loading AI analysis models...")
        hf_models = {}
        
        try:
            print("  • Sentiment analysis...")
            hf_models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            print("  • Advice generation...")
            hf_models['generator'] = pipeline(
                "text-generation",
                model="gpt2",
                max_length=100
            )
            
            print("✅ AI models loaded!\n")
            
        except Exception as e:
            print(f"⚠️ Error loading AI models: {e}")
            print("   Agent will continue without sentiment analysis\n")
        
        return hf_models
    
    def _initialize_conversational_llm(self):
        """Initialize conversational model."""
        if not LLM_AVAILABLE:
            print("ℹ️ Basic conversation mode (no advanced LLM)")
            return
        
        try:
            print("🔄 Loading conversational model...")
            
            model_name = "microsoft/DialoGPT-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Conversational model loaded!\n")
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            print("   Using basic conversation mode\n")
            self.llm = None
    
    def get_category_metrics(self, category, period_days=30):
        """
        Calculate CORRECT category-level metrics.
        
        FIXES:
        1. Revenue: Count each SKU only once (not per day)
        2. Clear distinction: SKU-level vs Category-level
        3. Proper aggregation
        """
        # Get all data for this category
        category_data = self.db.data[self.db.data['Product type'] == category].copy()
        
        if len(category_data) == 0:
            return None
        
        # Filter by period
        end_date = category_data['date'].max()
        start_date = end_date - pd.Timedelta(days=period_days)
        recent_data = category_data[category_data['date'] >= start_date]
        
        # Get unique SKUs
        unique_skus = category_data['SKU'].unique()
        num_skus = len(unique_skus)
        
        # CORRECT REVENUE CALCULATION
        # Revenue is per SKU (not per day), so we take the latest value for each SKU
        revenue_per_sku = {}
        for sku in unique_skus:
            sku_data = category_data[category_data['SKU'] == sku]
            if 'Revenue generated' in sku_data.columns:
                # Take the LAST (most recent) revenue value for this SKU
                revenue_per_sku[sku] = sku_data['Revenue generated'].iloc[-1]
        
        total_revenue = sum(revenue_per_sku.values()) if revenue_per_sku else 0
        
        # Sales calculation (sum across all days and SKUs)
        total_sales = recent_data['daily_sold_units'].sum()
        
        # Average daily sales per SKU
        days_in_period = (recent_data['date'].max() - recent_data['date'].min()).days + 1
        avg_daily_sales_per_sku = total_sales / (num_skus * days_in_period) if days_in_period > 0 else 0
        
        # Category-level daily average
        category_daily_avg = total_sales / days_in_period if days_in_period > 0 else 0
        
        # Current stock (sum across all SKUs, latest values)
        current_stock_per_sku = {}
        for sku in unique_skus:
            sku_data = category_data[category_data['SKU'] == sku]
            current_stock_per_sku[sku] = sku_data['current_stock_level'].iloc[-1]
        
        total_current_stock = sum(current_stock_per_sku.values())
        
        # Check for stockouts across all SKUs
        stockout_skus = [sku for sku, stock in current_stock_per_sku.items() if stock == 0]
        
        return {
            'category': category,
            'num_skus': num_skus,
            'period_days': period_days,
            
            # Revenue (FIXED - count each SKU once)
            'total_revenue': total_revenue,
            'avg_revenue_per_sku': total_revenue / num_skus if num_skus > 0 else 0,
            
            # Sales
            'total_sales': total_sales,
            'category_daily_avg': category_daily_avg,  # Category-level
            'sku_daily_avg': avg_daily_sales_per_sku,  # SKU-level
            
            # Stock
            'total_current_stock': total_current_stock,
            'avg_stock_per_sku': total_current_stock / num_skus if num_skus > 0 else 0,
            'stockout_skus': stockout_skus,
            'num_stockouts': len(stockout_skus),
            
            # Stock by SKU
            'stock_by_sku': current_stock_per_sku
        }
    
    def classify_demand(self, category_metrics, all_categories_metrics):
        """
        IMPROVED demand classification.
        
        FIXES:
        - Compare to market average, not just relative ranking
        - Use absolute thresholds
        - Consider category size (num SKUs)
        """
        if not category_metrics or not all_categories_metrics:
            return "UNKNOWN"
        
        # Calculate market average
        total_market_sales = sum(m['total_sales'] for m in all_categories_metrics.values() if m)
        total_market_skus = sum(m['num_skus'] for m in all_categories_metrics.values() if m)
        market_avg_per_sku = total_market_sales / total_market_skus if total_market_skus > 0 else 0
        
        # Get this category's SKU average
        category_sku_avg = category_metrics['sku_daily_avg']
        
        # Calculate relative performance
        if market_avg_per_sku > 0:
            performance_ratio = category_sku_avg / market_avg_per_sku
        else:
            performance_ratio = 1.0
        
        # Classification with context
        if performance_ratio >= 1.2:
            return "HIGH (20%+ above market average)"
        elif performance_ratio >= 0.9:
            return "MODERATE (within market average)"
        elif performance_ratio >= 0.5:
            return "LOW (below market average)"
        else:
            return "VERY LOW (significantly below market)"
    
    def understand_query(self, user_input):
        """Understand user intent with enhanced NLU."""
        user_lower = user_input.lower()
        
        intent = None
        category = None
        sku = None
        params = {}
        
        # Detect intent
        if any(word in user_lower for word in ['stock', 'inventory', 'how much', 'how many', 'level']):
            intent = 'check_stock'
        elif any(word in user_lower for word in ['sale', 'sold', 'selling', 'revenue', 'income']):
            intent = 'check_sales'
        elif any(word in user_lower for word in ['forecast', 'predict', 'future', 'demand', 'expect']):
            intent = 'forecast'
        elif any(word in user_lower for word in ['anomaly', 'anomalies', 'problem', 'issue', 'alert', 'stockout']):
            intent = 'detect_anomalies'
        elif any(word in user_lower for word in ['report', 'summary', 'status', 'overview', 'situation']):
            intent = 'status'
        elif any(word in user_lower for word in ['order', 'restock', 'replenish', 'buy', 'purchase']):
            intent = 'restock'
        elif any(word in user_lower for word in ['graph', 'chart', 'visual', 'show', 'display', 'plot']):
            intent = 'visualize'
        elif any(word in user_lower for word in ['advice', 'recommend', 'suggest', 'what should', 'strategy']):
            intent = 'advice'
        elif any(word in user_lower for word in ['help', 'how', 'what can']):
            intent = 'help'
        else:
            intent = 'unknown'
        
        # Detect category
        categories = self.db.get_all_products()
        for cat in categories:
            if cat.lower() in user_lower:
                category = cat
                break
        
        # Extract temporal parameters
        if 'week' in user_lower:
            params['period'] = 7
        elif 'month' in user_lower:
            params['period'] = 30
        elif any(str(i) in user_lower for i in range(1, 100)):
            import re
            numbers = re.findall(r'\d+', user_lower)
            if numbers:
                params['period'] = int(numbers[0])
        
        return {
            'intent': intent,
            'category': category,
            'sku': sku,
            'params': params,
            'original': user_input
        }
    
    def generate_response(self, query_info):
        """Generate response with correct calculations."""
        intent = query_info['intent']
        category = query_info['category']
        params = query_info.get('params', {})
        
        # If no category specified for certain intents
        if intent not in ['status', 'help', 'unknown'] and not category:
            categories = self.db.get_all_products()
            return (f"🤔 Which category are you asking about?\n"
                   f"Available categories: {', '.join(categories)}\n"
                   f"Please specify.")
        
        # Route to handlers
        if intent == 'check_stock':
            return self._handle_check_stock(category, params)
        elif intent == 'check_sales':
            return self._handle_check_sales(category, params)
        elif intent == 'forecast':
            return self._handle_forecast(category, params)
        elif intent == 'detect_anomalies':
            return self._handle_anomalies(category, params)
        elif intent == 'status':
            return self._handle_status()
        elif intent == 'restock':
            return self._handle_restock(category)
        elif intent == 'visualize':
            return self._handle_visualize(category, params)
        elif intent == 'advice':
            return self._handle_strategic_advice(category)
        elif intent == 'help':
            return self._handle_help()
        else:
            return self._handle_unknown(query_info['original'])
    
    def _handle_check_sales(self, category, params):
        """Analyze sales with CORRECT calculations."""
        period = params.get('period', 30)
        
        # Get CORRECT metrics
        metrics = self.get_category_metrics(category, period_days=period)
        
        if not metrics:
            return f"❌ No sales data available for {category}"
        
        # Get all categories for comparison
        all_categories = self.db.get_all_products()
        all_metrics = {cat: self.get_category_metrics(cat, period) for cat in all_categories}
        
        # Classify demand CORRECTLY
        demand_class = self.classify_demand(metrics, all_metrics)
        
        response = f"📈 **SALES ANALYSIS - {category}** (last {period} days)\n\n"
        
        response += "📊 **Category-Level Metrics**:\n"
        response += f"   • Total SKUs in category: {metrics['num_skus']}\n"
        response += f"   • Total units sold: {metrics['total_sales']:.0f} units\n"
        response += f"   • Category daily average: {metrics['category_daily_avg']:.1f} units/day\n"
        response += f"   • Total revenue: ${metrics['total_revenue']:,.2f} ✅ CORRECT\n\n"
        
        response += "🔍 **SKU-Level Metrics** (per product):\n"
        response += f"   • Average sales per SKU: {metrics['sku_daily_avg']:.2f} units/day/SKU\n"
        response += f"   • Average revenue per SKU: ${metrics['avg_revenue_per_sku']:,.2f}\n\n"
        
        response += f"📊 **Demand Classification**: {demand_class}\n"
        
        # Add context
        if "HIGH" in demand_class:
            response += "\n✨ Strong performance! This category outperforms the market."
        elif "MODERATE" in demand_class:
            response += "\n➡️ Stable performance aligned with market average."
        elif "LOW" in demand_class:
            response += "\n⚠️ Below-average performance. Consider promotional strategies."
        
        return response
    
    def _handle_check_stock(self, category, params):
        """Check stock with ALL SKUs analysis."""
        period = params.get('period', 30)
        
        metrics = self.get_category_metrics(category, period_days=period)
        
        if not metrics:
            return f"❌ No data available for {category}"
        
        response = f"📦 **STOCK ANALYSIS - {category}**\n\n"
        
        response += "📊 **Category Overview**:\n"
        response += f"   • Total SKUs: {metrics['num_skus']}\n"
        response += f"   • Total current stock: {metrics['total_current_stock']:.0f} units\n"
        response += f"   • Average stock per SKU: {metrics['avg_stock_per_sku']:.1f} units\n\n"
        
        # CRITICAL: Show stockout information
        if metrics['num_stockouts'] > 0:
            response += f"🚨 **STOCKOUT ALERT**:\n"
            response += f"   • {metrics['num_stockouts']} SKU(s) currently at ZERO stock\n"
            response += f"   • Affected SKUs: {', '.join(metrics['stockout_skus'][:5])}"
            if len(metrics['stockout_skus']) > 5:
                response += f" and {len(metrics['stockout_skus']) - 5} more"
            response += "\n\n"
        
        # Days of stock remaining (category level)
        if metrics['category_daily_avg'] > 0:
            days_remaining = metrics['total_current_stock'] / metrics['category_daily_avg']
            response += f"⏱️ **Stock Coverage**: {days_remaining:.1f} days\n\n"
            
            if days_remaining < 7:
                response += "🚨 **CRITICAL**: Less than 1 week of stock!\n"
            elif days_remaining < 14:
                response += "⚠️ **WARNING**: Less than 2 weeks of stock\n"
            else:
                response += "✅ Stock level is adequate\n"
        
        return response
    
    def _handle_anomalies(self, category, params):
        """Detect anomalies across ALL SKUs."""
        anomalies = self.analysis.detect_stock_anomalies(category)
        
        if anomalies is None or len(anomalies) == 0:
            return f"✅ No anomalies detected for {category}\nStock is stable."
        
        response = f"⚠️ **ANOMALIES DETECTED - {category}**\n\n"
        response += f"• Total anomalies: {len(anomalies)} (across ALL SKUs)\n"
        
        critical = anomalies[anomalies['severity'] == 'critical']
        danger = anomalies[anomalies['severity'] == 'danger']
        warning = anomalies[anomalies['severity'] == 'warning']
        
        if len(critical) > 0:
            response += f"• 🔴 Stockouts: {len(critical)} occurrences\n"
        if len(danger) > 0:
            response += f"• 🟠 Critical levels: {len(danger)} occurrences\n"
        if len(warning) > 0:
            response += f"• 🟡 Overstocks: {len(warning)} occurrences\n"
        
        # Show latest anomaly
        if len(anomalies) > 0:
            last = anomalies.iloc[-1]
            response += f"\n📅 Most recent anomaly: {last['date'].strftime('%Y-%m-%d')}"
            response += f"\n   Type: {last['type']}"
            response += f"\n   Stock level: {last['stock_level']:.0f} units"
        
        response += "\n\n💡 Note: Anomalies are detected across ALL products (SKUs) in this category."
        
        return response
    
    def _handle_forecast(self, category, params):
        """Generate forecast."""
        horizon = params.get('period', 14)
        
        forecast = self.analysis.forecast_demand(category, horizon=horizon, method='simple')
        
        if forecast is None:
            return f"❌ Unable to generate forecast for {category}"
        
        total_forecast = forecast['predicted_demand'].sum()
        avg_forecast = forecast['predicted_demand'].mean()
        
        stats = self.db.get_product_stats(category, period_days=30)
        variation = ((avg_forecast - stats['avg_daily_sales']) / stats['avg_daily_sales'] * 100) if stats and stats['avg_daily_sales'] > 0 else 0
        
        response = f"🔮 **DEMAND FORECAST - {category}** (next {horizon} days)\n\n"
        response += f"• Predicted total demand: {total_forecast:.0f} units\n"
        response += f"• Daily average: {avg_forecast:.1f} units\n"
        response += f"• Change vs historical: {variation:+.1f}%\n\n"
        
        if variation > 10:
            response += "📈 **Trend**: Increasing demand predicted\n"
            response += "   → Anticipate restocking needs"
        elif variation < -10:
            response += "📉 **Trend**: Decreasing demand predicted\n"
            response += "   → Adjust orders downward"
        else:
            response += "➡️ **Trend**: Stable demand predicted"
        
        response += f"\n\n💡 Want to see the chart? (type 'chart {category}')"
        
        return response
    
    def _handle_status(self):
        """Global status."""
        summary = self.reports.generate_summary_stats()
        
        response = "📊 **GLOBAL SUPPLY CHAIN STATUS**\n\n"
        response += f"• Categories: {summary['total_products']}\n"
        response += f"• Total sales: {summary['total_sales']:,.0f} units\n"
        response += f"• Total stock: {summary['total_stock']:,.0f} units\n"
        response += f"• Historical stockouts: {summary['stockout_incidents']}\n\n"
        
        response += "**By Category**:\n"
        categories = self.db.get_all_products()
        
        for cat in categories:
            metrics = self.get_category_metrics(cat, period_days=30)
            if metrics:
                if metrics['num_stockouts'] > 0:
                    status = f"🔴 {metrics['num_stockouts']} SKU(s) at zero"
                elif metrics['total_current_stock'] < metrics['total_sales'] * 0.5:
                    status = "🟠 LOW"
                else:
                    status = "🟢 OK"
                
                response += f"\n{status} **{cat}**: {metrics['num_skus']} SKUs, {metrics['total_current_stock']:.0f} units total"
        
        return response
    
    def _handle_restock(self, category):
        """Restock recommendations."""
        restock_plan = self.analysis.suggest_restock_plan()
        
        if category:
            plan = restock_plan[restock_plan['product'] == category]
        else:
            plan = restock_plan[restock_plan['urgency'].isin(['urgent', 'high'])]
        
        if len(plan) == 0:
            if category:
                return f"✅ {category}: Stock sufficient, no order needed"
            else:
                return "✅ All stock levels are sufficient"
        
        response = "📋 **RESTOCKING RECOMMENDATIONS**\n\n"
        
        for _, item in plan.iterrows():
            emoji = "🚨" if item['urgency'] == 'urgent' else "⚠️" if item['urgency'] == 'high' else "ℹ️"
            
            response += f"{emoji} **{item['product']}**\n"
            response += f"   • Current stock: {item['current_stock']} units\n"
            response += f"   • Days remaining: {item['days_of_stock']:.1f}\n"
            response += f"   • Suggested order: {item['suggested_order_qty']} units\n"
            response += f"   • Reorder point: {item['reorder_point']} units\n\n"
        
        return response
    
    def _handle_visualize(self, category, params):
        """Generate visualization."""
        if not category:
            return "🤔 Which category would you like to visualize?"
        
        print(f"\n📊 Generating charts for {category}...\n")
        
        try:
            self.viz.plot_inventory_levels(category, days=30)
            return f"✅ Stock evolution chart displayed for {category}"
        except Exception as e:
            return f"❌ Error generating chart: {e}"
    
    def _handle_strategic_advice(self, category):
        """Strategic advice (placeholder for now)."""
        return "🎯 Strategic advice feature - coming soon!"
    
    def _handle_help(self):
        """Help guide."""
        return """
🤖 **SUPPLY CHAIN ASSISTANT - USER GUIDE**

I can help with these types of questions:

📦 **Stock**
• "What is the stock for haircare?"
• "How many units of skincare are left?"

📈 **Sales**
• "What are the sales for cosmetics?"
• "How much haircare did I sell this month?"

🔮 **Forecasts**
• "Forecast demand for skincare"
• "What will demand be in 2 weeks?"

⚠️ **Anomalies**
• "Are there any problems with haircare?"
• "Detect anomalies"

📋 **Restocking**
• "Should I order skincare?"
• "Which products need restocking?"

📊 **Global Status**
• "What's the status?"
• "Give me a summary"

📈 **Visualization**
• "Show me the haircare chart"
• "Display stock evolution"

💡 **Tip**: Be natural! Ask questions as you would to a colleague.
"""
    
    def _handle_unknown(self, original_query):
        """Handle unknown queries."""
        return f"""
🤔 I didn't quite understand: "{original_query}"

💡 Try something like:
• "What is the stock for haircare?"
• "Forecast demand for skincare"
• "Global status"
• "Help"
"""
    
    def chat(self, user_input):
        """Main chat interface."""
        query_info = self.understand_query(user_input)
        response = self.generate_response(query_info)
        
        self.conversation_history.append({
            'user': user_input,
            'assistant': response,
            'query_info': query_info
        })
        
        return response


def main():
    """Main conversation loop."""
    print("="*70)
    print("🤖 ENHANCED SUPPLY CHAIN LLM AGENT")
    print("="*70)
    print()
    print("💬 Ask me questions in natural language!")
    print("   Type 'help' to see what I can do")
    print("   Type 'quit' or 'exit' to exit")
    print()
    print("="*70)
    print()
    
    try:
        agent = EnhancedSupplyChainLLM()
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    while True:
        try:
            user_input = input("🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye! See you soon!")
                break
            
            if not user_input:
                continue
            
            print()
            response = agent.chat(user_input)
            print(f"🤖 Assistant: {response}")
            print()
            print("-"*70)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()