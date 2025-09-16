import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account

class MerchandiseAnalyzerBQ:
    """BigQuery-powered merchandise analysis for GA4 ecommerce data."""
    
    def __init__(self, project_id, credentials_path='bigquery-credentials.json'):
        """Initialize BigQuery client."""
        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )
        self.client = bigquery.Client(
            credentials=self.credentials,
            project=project_id
        )
        self.project_id = project_id
        self.df = None
        self.df_all = None
        self.df_filtered = None
        self.recommendations = {}
        
    def load_from_bigquery(self, date_from=None, date_to=None, min_views=150, ga4_property_id=None):
        """
        Load and aggregate GA4 ecommerce data from BigQuery.
        
        Args:
            date_from: Start date (YYYY-MM-DD format)
            date_to: End date (YYYY-MM-DD format)  
            min_views: Minimum views threshold
            ga4_property_id: Your GA4 property ID
        """
        
        # Set default dates if not provided
        if not date_to:
            date_to = datetime.now().strftime('%Y-%m-%d')
        if not date_from:
            date_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Convert dates to BigQuery table suffix format (YYYYMMDD)
        start_suffix = datetime.strptime(date_from, '%Y-%m-%d').strftime('%Y%m%d')
        end_suffix = datetime.strptime(date_to, '%Y-%m-%d').strftime('%Y%m%d')
        
        query = f"""
        WITH item_metrics AS (
        SELECT
            item.item_name,
            -- Item views (view_item events)
            COUNTIF(event_name = 'view_item') as items_viewed,
            
            -- Items added to cart
            COUNTIF(event_name = 'add_to_cart') as items_added_to_cart,
            
            -- Items purchased
            COUNTIF(event_name = 'purchase') as items_purchased,
            
            -- Item revenue from purchase events
            -- Since individual item revenue is often NULL, we'll use transaction-level revenue
            SUM(
            CASE 
                WHEN event_name = 'purchase' AND ecommerce.purchase_revenue IS NOT NULL
                THEN ecommerce.purchase_revenue / ecommerce.total_item_quantity  -- Distribute revenue across items
                WHEN event_name = 'purchase' AND ecommerce.purchase_revenue_in_usd IS NOT NULL
                THEN ecommerce.purchase_revenue_in_usd / ecommerce.total_item_quantity
                ELSE 0 
            END
            ) as item_revenue
            
        FROM 
            `{self.project_id}.analytics_{ga4_property_id}.events_*`,
            UNNEST(items) AS item
        WHERE 
            _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
            AND event_name IN ('view_item', 'add_to_cart', 'purchase')
            AND item.item_name IS NOT NULL
            AND item.item_name != '(not set)'
        GROUP BY 
            item.item_name
        HAVING
            items_viewed > 0
        )

        SELECT 
        item_name,
        items_viewed,
        items_added_to_cart,
        items_purchased,
        COALESCE(item_revenue, 0) as item_revenue
        FROM item_metrics
        ORDER BY item_revenue DESC
        """
        
        print(f"🔍 Querying GA4 data from {date_from} to {date_to}...")
        print(f"📊 Using dataset: analytics_{ga4_property_id}")
        
        try:
            # Execute query
            query_job = self.client.query(query)
            df = query_job.to_dataframe()
            
            if df.empty:
                print("❌ No data returned from BigQuery. Check your date range and GA4 property ID.")
                return False
            
            print(f"✓ Loaded {len(df)} products from BigQuery")
            
            # Store all data
            self.df_all = df.copy()
            
            # Create filtered dataset
            self.df_filtered = df[df['items_viewed'] >= min_views].copy()
            
            print(f"✓ Analyzing {len(self.df_filtered)} products with {min_views}+ views")
            
            # Set main df
            self.df = self.df_all
            
            # Clean column names for compatibility
            self.clean_column_names()
            
            return True
            
        except Exception as e:
            print(f"❌ BigQuery error: {e}")
            print("Falling back to sample data...")
            return False
    
    def clean_column_names(self):
        """Ensure column names match expected format."""
        # Column names from BigQuery should already be correct
        # but let's ensure they're lowercase with underscores
        if self.df is not None:
            self.df.columns = [col.lower().replace(' ', '_') for col in self.df.columns]
        if self.df_all is not None:
            self.df_all.columns = [col.lower().replace(' ', '_') for col in self.df_all.columns]
        if self.df_filtered is not None:
            self.df_filtered.columns = [col.lower().replace(' ', '_') for col in self.df_filtered.columns]
    
    def calculate_metrics(self):
        """Calculate all conversion and performance metrics."""
        
        # Calculate metrics for both full and filtered datasets
        datasets = [self.df_all, self.df_filtered] if self.df_filtered is not None else [self.df]
        
        for df in datasets:
            # Avoid division by zero
            df['view_to_cart_rate'] = np.where(
                df['items_viewed'] > 0,
                df['items_added_to_cart'] / df['items_viewed'],
                0
            )
            
            df['cart_to_purchase_rate'] = np.where(
                df['items_added_to_cart'] > 0,
                df['items_purchased'] / df['items_added_to_cart'],
                0
            )
            
            df['view_to_purchase_rate'] = np.where(
                df['items_viewed'] > 0,
                df['items_purchased'] / df['items_viewed'],
                0
            )
            
            df['avg_order_value'] = np.where(
                df['items_purchased'] > 0,
                df['item_revenue'] / df['items_purchased'],
                0
            )
            
            df['revenue_per_view'] = np.where(
                df['items_viewed'] > 0,
                df['item_revenue'] / df['items_viewed'],
                0
            )
            
            # Performance scores (normalized 0-100)
            df['conversion_score'] = self.normalize_score(df['view_to_purchase_rate'])
            df['revenue_score'] = self.normalize_score(df['revenue_per_view'])
            df['traffic_score'] = self.normalize_score(df['items_viewed'])
            
            # Composite performance score
            df['performance_score'] = (
                df['conversion_score'] * 0.4 +
                df['revenue_score'] * 0.4 +
                df['traffic_score'] * 0.2
            )
        
        # Update self.df to maintain compatibility
        self.df = self.df_all
        
        print("✓ Calculated all performance metrics")
        return self.df
    
    def normalize_score(self, series, min_val=0, max_val=100):
        """Normalize values to 0-100 scale."""
        if series.max() == series.min():
            return pd.Series([50] * len(series))
        return ((series - series.min()) / (series.max() - series.min())) * (max_val - min_val) + min_val
    
    def segment_products(self):
        """Segment products using business-aligned criteria."""
        
        datasets = [self.df_all, self.df_filtered] if self.df_filtered is not None else [self.df]
        
        for df in datasets:
            # Calculate percentiles for key metrics
            revenue_p75 = df['item_revenue'].quantile(0.75)
            revenue_p50 = df['item_revenue'].quantile(0.50)
            views_p75 = df['items_viewed'].quantile(0.75)
            views_p50 = df['items_viewed'].quantile(0.50)
            conv_p75 = df['view_to_purchase_rate'].quantile(0.75)
            conv_p50 = df['view_to_purchase_rate'].quantile(0.50)
            
            segments = []
            for _, row in df.iterrows():
                revenue = row['item_revenue']
                views = row['items_viewed']
                conv_rate = row['view_to_purchase_rate']
                
                # Stars: High revenue AND high conversion (truly efficient)
                if revenue >= revenue_p75 and conv_rate >= conv_p75:
                    segments.append('Stars')
                
                # Hidden Gems: High conversion but low traffic
                elif conv_rate >= conv_p75 and views < views_p50:
                    segments.append('Hidden Gems')
                
                # Needs Attention: High traffic but poor conversion
                elif views >= views_p50 and conv_rate < conv_p50:
                    segments.append('Needs Attention')
                
                # Revenue Drivers: High revenue despite lower conversion (not underperformers!)
                elif revenue >= revenue_p75:
                    segments.append('Revenue Drivers')
                
                # Underperformers: Everything else
                else:
                    segments.append('Underperformers')
            
            df['segment'] = segments
        
        # Report on filtered dataset
        report_df = self.df_filtered if self.df_filtered is not None else self.df
        segment_counts = report_df['segment'].value_counts()
        print("\n✓ Product Segmentation Complete (150+ views only):")
        for segment, count in segment_counts.items():
            print(f"  - {segment}: {count} products")
        
        self.df = self.df_all
        return self.df
    
    def prepare_dashboard_data(self):
        """Prepare data structure for dashboard template."""
        
        # Use df_all for overview metrics (all products including low traffic)
        df_overview = self.df_all if self.df_all is not None else self.df
        
        # Use df_filtered for insights and recommendations (150+ views only)
        df_analysis = self.df_filtered if self.df_filtered is not None else self.df
        
        # Basic metrics from ALL products
        insights = {
            'total_revenue': df_overview['item_revenue'].sum(),
            'total_views': int(df_overview['items_viewed'].sum()),
            'total_products': len(df_overview),
            'avg_conversion': df_overview['view_to_purchase_rate'].mean() if 'view_to_purchase_rate' in df_overview.columns else 0
        }
        
        # Get top 25% by views threshold
        views_threshold = df_analysis['items_viewed'].quantile(0.75)
        high_traffic_df = df_analysis[df_analysis['items_viewed'] >= views_threshold]
        
        # Also get revenue threshold to exclude top performers
        revenue_threshold = df_analysis['item_revenue'].quantile(0.7)  # Top 30% by revenue

        # Calculate target cart-to-purchase rate from top efficient products
        # Get products with decent traffic (100+ views) to ensure statistical relevance
        products_with_traffic = df_analysis[df_analysis['items_viewed'] >= 100].copy()
        
        # Initialize default benchmarks
        target_cart_conversion = 0.5  # Default 50%
        benchmark_purchase_rate = 3  # Default 3%
        
        if not products_with_traffic.empty:
            products_with_traffic['revenue_per_view'] = products_with_traffic['item_revenue'] / products_with_traffic['items_viewed']
            
            # Get top 20 most efficient products
            top_efficient = products_with_traffic.nlargest(min(20, len(products_with_traffic)), 'revenue_per_view')
            
            if not top_efficient.empty:
                # Calculate their median cart-to-purchase rate as our target
                target_cart_conversion = top_efficient['cart_to_purchase_rate'].median()
                benchmark_purchase_rate = top_efficient['view_to_purchase_rate'].median() * 100
                print(f"DEBUG: Target cart-to-purchase rate: {target_cart_conversion:.1%}")
        
        # Add benchmarks to insights for the template
        insights['benchmark_purchase_rate'] = float(benchmark_purchase_rate)
        insights['benchmark_cart_to_purchase'] = float(target_cart_conversion * 100)
        
        # Initialize the lists first
        carted_not_purchased_list = []
        not_carted_list = []

        # Use df_filtered for insights and recommendations (150+ views only)
        df_analysis = self.df_filtered if self.df_filtered is not None else self.df
        
        # Basic metrics from ALL products
        insights = {
            'total_revenue': df_overview['item_revenue'].sum(),
            'total_views': int(df_overview['items_viewed'].sum()),
            'total_products': len(df_overview),
            'avg_conversion': df_overview['view_to_purchase_rate'].mean() if 'view_to_purchase_rate' in df_overview.columns else 0
        }
        
        # Get top 25% by views threshold
        views_threshold = df_analysis['items_viewed'].quantile(0.75)
        high_traffic_df = df_analysis[df_analysis['items_viewed'] >= views_threshold]
        
        # Top performers (Stars segment) - keep as is
        stars = df_analysis[df_analysis['segment'] == 'Stars'] if 'segment' in df_analysis.columns else pd.DataFrame()
        if not stars.empty:
            top_performers_list = []
            for _, row in stars.nlargest(5, 'performance_score').iterrows():
                performer_data = {
                    'item_name': row['item_name'],
                    'items_viewed': int(row['items_viewed']),
                    'view_to_cart_rate': float(row['view_to_cart_rate']) if 'view_to_cart_rate' in row else 0,
                    'view_to_purchase_rate': float(row['view_to_purchase_rate']),
                    'item_revenue': float(row['item_revenue']) if 'item_revenue' in row else 0,
                    'performance_score': float(row['performance_score']) if 'performance_score' in row else 0,
                    'items_purchased': int(row['items_purchased']) if 'items_purchased' in row else 0
                }
                top_performers_list.append(performer_data)
            
            insights['top_performers'] = top_performers_list
            insights['star_performer'] = stars.iloc[0].to_dict() if len(stars) > 0 else None
        else:
            insights['top_performers'] = []
            insights['star_performer'] = None
        
        # Hidden gems - keep as is
        hidden_gems = df_analysis[df_analysis['segment'] == 'Hidden Gems'] if 'segment' in df_analysis.columns else pd.DataFrame()
        if not hidden_gems.empty:
            hidden_gems_sorted = hidden_gems.nlargest(5, 'view_to_purchase_rate')
            hidden_gems_list = []
            
            for _, row in hidden_gems_sorted.iterrows():
                gem_data = {
                    'item_name': row['item_name'],
                    'items_viewed': int(row['items_viewed']),
                    'view_to_cart_rate': float(row['view_to_cart_rate']) if 'view_to_cart_rate' in row else 0,
                    'view_to_purchase_rate': float(row['view_to_purchase_rate']),
                    'item_revenue': float(row['item_revenue']) if 'item_revenue' in row else 0,
                    'performance_score': float(row['performance_score']) if 'performance_score' in row else 0
                }
                hidden_gems_list.append(gem_data)
            
            insights['hidden_gems'] = hidden_gems_list
            insights['hidden_gems_count'] = len(hidden_gems)
            insights['top_hidden_gem'] = hidden_gems.iloc[0].to_dict() if len(hidden_gems) > 0 else None
        else:
            insights['hidden_gems'] = []
            insights['hidden_gems_count'] = 0
            insights['top_hidden_gem'] = None
        
        # REDESIGNED NEED ATTENTION - Split into two categories

        # Get top 25% by views threshold
        views_threshold = df_analysis['items_viewed'].quantile(0.75)
        high_traffic_df = df_analysis[df_analysis['items_viewed'] >= views_threshold]

        # Also get revenue threshold to exclude top performers
        revenue_threshold = df_analysis['item_revenue'].quantile(0.7)  # Top 30% by revenue

        # Calculate target cart-to-purchase rate from top efficient products
        # Get products with decent traffic (100+ views) to ensure statistical relevance
        products_with_traffic = df_analysis[df_analysis['items_viewed'] >= 100].copy()
        products_with_traffic['revenue_per_view'] = products_with_traffic['item_revenue'] / products_with_traffic['items_viewed']

        # Get top 20 most efficient products
        top_efficient = products_with_traffic.nlargest(20, 'revenue_per_view')
        # Calculate their median cart-to-purchase rate as our target
        target_cart_conversion = top_efficient['cart_to_purchase_rate'].median()

        # Initialize the lists first
        carted_not_purchased_list = []
        not_carted_list = []

        # 1. Carted but not purchased (high cart abandonment on high-traffic products)
        carted_not_purchased = high_traffic_df[
            (high_traffic_df['view_to_cart_rate'] > 0.05) &
            (high_traffic_df['cart_to_purchase_rate'] < target_cart_conversion) &
            ((high_traffic_df['item_revenue'] < revenue_threshold) & (high_traffic_df['items_added_to_cart'] >= 20))
        ].copy()

        if not carted_not_purchased.empty:
            # Calculate lost revenue using your best performers as the benchmark
            # Only calculate lost revenue if product is below the target
            carted_not_purchased['lost_revenue'] = carted_not_purchased.apply(
                lambda row: (
                    row['items_added_to_cart'] * 
                    max(0, target_cart_conversion - row['cart_to_purchase_rate']) *  # Only positive gaps
                    (row['item_revenue'] / max(row['items_purchased'], 1))  # Average order value
                ), axis=1
            )
            
            for _, row in carted_not_purchased.nlargest(5, 'items_viewed').iterrows():
                attention_data = {
                    'item_name': row['item_name'],
                    'items_viewed': int(row['items_viewed']),
                    'items_added_to_cart': int(row['items_added_to_cart']),
                    'view_to_cart_rate': float(row['view_to_cart_rate']),
                    'cart_to_purchase_rate': float(row['cart_to_purchase_rate']),
                    'view_to_purchase_rate': float(row['view_to_purchase_rate']) if 'view_to_purchase_rate' in row else 0,
                    'item_revenue': float(row['item_revenue']) if 'item_revenue' in row else 0,
                    'lost_revenue': float(row['lost_revenue']),
                    'target_cart_conversion': float(target_cart_conversion),  # Include the target for reference
                    'category': 'carted_not_purchased'
                }
                carted_not_purchased_list.append(attention_data)

        # 2. Not carted (high traffic but low cart rate)
        # Always get the top 5 highest-traffic products with poor cart rates
        not_carted = df_analysis[
            (df_analysis['view_to_cart_rate'] < 0.05) &  # Poor cart rate (below 5%)
            (df_analysis['item_revenue'] < revenue_threshold)  # Not already a top revenue driver
        ].copy()

        if not not_carted.empty:
            # Calculate lost opportunity
            avg_cart_rate = df_analysis['view_to_cart_rate'].median()
            not_carted['lost_opportunity'] = not_carted['items_viewed'] * avg_cart_rate * 0.3  # Potential conversions
            
            # ALWAYS sort by views (highest traffic first) and take top 5
            for _, row in not_carted.nlargest(5, 'items_viewed').iterrows():
                attention_data = {
                    'item_name': row['item_name'],
                    'items_viewed': int(row['items_viewed']),
                    'items_added_to_cart': int(row['items_added_to_cart']),
                    'view_to_cart_rate': float(row['view_to_cart_rate']),
                    'cart_to_purchase_rate': float(row['cart_to_purchase_rate']) if row['items_added_to_cart'] > 0 else 0,
                    'view_to_purchase_rate': float(row['view_to_purchase_rate']) if 'view_to_purchase_rate' in row else 0,
                    'item_revenue': float(row['item_revenue']) if 'item_revenue' in row else 0,
                    'lost_opportunity': float(row['lost_opportunity']),
                    'category': 'not_carted'
                }
                not_carted_list.append(attention_data)

        # Combine both categories for need_attention
        insights['need_attention'] = carted_not_purchased_list + not_carted_list
        insights['carted_not_purchased'] = carted_not_purchased_list
        insights['not_carted'] = not_carted_list
        insights['views_threshold'] = int(views_threshold)

        # Update counts for the alert section
        insights['underperformers_count'] = len(carted_not_purchased_list) + len(not_carted_list)
        insights['high_abandonment_count'] = len(carted_not_purchased_list)
                
        # Temporarily use filtered df for identify_opportunities
        temp_df = self.df
        self.df = df_analysis
        
        # Prepare recommendations from filtered data
        if not self.recommendations:
            self.identify_opportunities()
        
        recommendations = self.recommendations
        
        # Restore original df
        self.df = temp_df
        
        # Ensure all recommendation keys exist with default empty lists
        for key in ['homepage_heroes', 'email_features', 'promotion_needed', 'increase_exposure']:
            if key not in recommendations:
                recommendations[key] = []
        
        # Products list for table (filtered products only)
        products = df_analysis.to_dict('records')

        # Sort products by views in descending order
        products = sorted(products, key=lambda x: x['items_viewed'], reverse=True)
        
        print(f"DEBUG: prepare_dashboard_data returning {len(products)} products")
        print(f"DEBUG: First product: {products[0]['item_name'] if products else 'None'}")
        print(f"DEBUG: Last product: {products[-1]['item_name'] if products else 'None'}")
        
        return insights, recommendations, products
    
    def identify_opportunities(self):
        """Identify specific optimization opportunities."""
        
        opportunities = {
            'homepage_heroes': [],
            'email_features': [],
            'banner_candidates': [],
            'promotion_needed': [],
            'price_review': [],
            'increase_exposure': [],
            'cross_sell_anchors': []
        }
        
        # Homepage Heroes
        if len(self.df) > 0:
            opportunities['homepage_heroes'] = self.df.nlargest(min(5, len(self.df)), 'performance_score')[
                ['item_name', 'performance_score', 'item_revenue']
            ].to_dict('records')
        
        # Email Features - Hidden gems with full data
        hidden_gems = self.df[self.df['segment'] == 'Hidden Gems'] if 'segment' in self.df.columns else pd.DataFrame()
        if not hidden_gems.empty:
            email_features = []
            for _, row in hidden_gems.nlargest(min(5, len(hidden_gems)), 'view_to_purchase_rate').iterrows():
                email_features.append({
                    'item_name': row['item_name'],
                    'view_to_purchase_rate': float(row['view_to_purchase_rate']),
                    'items_viewed': int(row['items_viewed']),
                    'view_to_cart_rate': float(row['view_to_cart_rate']) if 'view_to_cart_rate' in row else 0
                })
            opportunities['email_features'] = email_features
        
        # Banner Candidates - Stars
        stars = self.df[self.df['segment'] == 'Stars'] if 'segment' in self.df.columns else pd.DataFrame()
        if not stars.empty:
            opportunities['banner_candidates'] = stars.nlargest(min(5, len(stars)), 'revenue_per_view')[
                ['item_name', 'revenue_per_view', 'items_viewed']
            ].to_dict('records')
        
        # Promotion Needed - High cart abandonment
        high_abandonment = self.df[
            (self.df['items_added_to_cart'] > 20) & 
            (self.df['cart_to_purchase_rate'] < 0.3)
        ]
        if not high_abandonment.empty:
            opportunities['promotion_needed'] = high_abandonment.nsmallest(min(5, len(high_abandonment)), 'cart_to_purchase_rate')[
                ['item_name', 'cart_to_purchase_rate', 'items_added_to_cart']
            ].to_dict('records')
        
        # Price Review - Underperformers
        underperformers = self.df[self.df['segment'] == 'Underperformers'] if 'segment' in self.df.columns else pd.DataFrame()
        if not underperformers.empty:
            opportunities['price_review'] = underperformers.nlargest(min(5, len(underperformers)), 'items_viewed')[
                ['item_name', 'items_viewed', 'view_to_cart_rate']
            ].to_dict('records')
        
        # Increase Exposure - Low traffic high conversion with full data
        low_traffic_gems = self.df[
            (self.df['items_viewed'] < self.df['items_viewed'].quantile(0.3)) &
            (self.df['view_to_purchase_rate'] > self.df['view_to_purchase_rate'].quantile(0.7))
        ]
        if not low_traffic_gems.empty:
            increase_exposure = []
            for _, row in low_traffic_gems.nlargest(min(5, len(low_traffic_gems)), 'view_to_purchase_rate').iterrows():
                increase_exposure.append({
                    'item_name': row['item_name'],
                    'items_viewed': int(row['items_viewed']),
                    'view_to_purchase_rate': float(row['view_to_purchase_rate']),
                    'view_to_cart_rate': float(row['view_to_cart_rate']) if 'view_to_cart_rate' in row else 0
                })
            opportunities['increase_exposure'] = increase_exposure
        
        # Cross-sell Anchors
        opportunities['cross_sell_anchors'] = self.df.nlargest(min(5, len(self.df)), 'item_revenue')[
            ['item_name', 'item_revenue', 'items_purchased']
        ].to_dict('records')
        
        self.recommendations = opportunities
        return opportunities
    
    def generate_insights(self):
        """Generate actionable insights and recommendations."""
        
        # Use filtered data for insights
        analysis_df = self.df_filtered if self.df_filtered is not None else self.df
        
        insights = []
        
        # Overall performance from ALL data
        total_revenue = self.df_all['item_revenue'].sum() if self.df_all is not None else self.df['item_revenue'].sum()
        total_views = self.df_all['items_viewed'].sum() if self.df_all is not None else self.df['items_viewed'].sum()
        avg_conversion = self.df_all['view_to_purchase_rate'].mean() if self.df_all is not None else self.df['view_to_purchase_rate'].mean()
        
        insights.append(f"📊 OVERALL PERFORMANCE")
        insights.append(f"• Total Revenue: R{total_revenue:,.2f}")
        insights.append(f"• Total Product Views: {total_views:,}")
        insights.append(f"• Average Conversion Rate: {avg_conversion:.2%}")
        insights.append("")
        
        insights.append(f"🎯 KEY OPPORTUNITIES (150+ views products)")
        
        # Stars - from filtered data
        stars = analysis_df[analysis_df['segment'] == 'Stars'] if 'segment' in analysis_df.columns else pd.DataFrame()
        if not stars.empty:
            top_star = stars.iloc[0]
            insights.append(f"• Star Performer: {top_star['item_name']}")
            insights.append(f"  - Revenue: R{top_star['item_revenue']:,.2f}")
            insights.append(f"  - Consider featuring in banners and cross-sell campaigns")
        
        # Hidden Gems - from filtered data
        hidden_gems = analysis_df[analysis_df['segment'] == 'Hidden Gems'] if 'segment' in analysis_df.columns else pd.DataFrame()
        if not hidden_gems.empty:
            insights.append(f"\n• Found {len(hidden_gems)} Hidden Gems with high conversion but low traffic")
            if len(hidden_gems) > 0:
                top_gem = hidden_gems.iloc[0]
                insights.append(f"  - Top opportunity: {top_gem['item_name']}")
                insights.append(f"  - Current views: {int(top_gem['items_viewed'])}, Conversion: {top_gem['view_to_purchase_rate']:.2%}")
                insights.append(f"  - ACTION: Increase exposure through email and homepage placement")
        
        # Underperformers - from filtered data
        underperformers = analysis_df[analysis_df['segment'] == 'Underperformers'] if 'segment' in analysis_df.columns else pd.DataFrame()
        if not underperformers.empty:
            insights.append(f"\n• {len(underperformers)} Underperformers need optimization")
            if len(underperformers) > 0:
                worst = underperformers.nsmallest(1, 'view_to_cart_rate').iloc[0]
                insights.append(f"  - Urgent review: {worst['item_name']}")
                insights.append(f"  - Views: {int(worst['items_viewed'])}, Cart Rate: {worst['view_to_cart_rate']:.2%}")
                insights.append(f"  - ACTION: Review pricing, images, and descriptions")
        
        # Cart abandonment - from filtered data
        high_abandonment = analysis_df[analysis_df['cart_to_purchase_rate'] < 0.3]
        if not high_abandonment.empty:
            insights.append(f"\n• {len(high_abandonment)} products have high cart abandonment")
            insights.append(f"  - ACTION: Consider promotions or bundle deals")
        
        return "\n".join(insights)
        
    def export_analysis(self, output_path='merefficencydise_analysis.csv'):
        """Export full analysis to CSV."""
        # Export filtered data for analysis
        export_df_source = self.df_filtered if self.df_filtered is not None else self.df
        
        if export_df_source is not None:
            export_cols = [
                'item_name', 'segment', 'performance_score',
                'items_viewed', 'items_added_to_cart', 'items_purchased',
                'view_to_cart_rate', 'cart_to_purchase_rate', 'view_to_purchase_rate',
                'item_revenue', 'revenue_per_view', 'avg_order_value'
            ]
            
            # Filter to only existing columns
            export_cols = [col for col in export_cols if col in export_df_source.columns]
            
            export_df = export_df_source[export_cols].copy()
            export_df = export_df.sort_values('performance_score', ascending=False)
            
            # Format percentage columns
            pct_cols = ['view_to_cart_rate', 'cart_to_purchase_rate', 'view_to_purchase_rate']
            for col in pct_cols:
                if col in export_df.columns:
                    export_df[col] = export_df[col].apply(lambda x: f"{x:.2%}")
            
            # Format currency columns
            currency_cols = ['item_revenue', 'revenue_per_view', 'avg_order_value']
            for col in currency_cols:
                if col in export_df.columns:
                    export_df[col] = export_df[col].apply(lambda x: f"R{x:,.2f}")
            
            export_df.to_csv(output_path, index=False)
            print(f"\n✓ Analysis exported to {output_path}")
            return export_df
        else:
            print("No data to export")
            return None
    
    def run_full_analysis(self, generate_dashboard=True):
        """Execute complete analysis pipeline."""
        print("\n" + "="*50)
        print("MERCHANDISE ANALYSIS REPORT")
        print("="*50)
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Segment products
        self.segment_products()
        
        # Use filtered data for opportunities
        temp_df = self.df
        self.df = self.df_filtered if self.df_filtered is not None else self.df
        
        # Identify opportunities
        opportunities = self.identify_opportunities()
        
        # Restore df
        self.df = temp_df
        
        # Generate insights
        insights = self.generate_insights()
        print("\n" + insights)
        
        # Print specific recommendations
        print("\n" + "="*50)
        print("ACTIONABLE RECOMMENDATIONS")
        print("="*50)
        
        print("\n🏠 HOMEPAGE PLACEMENT:")
        for item in opportunities['homepage_heroes'][:3]:
            if item:
                print(f"  • {item['item_name']} (Score: {item['performance_score']:.1f})")
        
        print("\n📧 EMAIL CAMPAIGNS:")
        for item in opportunities['email_features'][:3]:
            if item:
                print(f"  • {item['item_name']} (Conversion: {item['view_to_purchase_rate']:.2%})")
        
        print("\n💰 NEEDS PROMOTION:")
        for item in opportunities['promotion_needed'][:3]:
            if item:
                print(f"  • {item['item_name']} (Cart→Purchase: {item['cart_to_purchase_rate']:.2%})")
        
        print("\n🔍 INCREASE EXPOSURE:")
        for item in opportunities['increase_exposure'][:3]:
            if item:
                print(f"  • {item['item_name']} (Views: {int(item['items_viewed'])}, Conv: {item['view_to_purchase_rate']:.2%})")
        
        return self.df