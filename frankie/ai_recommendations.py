# ai_recommendations.py
from openai import OpenAI
import pandas as pd
import json
from datetime import datetime
import os
from typing import Dict, List, Optional

class FrankieAIRecommendations:
    """
    AI-powered product recommendations using OpenAI GPT API
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the AI recommendations module
        
        Args:
            api_key: OpenAI API key (or set as environment variable OPENAI_API_KEY)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                print("✓ OpenAI client initialized successfully")
            except Exception as e:
                print(f"❌ Error initializing OpenAI client: {e}")
                raise ValueError(f"Failed to initialize OpenAI client: {e}")
        else:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    def prepare_product_data_for_ai(self, df: pd.DataFrame, top_n: int = 150) -> str:
        """
        Prepare product data in a format suitable for AI analysis
        
        Args:
            df: DataFrame with product data
            top_n: Number of top products by views to analyze
        
        Returns:
            Formatted string of product data for AI
        """
        # Get top products by views
        df_top = df.nlargest(top_n, 'items_viewed') if len(df) > top_n else df
        
        # Create a summary for AI
        data_summary = []
        data_summary.append(f"Total Products Analyzed: {len(df_top)}")
        data_summary.append(f"Date: {datetime.now().strftime('%B %d, %Y')}\n")
        
        # Add product details
        data_summary.append("PRODUCT PERFORMANCE DATA:")
        data_summary.append("-" * 50)
        
        for _, row in df_top.iterrows():
            product_info = [
                f"Product: {row['item_name']}",
                f"  Views: {int(row['items_viewed'])}",
                f"  Cart Rate: {row['view_to_cart_rate']:.1%}",
                f"  Conversion Rate: {row['view_to_purchase_rate']:.1%}",
                f"  Revenue: R{row['item_revenue']:,.2f}",
                f"  Segment: {row.get('segment', 'Unknown')}",
                ""
            ]
            data_summary.extend(product_info)
        
        return "\n".join(data_summary)
    
    def generate_recommendations(self, df: pd.DataFrame, business_context: str = "") -> Dict:
        """
        Generate AI-powered recommendations based on product data
        
        Args:
            df: DataFrame with product data
            business_context: Optional business context to help AI understand the store
        
        Returns:
            Dictionary with recommendations and metadata
        """
        # Check if client is initialized
        if self.client is None:
            return {
                'success': False,
                'recommendations': self.get_fallback_recommendations(df),
                'timestamp': datetime.now().isoformat(),
                'error': 'OpenAI client not initialized'
            }
        
        # Prepare data for AI
        product_data = self.prepare_product_data_for_ai(df)
        
        # Create the prompt
        prompt = f"""You are Frankie, an expert e-commerce merchandising AI analyst. Analyze the following product performance data and provide actionable recommendations.

{business_context}

{product_data}

Based on this data, provide:
1. THREE KEY INSIGHTS about the current product performance
2. FIVE SPECIFIC ACTIONS the merchant should take immediately to improve sales
3. TWO WARNINGS about potential issues you've identified

Format your response in a conversational but professional tone. Be specific with product names and numbers. Focus on actionable advice that can be implemented quickly.

Important guidelines:
- Reference specific products by name when making recommendations
- Use the actual metrics (views, conversion rates, revenue) to support your points
- Prioritize recommendations that will have the biggest revenue impact
- Be direct and confident in your recommendations
- Keep each point concise (2-3 sentences max)"""

        try:
            # Call OpenAI API with the new client format
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using the more cost-effective model
                messages=[
                    {"role": "system", "content": "You are Frankie, an expert e-commerce merchandising AI that provides actionable, data-driven recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            # Extract the recommendation text
            recommendation_text = response.choices[0].message.content
            
            # Debug: Print what we got from the AI
            print(f"AI Response received ({len(recommendation_text)} chars):")
            print("-" * 50)
            print(recommendation_text[:500])  # Print first 500 chars for debugging
            print("-" * 50)
            
            # Return structured response
            return {
                'success': True,
                'recommendations': recommendation_text,
                'timestamp': datetime.now().isoformat(),
                'products_analyzed': len(df),
                'model_used': response.model,
                'tokens_used': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            print(f"Error generating AI recommendations: {e}")
            return {
                'success': False,
                'recommendations': self.get_fallback_recommendations(df),
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_fallback_recommendations(self, df: pd.DataFrame) -> str:
        """
        Generate fallback recommendations if AI API fails
        
        Args:
            df: DataFrame with product data
        
        Returns:
            String with basic recommendations
        """
        recommendations = []
        
        # Get basic insights from data
        top_performer = df.nlargest(1, 'item_revenue').iloc[0] if len(df) > 0 else None
        hidden_gems = df[(df['segment'] == 'Hidden Gems')] if 'segment' in df.columns else pd.DataFrame()
        underperformers = df[(df['segment'] == 'Underperformers')] if 'segment' in df.columns else pd.DataFrame()
        
        recommendations.append("**Frankie's Analysis** (Generated locally)")
        recommendations.append("")
        
        if top_performer is not None:
            recommendations.append(f"💡 **Top Performer Alert**: '{top_performer['item_name']}' is crushing it with R{top_performer['item_revenue']:,.0f} in revenue. Feature this prominently on your homepage and in email campaigns.")
            recommendations.append("")
        
        if not hidden_gems.empty:
            top_gem = hidden_gems.iloc[0]
            recommendations.append(f"💎 **Hidden Gem Found**: '{top_gem['item_name']}' has a {top_gem['view_to_purchase_rate']:.1%} conversion rate but only {int(top_gem['items_viewed'])} views. Increase exposure immediately for quick wins.")
            recommendations.append("")
        
        if not underperformers.empty and len(underperformers) > 3:
            recommendations.append(f"⚠️ **Attention Needed**: You have {len(underperformers)} underperforming products. Consider bundling them with top performers or running targeted promotions.")
            recommendations.append("")
        
        # General recommendations
        avg_conversion = df['view_to_purchase_rate'].mean() if 'view_to_purchase_rate' in df.columns else 0
        if avg_conversion < 0.02:
            recommendations.append("📈 **Conversion Optimization**: Your average conversion rate is below 2%. Focus on improving product descriptions, images, and pricing to boost conversions.")
        
        return "\n".join(recommendations)
    
    def format_for_dashboard(self, recommendations_response: Dict) -> str:
        """
        Format AI recommendations for dashboard display
        
        Args:
            recommendations_response: Response from generate_recommendations
        
        Returns:
            HTML-formatted string for dashboard
        """
        if not recommendations_response['success']:
            error_msg = recommendations_response.get('error', 'Unknown error')
            return f"""
            <div style="padding: 20px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px;">
                <strong>Frankie is thinking locally...</strong><br>
                <small>AI connection issue: {error_msg}</small>
            </div>
            <div style="padding: 10px;">
                {recommendations_response['recommendations'].replace('**', '<strong>').replace('</strong><strong>', '').replace('\n', '<br>')}
            </div>
            """
        
        # Format successful AI recommendations
        text = recommendations_response['recommendations']
        
        # Parse the text to extract sections
        sections = []
        current_section = None
        current_bullets = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if it's a heading (KEY INSIGHTS or WARNINGS)
            if 'KEY INSIGHTS' in line.upper():
                if current_section:
                    sections.append((current_section, current_bullets))
                current_section = 'Key Insights'
                current_bullets = []
            elif 'WARNINGS' in line.upper() or 'WARNING' in line.upper():
                if current_section:
                    sections.append((current_section, current_bullets))
                current_section = 'Warnings'
                current_bullets = []
            elif line.startswith('-') or line.startswith('•'):
                # It's a bullet point
                bullet_text = line.lstrip('-•').strip()
                if bullet_text:
                    current_bullets.append(bullet_text)
            elif current_bullets and not any(keyword in line.upper() for keyword in ['INSIGHTS', 'WARNINGS', 'ACTIONS']):
                # Continuation of previous bullet
                if current_bullets:
                    current_bullets[-1] += ' ' + line
        
        # Add the last section
        if current_section:
            sections.append((current_section, current_bullets))
        
        # Build HTML output
        html_output = []
        
        for section_title, bullets in sections:
            if section_title == 'Key Insights':
                icon = '💡'
                color = '#3b82f6'
            else:  # Warnings
                icon = '⚠️'
                color = '#dc3545'
            
            html_output.append(f"""
            <div style="margin-bottom: 20px;">
                <h4 style="font-size: 16px; font-weight: 600; color: {color}; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                    <span>{icon}</span> {section_title}
                </h4>
                <ul style="margin: 0; padding-left: 20px; color: #333;">
            """)
            
            # Add only the first 2 bullets
            for bullet in bullets[:2]:
                html_output.append(f'<li style="margin-bottom: 8px; line-height: 1.5;">{bullet}</li>')
            
            html_output.append('</ul></div>')
        
        # If parsing failed, fall back to simple formatting
        if not sections:
            html_output = [f"<div style='padding: 10px;'>{text.replace('**', '<strong>').replace('</strong><strong>', '').replace('\n', '<br>')}</div>"]
        
        # Add metadata footer
        footer = f"""
        <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #e0e0e0; font-size: 12px; color: #666;">
            <em>Analysis completed at {datetime.fromisoformat(recommendations_response['timestamp']).strftime('%I:%M %p')} • 
            {recommendations_response['products_analyzed']} products analyzed • 
            Model: {recommendations_response.get('model_used', 'gpt-4o-mini')}</em>
        </div>
        """
        
        return ''.join(html_output) + footer


# Note: This section shows how to integrate with Flask
# The actual implementation is in frankiedash.py
# This is just documentation for reference


# Example usage
if __name__ == "__main__":
    # Test the module
    print("Testing AI Recommendations Module...")
    
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
    else:
        print(f"✓ API Key found: {api_key[:10]}...")
        
        try:
            # Initialize the recommender
            ai_recommender = FrankieAIRecommendations()
            print("✓ AI Recommender initialized successfully")
            
            # Create sample data
            sample_df = pd.DataFrame({
                'item_name': ['Product A', 'Product B', 'Product C'],
                'items_viewed': [1000, 500, 200],
                'view_to_cart_rate': [0.10, 0.05, 0.15],
                'view_to_purchase_rate': [0.02, 0.01, 0.03],
                'item_revenue': [20000, 5000, 6000],
                'segment': ['Stars', 'Underperformers', 'Hidden Gems']
            })
            
            print("Testing recommendation generation...")
            result = ai_recommender.generate_recommendations(sample_df)
            
            if result['success']:
                print("✓ Recommendations generated successfully!")
                print("\n--- AI Recommendations ---")
                print(result['recommendations'])
            else:
                print(f"❌ Failed to generate recommendations: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")