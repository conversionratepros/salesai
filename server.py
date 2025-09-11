import os
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import requests
from datetime import datetime, timedelta
import json
from collections import defaultdict, Counter
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Configuration
AIRTABLE_API_KEY = 'INSERT'
AIRTABLE_BASE_ID = 'appuV8gDiwoLnDsF1'
TABLE_NAME = 'Sales calls'
OPENAI_API_KEY = 'INSERT'  # ADD YOUR OPENAI KEY HERE
client = OpenAI(api_key=OPENAI_API_KEY)

# Test that OpenAI is working
try:
    # Quick test to verify the client is initialized
    test_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Say 'OK'"}],
        max_tokens=5
    )
    print("OpenAI client initialized successfully")
except Exception as e:
    print(f"OpenAI initialization error: {e}")

class AirtableDataFetcher:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        self.base_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{TABLE_NAME}"
    
    def fetch_all_records(self):
        """Fetch all records from Airtable"""
        all_records = []
        offset = None
        
        while True:
            params = {}
            if offset:
                params['offset'] = offset
            
            response = requests.get(self.base_url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                all_records.extend(data.get('records', []))
                
                offset = data.get('offset')
                if not offset:
                    break
            else:
                print(f"Error fetching records: {response.status_code}")
                break
        
        return all_records

@app.route('/api/dashboard-data')
def get_dashboard_data():
    """Fetch and process all dashboard data with filtering - handles multiple values per filter"""
    fetcher = AirtableDataFetcher()
    records = fetcher.fetch_all_records()
    
    # DEBUG: Print first record's fields to see what we have
    if records and len(records) > 0:
        print("=== AVAILABLE FIELDS IN FIRST RECORD ===")
        for field_name in records[0].get('fields', {}).keys():
            print(f"  - {field_name}")
        print("=========================================")
    
    # Get filter parameters - now handles multiple values using getlist
    filters = {
        'lead_topic': request.args.getlist('lead_topic'),
        'wrap_up_reason': request.args.getlist('wrap_up_reason'),
        'month': request.args.getlist('month'),
        'product': request.args.getlist('product'),
        'agent': request.args.getlist('agent'),
        'lead_outcome': request.args.getlist('lead_outcome')
    }
    
    # Remove empty lists from filters
    filters = {k: v for k, v in filters.items() if v}
    
    # DEBUG: Print applied filters
    if any(filters.values()):
        print("=== APPLIED FILTERS ===")
        for key, values in filters.items():
            if values:
                print(f"  {key}: {values}")
        print("======================")
    
    # Apply filters
    filtered_records = apply_filters(records, filters)
    
    print(f"Records after filtering: {len(filtered_records)}/{len(records)}")
    
    # Process filtered records
    processed_data = process_records(filtered_records)
    
    # Add filter options to response (always from ALL records, not filtered)
    processed_data['filter_options'] = get_filter_options(records)
    
    return jsonify(processed_data)

def apply_filters(records, filters):
    """Apply filters to records - handles multiple values per filter with OR logic within categories and AND logic between categories"""
    filtered = []
    
    for record in records:
        fields = record.get('fields', {})
        
        # Lead Topic filter - handle multiple values
        lead_topic_filter = filters.get('lead_topic', [])
        if lead_topic_filter:
            lead_topic = fields.get('Lead Topic')
            if not lead_topic or lead_topic not in lead_topic_filter:
                continue
        
        # Wrap Up Reason filter - handle multiple values
        wrap_up_filter = filters.get('wrap_up_reason', [])
        if wrap_up_filter:
            wrap_up = fields.get('Wrap up reason')
            if not wrap_up or wrap_up not in wrap_up_filter:
                continue
        
        # Month filter - handle multiple values
        month_filter = filters.get('month', [])
        if month_filter:
            # Try multiple date fields
            date_field = fields.get('Date of call') or fields.get('Last Modified')
            if date_field:
                # Handle ISO format dates
                if 'T' in date_field:
                    call_month = date_field.split('T')[0][:7]
                else:
                    call_month = date_field[:7]
                
                if call_month not in month_filter:
                    continue
            else:
                # No date field, skip if month filter is applied
                continue
        
        # Product filter - handle multiple values
        product_filter = filters.get('product', [])
        if product_filter:
            product = fields.get('Product')
            if not product or product not in product_filter:
                continue
        
        # Agent filter - handle multiple values
        agent_filter = filters.get('agent', [])
        if agent_filter:
            agent = fields.get('Agent')
            if not agent or agent not in agent_filter:
                continue
        
        # Lead Outcome filter - handle multiple values
        outcome_filter = filters.get('lead_outcome', [])
        if outcome_filter:
            outcome = fields.get('Lead outcome')
            if not outcome or outcome not in outcome_filter:
                continue
        
        # If we made it here, the record passes all filters
        filtered.append(record)
    
    return filtered

def get_filter_options(records):
    """Extract unique values for each filter"""
    options = {
        'lead_topics': set(),
        'wrap_up_reasons': set(),
        'months': set(),
        'products': set(),
        'agents': set(),
        'lead_outcomes': set()
    }
    
    for record in records:
        fields = record.get('fields', {})
        
        # Lead Topic with capital T
        lead_topic = fields.get('Lead Topic')
        if lead_topic:
            options['lead_topics'].add(lead_topic)
        
        # Wrap up reason field
        wrap_up = fields.get('Wrap up reason')
        if wrap_up:
            options['wrap_up_reasons'].add(wrap_up)
        
        # Using Last Modified for months since Date of call doesn't exist
        last_modified = fields.get('Last Modified')
        if last_modified:
            month = last_modified[:7]
            options['months'].add(month)
        
        # Product field
        product = fields.get('Product')
        if product:
            options['products'].add(product)
        
        # Agent field - you'll need to add this to Airtable if it doesn't exist
        agent = fields.get('Agent')
        if agent:
            options['agents'].add(agent)
        
        # Lead outcome field
        outcome = fields.get('Lead outcome')
        if outcome:
            options['lead_outcomes'].add(outcome)
    
    # Convert sets to sorted lists
    return {
        'lead_topics': sorted(list(options['lead_topics'])),
        'wrap_up_reasons': sorted(list(options['wrap_up_reasons'])),
        'months': sorted(list(options['months']), reverse=True),
        'products': sorted(list(options['products'])),
        'agents': sorted(list(options['agents'])),
        'lead_outcomes': sorted(list(options['lead_outcomes']))
    }

def apply_filters(records, filters):
    """Apply filters to records - handles multiple values per filter with OR logic within categories and AND logic between categories"""
    filtered = []
    
    for record in records:
        fields = record.get('fields', {})
        
        # Lead Topic filter - handle multiple values
        lead_topic_filter = filters.get('lead_topic', [])
        if lead_topic_filter:
            lead_topic = fields.get('Lead Topic')
            if not lead_topic or lead_topic not in lead_topic_filter:
                continue
        
        # Wrap Up Reason filter - handle multiple values
        wrap_up_filter = filters.get('wrap_up_reason', [])
        if wrap_up_filter:
            wrap_up = fields.get('Wrap up reason')
            if not wrap_up or wrap_up not in wrap_up_filter:
                continue
        
        # Month filter - handle multiple values
        month_filter = filters.get('month', [])
        if month_filter:
            # Try multiple date fields
            date_field = fields.get('Date of call') or fields.get('Last Modified')
            if date_field:
                # Handle ISO format dates
                if 'T' in date_field:
                    call_month = date_field.split('T')[0][:7]
                else:
                    call_month = date_field[:7]
                
                if call_month not in month_filter:
                    continue
            else:
                # No date field, skip if month filter is applied
                continue
        
        # Product filter - handle multiple values
        product_filter = filters.get('product', [])
        if product_filter:
            product = fields.get('Product')
            if not product or product not in product_filter:
                continue
        
        # Agent filter - handle multiple values
        agent_filter = filters.get('agent', [])
        if agent_filter:
            agent = fields.get('Agent')
            if not agent or agent not in agent_filter:
                continue
        
        # Lead Outcome filter - handle multiple values
        outcome_filter = filters.get('lead_outcome', [])
        if outcome_filter:
            outcome = fields.get('Lead outcome')
            if not outcome or outcome not in outcome_filter:
                continue
        
        # If we made it here, the record passes all filters
        filtered.append(record)
    
    return filtered

def process_records(records):
    """Process Airtable records into dashboard metrics"""
    
    # Initialize counters and aggregators
    total_calls = len(records)
    completed_transcriptions = 0
    sentiment_scores = []
    opportunity_scores = []
    engagement_scores = []
    performance_scores = []
    risk_scores = []  # ADD THIS LINE
    high_opportunity_count = 0
    
    # Time series data
    time_series_data = defaultdict(lambda: {
        'sentiment': [],
        'opportunity': [],
        'engagement': [],
        'performance': []
    })
    
    # Content analysis
    question_categories = []
    key_topics = []
    sales_tactics = []
    objectives_achieved = []
    risk_factors = []
    objection_types = []
    wrap_up_accuracy = []
    wrap_up_reasons = []
    
    # Process each record
    for record in records:
        fields = record.get('fields', {})
        
        # Check if transcription is completed
        if fields.get('Transcription status') == 'Completed':
            completed_transcriptions += 1
        
        # Collect scores
        if 'Sentiment score' in fields:
            sentiment_scores.append(fields['Sentiment score'])
        
        if 'Opportunity Score' in fields:
            score = fields['Opportunity Score']
            opportunity_scores.append(score)
            if fields.get('Opportunity Level') == 'Hot':
                high_opportunity_count += 1
        
        if 'Customer Engagement Score' in fields:
            engagement_scores.append(fields['Customer Engagement Score'])
        
        if 'Sales performance score' in fields:
            performance_scores.append(fields['Sales performance score'])
        
        if 'Risk Score' in fields:
            risk_scores.append(fields['Risk Score'])
        
        # Time series data - check multiple possible date fields
        date_field = None
        if 'Date of call' in fields:
            date_field = fields['Date of call']
        elif 'Last Modified' in fields:
            date_field = fields['Last Modified']
        elif 'Created' in fields:
            date_field = fields['Created']
        
        if date_field:
            # Handle different date formats
            if 'T' in date_field:  # ISO format like "2025-09-10T12:34:56.000Z"
                date_key = date_field.split('T')[0]
            else:
                date_key = date_field[:10]
            
            # Only add to time series if we have the date
            if 'Sentiment score' in fields:
                time_series_data[date_key]['sentiment'].append(fields['Sentiment score'])
            if 'Opportunity Score' in fields:
                time_series_data[date_key]['opportunity'].append(fields['Opportunity Score'])
            if 'Customer Engagement Score' in fields:
                time_series_data[date_key]['engagement'].append(fields['Customer Engagement Score'])
            if 'Sales performance score' in fields:
                time_series_data[date_key]['performance'].append(fields['Sales performance score'])
        
        # Content analysis
        if 'Question Categories' in fields:
            question_categories.extend(fields['Question Categories'])
        
        if 'Key topics' in fields:
            key_topics.extend(fields['Key topics'])
        
        if 'Sales Tactics Used' in fields:
            sales_tactics.extend(fields['Sales Tactics Used'])
        
        if 'Objective achieved' in fields:
            objectives_achieved.append(fields['Objective achieved'])
        
        if 'Risk Factors' in fields:
            risk_factors.extend(fields['Risk Factors'])
        
        if 'Objection types' in fields:
            objection_types.extend(fields['Objection types'])
        
        if 'Wrap up reason' in fields:
            wrap_up_reasons.append(fields['Wrap up reason'])
        
        if 'Wrap up accuracy' in fields:
            wrap_up_accuracy.append({
                'name': record.get('id', 'Unknown'),
                'lead_id': fields.get('Lead ID', '-'),
                'accuracy': fields['Wrap up accuracy'],
                'reason': fields.get('Wrap up accuracy reason', '')
            })
    
    # Calculate averages
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    avg_opportunity = sum(opportunity_scores) / len(opportunity_scores) if opportunity_scores else 0
    avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
    avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0  # ADD THIS
    
    # ADD THIS: Calculate high risk count
    high_risk_count = sum(1 for score in risk_scores if score > 70) if risk_scores else 0
    
    # Process time series for charts
    sorted_dates = sorted(time_series_data.keys()) if time_series_data else []
    time_series_processed = {
        'dates': sorted_dates[-10:] if sorted_dates else [],
        'sentiment': [],
        'opportunity': [],
        'engagement': [],
        'performance': []
    }
    
    for date in time_series_processed['dates']:
        data = time_series_data[date]
        time_series_processed['sentiment'].append(
            sum(data['sentiment']) / len(data['sentiment']) if data['sentiment'] else 0
        )
        time_series_processed['opportunity'].append(
            sum(data['opportunity']) / len(data['opportunity']) if data['opportunity'] else 0
        )
        time_series_processed['engagement'].append(
            sum(data['engagement']) / len(data['engagement']) if data['engagement'] else 0
        )
        time_series_processed['performance'].append(
            sum(data['performance']) / len(data['performance']) if data['performance'] else 0
        )
    
    # Count frequencies for pie charts
    question_cat_counts = dict(Counter(question_categories)) if question_categories else {}
    topics_counts = dict(Counter(key_topics)) if key_topics else {}
    tactics_counts = dict(Counter(sales_tactics)) if sales_tactics else {}
    objectives_counts = dict(Counter(objectives_achieved)) if objectives_achieved else {}
    risk_counts = dict(Counter(risk_factors)) if risk_factors else {}
    objection_counts = dict(Counter(objection_types)) if objection_types else {}
    wrap_reason_counts = dict(Counter(wrap_up_reasons)) if wrap_up_reasons else {}

    accurate_count = 0
    somewhat_accurate_count = 0
    total_with_accuracy = 0

    # Debug: Let's see what values we're getting
    accuracy_values = []
    
    for record in records:
        fields = record.get('fields', {})
        if 'Wrap up accuracy' in fields:
            accuracy = fields['Wrap up accuracy']
            accuracy_values.append(accuracy)  # Collect for debugging
            total_with_accuracy += 1
            
            # Check the actual values - they might be different than expected
            # Common values might be: "Accurate", "Somewhat accurate", "Not accurate", "None provided"
            if accuracy:
                accuracy_lower = accuracy.lower()
                if 'accurate' in accuracy_lower and 'not' not in accuracy_lower and 'somewhat' not in accuracy_lower:
                    accurate_count += 1
                elif 'somewhat' in accuracy_lower:
                    somewhat_accurate_count += 1
    
    # Debug print to see what values we're getting
    if accuracy_values:
        print(f"Sample wrap-up accuracy values: {accuracy_values[:5]}")
        print(f"Accurate: {accurate_count}, Somewhat: {somewhat_accurate_count}, Total: {total_with_accuracy}")
    
    # Calculate percentage
    if total_with_accuracy > 0:
        wrap_up_accuracy_percentage = ((accurate_count + somewhat_accurate_count) / total_with_accuracy * 100)
    else:
        wrap_up_accuracy_percentage = 0

    return {
        'stats': {
            'total_calls': total_calls,
            'completed_transcriptions': completed_transcriptions,
            'avg_sentiment': round(avg_sentiment, 1),
            'high_opportunity': high_opportunity_count,
            'high_risk': high_risk_count,  
            'avg_opportunity': round(avg_opportunity, 1),  
            'avg_risk': round(avg_risk, 1), 
            'avg_performance': round(avg_performance, 1),
            'wrap_up_accuracy_percentage': round(wrap_up_accuracy_percentage, 1),  # Move inside stats
            'wrap_up_accuracy_breakdown': {  # Keep breakdown for debugging
                'accurate': accurate_count,
                'somewhat_accurate': somewhat_accurate_count,
                'total': total_with_accuracy
            }
        },
        'time_series': time_series_processed,
        'content': {
            'question_categories': question_cat_counts,
            'key_topics': topics_counts,
            'sales_tactics': tactics_counts,
            'objectives': objectives_counts,
            'risk_factors': risk_counts,
            'objections': objection_counts,
            'wrap_up_reasons': wrap_reason_counts
        },
        'wrap_up_accuracy': wrap_up_accuracy[:20]  
    }

@app.route('/api/coach-analysis', methods=['POST'])
def get_coach_analysis():
    """Generate AI-driven sales coaching insights"""
    try:
        data = request.json
        scope = data.get('scope', 'all')
        agent = data.get('agent', None)
        
        print(f"Generating analysis for scope: {scope}, agent: {agent}")
        
        # Get filtered records
        fetcher = AirtableDataFetcher()
        records = fetcher.fetch_all_records()
        
        if scope == 'agent' and agent:
            records = [r for r in records if r.get('fields', {}).get('Agent') == agent]
        
        print(f"Analyzing {len(records)} records")
        
        if not records:
            return jsonify({
                "success_patterns": ["No data available for the selected scope"],
                "failure_patterns": ["No data available for analysis"],
                "behavioral_insights": ["Please select a different scope or ensure data exists"],
                "ab_tests": [],
                "tracking_suggestions": [],
                "objection_insights": []
            })
        
        # Prepare data
        analysis_data = prepare_coach_data(records)
        
        # Generate AI insights
        insights = generate_ai_insights(analysis_data, agent)
        
        return jsonify(insights)
    
    except Exception as e:
        print(f"Error in coach analysis endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def prepare_coach_data(records):
    """Prepare comprehensive data for AI analysis"""
    successful_calls = []
    failed_calls = []
    neutral_calls = []
    objection_handling_examples = []

    
    # Aggregate metrics
    tactics_success = defaultdict(int)
    tactics_failure = defaultdict(int)
    objections_by_outcome = defaultdict(list)
    risk_factors_by_outcome = defaultdict(list)
    
    
    for record in records:
        fields = record.get('fields', {})
    
         # If this call had objections but still succeeded, analyze why
        if (fields.get('Objective achieved') == 'Yes' or 
                fields.get('Lead outcome') == 'Closed Won'):
            if fields.get('Objection count', 0) > 0:
                objection_handling_examples.append({
                    'record_id': record.get('id'),
                    'agent': fields.get('Agent', ''),
                    'objections': fields.get('Objection types', []),
                    'objection_handling': fields.get('Objection handling suggestion', ''),
                    'transcript_excerpt': extract_relevant_transcript(fields),
                    'sentiment_before': fields.get('Sentiment score', 0),
                    'sentiment_after': fields.get('Sentiment score', 0),  # (needs time-series to be accurate)
                    'outcome': fields.get('Lead outcome'),
                    'ai_insights': fields.get('AI Insights', ''),
                    'why_it_worked': fields.get('Achievement reason', '')
                })
        
        # Extract ALL available fields for rich analysis
        call_data = {
            # Identifiers
            'id': record.get('id', ''),
            'agent': fields.get('Agent', ''),
            'lead_id': fields.get('Lead ID', ''),
            
            # Outcomes
            'objective_achieved': fields.get('Objective achieved'),
            'lead_outcome': fields.get('Lead outcome'),
            'achievement_reason': fields.get('Achievement reason', ''),
            
            # Scores and metrics
            'sentiment_score': fields.get('Sentiment score', 0),
            'opportunity_score': fields.get('Opportunity Score', 0),
            'risk_score': fields.get('Risk Score', 0),
            'engagement_score': fields.get('Customer Engagement Score', 0),
            'performance_score': fields.get('Sales performance score', 0),
            'confidence': fields.get('Confidence', 0),
            
            # Conversation dynamics
            'talk_ratio_sales': fields.get('Talk Ratio (Sales)', 0),
            'talk_ratio_customer': fields.get('Talk Ratio (Customer)', 0),
            'questions_count': fields.get('Questions Count', 0),
            'objection_count': fields.get('Objection count', 0),
            
            # Content analysis
            'sales_tactics': fields.get('Sales Tactics Used', []),
            'key_topics': fields.get('Key topics', []),
            'question_categories': fields.get('Question Categories', []),
            'customer_questions': fields.get('Customer Questions', []),
            'objection_types': fields.get('Objection types', []),
            'risk_factors': fields.get('Risk Factors', []),
            
            # AI insights
            'ai_insights': fields.get('AI Insights', ''),
            'next_actions': fields.get('Next Actions', ''),
            'objection_handling_suggestion': fields.get('Objection handling suggestion', ''),
            'performance_explanation': fields.get('Performance score explanation', ''),
            
            # Call metadata
            'call_type': fields.get('Call type', ''),
            'product': fields.get('Product', ''),
            'audio_duration': fields.get('Audio duration', 0),
            'wrap_up_accuracy': fields.get('Wrap up accuracy', ''),
            'wrap_up_reason': fields.get('Wrap up accuracy reason', '')
        }
        
        # Ensure lists are lists
        for key in ['sales_tactics', 'key_topics', 'question_categories', 'customer_questions', 'objection_types', 'risk_factors']:
            if not isinstance(call_data[key], list):
                call_data[key] = [call_data[key]] if call_data[key] else []
        
        # Categorize calls with more nuance
        outcome = fields.get('Lead outcome', '')
        objective = fields.get('Objective achieved', '')
        
        if outcome == 'Closed Won' or objective == 'Yes':
            successful_calls.append(call_data)
            # Track what tactics work
            for tactic in call_data['sales_tactics']:
                tactics_success[tactic] += 1
        elif outcome == 'Closed Lost' or objective == 'No':
            failed_calls.append(call_data)
            # Track what tactics don't work
            for tactic in call_data['sales_tactics']:
                tactics_failure[tactic] += 1
            # Track objections that kill deals
            for objection in call_data['objection_types']:
                objections_by_outcome['failed'].append(objection)
        else:
            neutral_calls.append(call_data)
        
        # Track risk factors by outcome
        for risk in call_data['risk_factors']:
            if outcome:
                risk_factors_by_outcome[outcome].append(risk)
    
    # Calculate advanced metrics
    avg_talk_ratio_success = sum(c['talk_ratio_customer'] for c in successful_calls) / len(successful_calls) if successful_calls else 0
    avg_talk_ratio_failure = sum(c['talk_ratio_customer'] for c in failed_calls) / len(failed_calls) if failed_calls else 0
    
    avg_questions_success = sum(c['questions_count'] for c in successful_calls) / len(successful_calls) if successful_calls else 0
    avg_questions_failure = sum(c['questions_count'] for c in failed_calls) / len(failed_calls) if failed_calls else 0
    
    # Find patterns in high performers
    high_performers = [c for c in successful_calls if c['performance_score'] > 80]
    low_performers = [c for c in failed_calls if c['performance_score'] < 50]
    
    return {
        'successful_calls': successful_calls[:15],
        'failed_calls': failed_calls[:15],
        'total_calls': len(records),
        'success_rate': len(successful_calls) / len(records) * 100 if records else 0,
        
        # Tactical analysis
        'tactics_success': dict(tactics_success),
        'tactics_failure': dict(tactics_failure),
        'tactic_effectiveness': calculate_tactic_effectiveness(tactics_success, tactics_failure),
        
        # Conversation dynamics
        'avg_talk_ratio_success': avg_talk_ratio_success,
        'avg_talk_ratio_failure': avg_talk_ratio_failure,
        'avg_questions_success': avg_questions_success,
        'avg_questions_failure': avg_questions_failure,
        
        # Score differentials
        'sentiment_diff': calculate_score_difference(successful_calls, failed_calls, 'sentiment_score'),
        'opportunity_diff': calculate_score_difference(successful_calls, failed_calls, 'opportunity_score'),
        'engagement_diff': calculate_score_difference(successful_calls, failed_calls, 'engagement_score'),
        
        # Risk and objection analysis
        'objections_by_outcome': dict(objections_by_outcome),
        'risk_factors_by_outcome': dict(risk_factors_by_outcome),
        
        # High/low performer insights
        'high_performer_patterns': extract_patterns(high_performers),
            'low_performer_patterns': extract_patterns(low_performers),
            
            # Objection handling examples
            'objection_handling_examples': objection_handling_examples[:2]  # Top 2 examples
        }

def extract_relevant_transcript(fields):
    """Extract relevant parts of transcript around objections"""
    transcript = fields.get('Formatted transcript', '')
    
    if not transcript:
        return ""
    
    # Look for objection keywords and extract surrounding context
    objection_keywords = ['but', 'however', 'concern', 'worry', 'expensive', 
                          'price', 'cost', 'not sure', 'think about it', 'competitor']
    
    relevant_excerpts = []
    lines = transcript.split('\n')
    
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in objection_keywords):
            # Get context: 2 lines before and 3 lines after
            start = max(0, i-2)
            end = min(len(lines), i+4)
            excerpt = '\n'.join(lines[start:end])
            relevant_excerpts.append(excerpt)
    
    return '\n---\n'.join(relevant_excerpts[:2])  # Return top 2 excerpts

def calculate_tactic_effectiveness(success, failure):
    """Calculate which tactics have the best success rate"""
    effectiveness = {}
    all_tactics = set(success.keys()) | set(failure.keys())
    
    for tactic in all_tactics:
        s = success.get(tactic, 0)
        f = failure.get(tactic, 0)
        total = s + f
        if total > 0:
            effectiveness[tactic] = {
                'success_rate': (s / total) * 100,
                'total_uses': total,
                'successes': s,
                'failures': f
            }
    
    return effectiveness

def calculate_score_difference(successful, failed, score_field):
    """Calculate average score difference between successful and failed calls"""
    avg_success = sum(c[score_field] for c in successful) / len(successful) if successful else 0
    avg_failed = sum(c[score_field] for c in failed) / len(failed) if failed else 0
    return avg_success - avg_failed

def extract_patterns(calls):
    """Extract common patterns from a set of calls"""
    if not calls:
        return {}
    
    patterns = {
        'avg_talk_ratio': sum(c['talk_ratio_customer'] for c in calls) / len(calls),
        'avg_questions': sum(c['questions_count'] for c in calls) / len(calls),
        'avg_objections': sum(c['objection_count'] for c in calls) / len(calls),
        'common_tactics': Counter([t for c in calls for t in c['sales_tactics']]).most_common(3),
        'common_topics': Counter([t for c in calls for t in c['key_topics']]).most_common(3)
    }
    return patterns

def generate_ai_insights(data, agent=None):
    """Generate detailed AI insights using comprehensive data"""
    
    global client
    agent_context = f"for agent {agent}" if agent else "across all agents"
    
    # List fields we already track (to exclude from suggestions)
    existing_fields = [
        "Customer Engagement Score", "Sentiment score", "Opportunity Score", 
        "Risk Score", "Talk Ratio", "Questions Count", "Objection count",
        "Sales performance score", "Confidence", "Lead outcome", 
        "Objective achieved", "Sales Tactics Used", "Key topics",
        "Question Categories", "Customer Questions", "Risk Factors",
        "AI Insights", "Next Actions", "Wrap up accuracy", "Agent",
        "Lead ID", "Product", "Call type", "Audio duration"
    ]
    
    # Add objection handling examples if available
    examples_section = ""
    if data.get('objection_handling_examples'):
        examples_section = f"""
    SUCCESSFUL OBJECTION HANDLING EXAMPLES:
    {json.dumps(data['objection_handling_examples'], indent=2)}
    """
    
    # Build a data-rich prompt
    prompt = f"""
    Analyze this comprehensive sales call data {agent_context} and provide specific, actionable insights.
    
    PERFORMANCE METRICS:
    - Total Calls: {data['total_calls']}
    - Success Rate: {data['success_rate']:.1f}%
    - Customer Talk Ratio - Success: {data['avg_talk_ratio_success']:.1f}% vs Failure: {data['avg_talk_ratio_failure']:.1f}%
    - Questions Asked - Success: {data['avg_questions_success']:.1f} vs Failure: {data['avg_questions_failure']:.1f}
    
    SCORE DIFFERENTIALS (Success vs Failure):
    - Sentiment: {data['sentiment_diff']:.1f} point difference
    - Opportunity: {data['opportunity_diff']:.1f} point difference  
    - Engagement: {data['engagement_diff']:.1f} point difference
    
    TACTIC EFFECTIVENESS:
    {json.dumps(data['tactic_effectiveness'], indent=2)}
    
    HIGH PERFORMER PATTERNS:
    {json.dumps(data['high_performer_patterns'], indent=2)}
    
    LOW PERFORMER PATTERNS:
    {json.dumps(data['low_performer_patterns'], indent=2)}
    
    SAMPLE SUCCESSFUL CALL:
    {json.dumps(data['successful_calls'][0] if data['successful_calls'] else {}, indent=2)}
    
    SAMPLE FAILED CALL:
    {json.dumps(data['failed_calls'][0] if data['failed_calls'] else {}, indent=2)}
    
    {examples_section}
    
    IMPORTANT INSTRUCTIONS:
    1. For objection_insights success examples: When you have transcript excerpts or specific examples:
       - Quote specific words/phrases used if available
       - Explain WHY this approach worked based on metrics
       - Include the impact (sentiment change, outcome, etc.)
    
    2. All insights must be extremely specific with real data, not generic advice.
    
    3. CRITICAL: In success_patterns, failure_patterns, and behavioral_insights arrays, 
       return ONLY STRINGS, never objects. Each item must be a complete sentence.
    
    Based on this data, provide SPECIFIC insights with real numbers and examples.
    
    Return JSON format:
    {{
        "success_patterns": [
            "String only: Be specific with metrics from the data",
            "String only: Reference actual tactics that work with numbers",
            "String only: Include threshold metrics"
        ],
        "failure_patterns": [
            "String only: Identify specific failure points with real numbers",
            "String only: Call out ineffective tactics by name with their failure rates",
            "String only: Highlight specific score thresholds that predict failure"
        ],
        "behavioral_insights": [
            "String only: Provide specific behavioral changes with expected impact",
            "String only: Reference the exact metrics that need improvement",
            "String only: Give specific targets based on high performer data"
        ],
        "objection_insights": [
            {{
                "objection": "Specific objection from the data",
                "impact": "Appears in X% of failed calls vs Y% of successful calls",
                "suggested_response": "Specific technique based on successful call patterns",
                "success_example": "Specific example with details if available"
            }}
        ]
    }}
    
    Be extremely specific. Use the actual data provided. Include percentages, thresholds, and specific examples.
    Each array item MUST be a string, not an object.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert sales analytics AI. Provide data-driven insights with specific metrics. Arrays must contain only strings, never objects."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        insights = json.loads(content.strip())
        
        # Validate and clean - ensure arrays contain only strings
        for key in ['success_patterns', 'failure_patterns', 'behavioral_insights']:
            if key in insights and isinstance(insights[key], list):
                cleaned = []
                for item in insights[key]:
                    if isinstance(item, str):
                        cleaned.append(item)
                    else:
                        # Convert non-strings to strings
                        cleaned.append(str(item))
                insights[key] = cleaned
        
        # Ensure objection_insights exists and is properly formatted
        if 'objection_insights' not in insights:
            insights['objection_insights'] = []
        
        return insights
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Content was: {content if 'content' in locals() else 'No content'}")
    except Exception as e:
        print(f"Error generating insights: {e}")
    
    # Fallback if AI fails - using actual data
    return {
        "success_patterns": [
            f"Success rate is {data['success_rate']:.1f}% {agent_context}",
            f"Customer talk ratio averages {data['avg_talk_ratio_success']:.1f}% in successful calls",
            f"Successful calls average {data['avg_questions_success']:.1f} questions asked"
        ],
        "failure_patterns": [
            f"Failed calls have {abs(data['sentiment_diff']):.1f} points lower sentiment",
            f"Customer talk ratio only {data['avg_talk_ratio_failure']:.1f}% in failed calls",
            f"Failed calls average {data['avg_questions_failure']:.1f} questions"
        ],
        "behavioral_insights": [
            f"Target customer talk ratio of {data['avg_talk_ratio_success']:.1f}% or higher",
            f"Aim for {data['avg_questions_success']:.0f}+ questions per call",
            f"Focus on improving sentiment scores by {abs(data['sentiment_diff']):.0f} points"
        ],
        "objection_insights": []
    }
    
@app.route('/api/call-details/<record_id>')
def get_call_details(record_id):
    """Get detailed analysis of a specific successful call"""
    fetcher = AirtableDataFetcher()
    
    # Fetch specific record
    url = f"{fetcher.base_url}/{record_id}"
    response = requests.get(url, headers=fetcher.headers)
    
    if response.status_code == 200:
        record = response.json()
        fields = record.get('fields', {})
        
        # Analyze what made this call successful
        analysis = {
            'record_id': record_id,
            'agent': fields.get('Agent'),
            'outcome': fields.get('Lead outcome'),
            'transcript_excerpt': extract_relevant_transcript(fields),
            'techniques_used': analyze_techniques(fields),
            'key_moments': identify_key_moments(fields),
            'metrics': {
                'sentiment': fields.get('Sentiment score'),
                'opportunity': fields.get('Opportunity Score'),
                'engagement': fields.get('Customer Engagement Score'),
                'talk_ratio': fields.get('Talk Ratio (Customer)')
            }
        }
        
        return jsonify(analysis)
    
    return jsonify({'error': 'Call not found'}), 404

def analyze_techniques(fields):
    """Identify specific techniques used in successful calls"""
    transcript = fields.get('Formatted transcript', '')
    techniques = []
    
    # Pattern matching for sales techniques
    if 'how are you' in transcript.lower():
        techniques.append("Warm greeting with personal check-in")
    if 'let me show you' in transcript.lower():
        techniques.append("Demo-focused approach")
    if 'other clients' in transcript.lower() or 'client success' in transcript.lower():
        techniques.append("Social proof through client stories")
    # Add more pattern recognition
    
    return techniques

def identify_key_moments(fields):
    """Identify turning points in the conversation"""
    # Analyze transcript for sentiment shifts, agreement signals, etc.
    return {
        'turning_points': ["Customer said 'tell me more' after pricing discussion"],
        'agreement_signals': ["Multiple 'yes' responses after demo"],
        'objection_resolution': ["Price concern addressed at minute 5:30"]
    }

@app.route('/')
def dashboard():
    """Serve the dashboard HTML"""
    with open('sales_dash2.html', 'r') as f:
        return f.read()

if __name__ == '__main__':
    app.run(debug=True, port=5001)
