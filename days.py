import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import base64
import requests
from io import StringIO

st.set_page_config(layout="wide", page_title="Leads Analysis Dashboard", page_icon="📊")

# Function to load data from GitHub
def load_github_data(github_url):
    try:
        # Convert GitHub URL to raw URL
        if "github.com" in github_url and "/blob/" in github_url:
            raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        else:
            raw_url = github_url
            
        response = requests.get(raw_url)
        
        if response.status_code == 200:
            content = StringIO(response.text)
            df = pd.read_csv(content, encoding='utf-8')
            return df
        else:
            st.error(f"Failed to load data: HTTP Status {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to clean column names
def clean_column_names(df):
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df

# Function to clean the data
def clean_data(df):
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert date columns if they exist
    date_columns = [col for col in df_clean.columns if 'date' in col.lower()]
    for col in date_columns:
        try:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        except:
            pass
            
    # Ensure consistency in column names
    rename_dict = {}
    for col in df_clean.columns:
        if 'regist' in col.lower() and 'name' in col.lower():
            rename_dict[col] = 'Registered Name'
        elif 'regist' in col.lower() and 'email' in col.lower():
            rename_dict[col] = 'Registered Email'
        elif 'regist' in col.lower() and 'mobile' in col.lower():
            rename_dict[col] = 'Registered Mobile'
        elif 'regist' in col.lower() and 'date' in col.lower():
            rename_dict[col] = 'User Registration Date'
        elif 'primary' in col.lower() and 'campaign' in col.lower():
            rename_dict[col] = 'Primary Registration Campaign'
        elif col.lower() == 'region':
            rename_dict[col] = 'Region'
        elif col.lower() == 'country':
            rename_dict[col] = 'Country'
        elif 'form' in col.lower() and 'stage' in col.lower():
            rename_dict[col] = 'Form stage'
        elif 'income' in col.lower() and 'household' in col.lower():
            rename_dict[col] = 'Annual Household Income (USD)'
        elif 'payment' in col.lower() and 'status' in col.lower():
            rename_dict[col] = 'Payment Status'
    
    # Only rename if the column exists
    rename_dict = {k: v for k, v in rename_dict.items() if k in df_clean.columns}
    if rename_dict:
        df_clean = df_clean.rename(columns=rename_dict)
            
    return df_clean

# Function to calculate key metrics
def calculate_metrics(df):
    metrics = {}
    
    # Total leads
    metrics['total_leads'] = len(df)
    
    # Form progression rate (users who moved beyond personal information)
    if 'Form stage' in df.columns:
        form_stages = df['Form stage'].value_counts()
        personal_info_count = form_stages.get('PERSONAL INFORMATION', 0)
        metrics['form_progression_rate'] = round(((metrics['total_leads'] - personal_info_count) / metrics['total_leads'] * 100), 1) if metrics['total_leads'] > 0 else 0
    else:
        metrics['form_progression_rate'] = 0
    
    # Completion rate (approved payments)
    if 'Payment Status' in df.columns:
        approved_payments = df[df['Payment Status'] == 'PAYMENT APPROVED'].shape[0]
        metrics['completion_rate'] = round((approved_payments / metrics['total_leads'] * 100), 1) if metrics['total_leads'] > 0 else 0
        metrics['approved_payments'] = approved_payments
    else:
        metrics['completion_rate'] = 0
        metrics['approved_payments'] = 0
    
    # Count unique countries
    if 'Country' in df.columns:
        metrics['countries_count'] = df['Country'].nunique()
    else:
        metrics['countries_count'] = 0
    
    # Top campaign
    if 'Primary Registration Campaign' in df.columns:
        campaign_counts = df['Primary Registration Campaign'].value_counts()
        if not campaign_counts.empty:
            metrics['top_campaign'] = campaign_counts.index[0]
            metrics['top_campaign_count'] = campaign_counts.iloc[0]
            metrics['top_campaign_percentage'] = round((campaign_counts.iloc[0] / metrics['total_leads'] * 100), 1) if metrics['total_leads'] > 0 else 0
        else:
            metrics['top_campaign'] = 'N/A'
            metrics['top_campaign_count'] = 0
            metrics['top_campaign_percentage'] = 0
    else:
        metrics['top_campaign'] = 'N/A'
        metrics['top_campaign_count'] = 0
        metrics['top_campaign_percentage'] = 0
    
    # Top country
    if 'Country' in df.columns:
        country_counts = df['Country'].value_counts()
        if not country_counts.empty:
            metrics['top_country'] = country_counts.index[0]
            metrics['top_country_count'] = country_counts.iloc[0]
            metrics['top_country_percentage'] = round((country_counts.iloc[0] / metrics['total_leads'] * 100), 1) if metrics['total_leads'] > 0 else 0
        else:
            metrics['top_country'] = 'N/A'
            metrics['top_country_count'] = 0
            metrics['top_country_percentage'] = 0
    else:
        metrics['top_country'] = 'N/A'
        metrics['top_country_count'] = 0
        metrics['top_country_percentage'] = 0
    
    # Top region
    if 'Region' in df.columns:
        region_counts = df['Region'].value_counts()
        if not region_counts.empty:
            metrics['top_region'] = region_counts.index[0]
            metrics['top_region_count'] = region_counts.iloc[0]
            metrics['top_region_percentage'] = round((region_counts.iloc[0] / metrics['total_leads'] * 100), 1) if metrics['total_leads'] > 0 else 0
        else:
            metrics['top_region'] = 'N/A'
            metrics['top_region_count'] = 0
            metrics['top_region_percentage'] = 0
    else:
        metrics['top_region'] = 'N/A'
        metrics['top_region_count'] = 0
        metrics['top_region_percentage'] = 0
        
    return metrics

# Function to analyze form stages
def analyze_form_stages(df):
    if 'Form stage' not in df.columns:
        return None, None
    
    # Count leads in each stage
    stage_counts = df['Form stage'].value_counts().reset_index()
    stage_counts.columns = ['stage', 'count']
    
    # Create a funnel chart
    funnel_data = []
    
    # Define the order of stages
    stage_order = [
        'PERSONAL INFORMATION', 
        'ACADEMIC INFORMATION & EXTRACURRICULARS', 
        'ESSAYS', 
        'BOOK YOUR TETR TRIAL - APTITUDE TEST'
    ]
    
    # Add any stages that might not be in our predefined list
    for stage in stage_counts['stage'].unique():
        if stage not in stage_order:
            stage_order.append(stage)
    
    # Create sorted data for funnel
    for stage in stage_order:
        count = stage_counts[stage_counts['stage'] == stage]['count'].sum() if stage in stage_counts['stage'].values else 0
        funnel_data.append({'stage': stage, 'count': count})
    
    funnel_df = pd.DataFrame(funnel_data)
    
    return stage_counts, funnel_df

# Function to analyze countries and regions
def analyze_geography(df):
    country_data = None
    region_data = None
    
    if 'Country' in df.columns:
        country_data = df['Country'].value_counts().reset_index()
        country_data.columns = ['country', 'count']
        country_data['percentage'] = round((country_data['count'] / len(df) * 100), 1)
    
    if 'Region' in df.columns:
        region_data = df['Region'].value_counts().reset_index()
        region_data.columns = ['region', 'count']
        region_data['percentage'] = round((region_data['count'] / len(df) * 100), 1)
    
    return country_data, region_data

# Function to analyze campaigns for leads and conversions
def analyze_campaigns(df):
    if 'Primary Registration Campaign' not in df.columns:
        return None, None, None

    # Get all campaigns and their counts
    lead_data = df['Primary Registration Campaign'].value_counts().reset_index()
    lead_data.columns = ['campaign', 'count']
    
    # Calculate conversion rates for each campaign
    conversion_data = []
    
    for campaign in lead_data['campaign']:
        campaign_df = df[df['Primary Registration Campaign'] == campaign]
        if 'Payment Status' in campaign_df.columns and len(campaign_df) > 0:
            completed = campaign_df[campaign_df['Payment Status'] == 'PAYMENT APPROVED'].shape[0]
            rate = round((completed / len(campaign_df) * 100), 1)
            conversion_data.append({
                'campaign': campaign, 
                'conversion_rate': rate,
                'completed': completed,
                'total': len(campaign_df)
            })
    
    conversion_df = pd.DataFrame(conversion_data) if conversion_data else None
    
    # Get detailed campaign stage data
    campaign_stage_data = {}
    
    for campaign in lead_data['campaign']:
        campaign_df = df[df['Primary Registration Campaign'] == campaign]
        
        # Initialize counts
        stage_counts = {
            'total': len(campaign_df),
            'PERSONAL INFORMATION': 0,
            'ACADEMIC INFORMATION & EXTRACURRICULARS': 0,
            'ESSAYS': 0,
            'BOOK YOUR TETR TRIAL - APTITUDE TEST': 0
        }
        
        # Count stages if applicable
        if 'Form stage' in campaign_df.columns:
            for stage in ['PERSONAL INFORMATION', 'ACADEMIC INFORMATION & EXTRACURRICULARS', 'ESSAYS', 'BOOK YOUR TETR TRIAL - APTITUDE TEST']:
                stage_counts[stage] = campaign_df[campaign_df['Form stage'] == stage].shape[0]
        
        # Add conversion rate
        if 'Payment Status' in campaign_df.columns:
            completed = campaign_df[campaign_df['Payment Status'] == 'PAYMENT APPROVED'].shape[0]
            conversion = round((completed / len(campaign_df) * 100), 1) if len(campaign_df) > 0 else 0
            stage_counts['conversion_rate'] = conversion
        else:
            stage_counts['conversion_rate'] = 0
            
        campaign_stage_data[campaign] = stage_counts
    
    return lead_data, conversion_df, campaign_stage_data

# Function to analyze income data
def analyze_income(df):
    if 'Annual Household Income (USD)' not in df.columns or 'Payment Status' not in df.columns:
        return None
    
    # Count leads in each income bracket
    income_data = df['Annual Household Income (USD)'].value_counts().reset_index()
    income_data.columns = ['income_bracket', 'total_leads']
    
    # Count completed applications in each income bracket
    income_completed = df[df['Payment Status'] == 'PAYMENT APPROVED'].groupby('Annual Household Income (USD)').size().reset_index()
    income_completed.columns = ['income_bracket', 'completed_leads']
    
    # Merge the data
    income_analysis = pd.merge(income_data, income_completed, on='income_bracket', how='left')
    income_analysis['completed_leads'] = income_analysis['completed_leads'].fillna(0).astype(int)
    income_analysis['conversion_rate'] = round((income_analysis['completed_leads'] / income_analysis['total_leads'] * 100), 1)
    
    return income_analysis

# Function to create a download link for the processed CSV
def get_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download processed CSV</a>'
    return href

# Sidebar for dashboard setup
st.sidebar.title("Leads Analysis Dashboard")
st.sidebar.subheader("Data Source")

# Default GitHub URL
default_url = "https://github.com/Harsh220802/Daily_Leads/blob/main/Leads.csv"
github_url = st.sidebar.text_input("GitHub CSV URL", default_url)

# Button to load data
load_button = st.sidebar.button("Load Data")

# Main dashboard content
st.title("📊 Leads Analysis Dashboard")

# Initialize or load data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    
if load_button or ('data_loaded' in st.session_state and st.session_state.data_loaded):
    # Load data from GitHub
    if not st.session_state.data_loaded or load_button:
        with st.spinner('Loading data from GitHub...'):
            df_original = load_github_data(github_url)
            if df_original is not None:
                st.session_state.df_original = df_original
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data. Please check the URL and try again.")
                st.session_state.data_loaded = False
    else:
        df_original = st.session_state.df_original
    
    if st.session_state.data_loaded:
        # Process and clean the data
        df = clean_column_names(df_original)
        df = clean_data(df)
        
        # Add a "Processed Data" section in the sidebar
        st.sidebar.subheader("Processed Data")
        st.sidebar.markdown(get_csv_download_link(df, "processed_leads.csv"), unsafe_allow_html=True)
        
        # Date filter if date column exists
        if 'User Registration Date' in df.columns:
            st.sidebar.subheader("Date Filter")
            
            # Convert to datetime if not already
            if df['User Registration Date'].dtype != 'datetime64[ns]':
                try:
                    df['User Registration Date'] = pd.to_datetime(df['User Registration Date'], errors='coerce')
                except:
                    st.sidebar.warning("Could not convert date column to datetime format")
            
            # Get min and max dates from the data
            if pd.api.types.is_datetime64_dtype(df['User Registration Date']):
                min_date = df['User Registration Date'].min().date()
                max_date = df['User Registration Date'].max().date()
                
                # Date range selector
                start_date = st.sidebar.date_input("Start Date", min_date)
                end_date = st.sidebar.date_input("End Date", max_date)
                
                # Filter data based on date range
                df = df[(df['User Registration Date'].dt.date >= start_date) & 
                         (df['User Registration Date'].dt.date <= end_date)]
                
                st.info(f"Showing data from {start_date} to {end_date}")
        
        # Calculate metrics
        metrics = calculate_metrics(df)
        
        # Show insights for the current date data
        if 'User Registration Date' in df.columns:
            current_date = pd.to_datetime('today').date()
            today_df = df[df['User Registration Date'].dt.date == current_date] if pd.api.types.is_datetime64_dtype(df['User Registration Date']) else pd.DataFrame()
            
            if not today_df.empty:
                st.subheader(f"Today's Insights ({current_date})")
                st.write(f"New leads today: {len(today_df)}")
            
            # Get latest date in the data
            if pd.api.types.is_datetime64_dtype(df['User Registration Date']):
                latest_date = df['User Registration Date'].max().date()
                st.write(f"Data last updated: {latest_date}")
        
        # Display Key Metrics in a row
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Leads", metrics['total_leads'])
        
        with col2:
            st.metric("Form Progression Rate", f"{metrics['form_progression_rate']}%")
            st.caption("% leads that moved to stage 2")
        
        with col3:
            st.metric("Completion Rate", f"{metrics['completion_rate']}%")
            st.caption("% leads with approved payment")
        
        with col4:
            st.metric("Countries", metrics['countries_count'])
        
        # Top performers
        st.subheader("Top Performers")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Top Campaign", metrics['top_campaign'])
            st.caption(f"Represents {metrics['top_campaign_percentage']}% of leads")
        
        with col2:
            st.metric("Top Country", metrics['top_country'])
            st.caption(f"Represents {metrics['top_country_percentage']}% of leads")
        
        with col3:
            st.metric("Top Region", metrics['top_region'])
            st.caption(f"Represents {metrics['top_region_percentage']}% of leads")
        
        # Form Stage Analysis
        st.subheader("Form Stage Analysis")
        
        stage_counts, funnel_df = analyze_form_stages(df)
        
        if stage_counts is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Stage Distribution")
                
                # Allow selection of a form stage to see details
                form_stages = stage_counts['stage'].tolist()
                selected_stage = st.selectbox("Select Form Stage", form_stages)
                
                # Show total users in that stage
                stage_users = stage_counts[stage_counts['stage'] == selected_stage]['count'].values[0]
                st.metric("Total Users", stage_users)
                
                # Show top sources for selected stage
                if 'Country' in df.columns:
                    stage_df = df[df['Form stage'] == selected_stage]
                    top_countries = stage_df['Country'].value_counts().head(3)
                    
                    st.write(f"Top Sources for \"{selected_stage}\":")
                    
                    if 'Country' in df.columns:
                        country_text = f"Country: {top_countries.index[0]}" if not top_countries.empty else "Country: Not Specified"
                        st.markdown(f"• {country_text}")
                    
                    if 'Region' in df.columns:
                        top_regions = stage_df['Region'].value_counts().head(1)
                        region_text = f"Region: {top_regions.index[0]}" if not top_regions.empty else "Region: Not Specified"
                        st.markdown(f"• {region_text}")
                    
                    if 'Primary Registration Campaign' in df.columns:
                        top_campaigns = stage_df['Primary Registration Campaign'].value_counts().head(1)
                        campaign_text = f"Campaign: {top_campaigns.index[0]}" if not top_campaigns.empty else "Campaign: Not Specified"
                        st.markdown(f"• {campaign_text}")
            
            with col2:
                st.subheader("Application Funnel")
                
                # Create funnel chart (without percentages)
                fig = go.Figure(go.Funnel(
                    y=funnel_df['stage'],
                    x=funnel_df['count'],
                    textinfo="value"  # Only show values, not percentages
                ))
                
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Form stage information not available in the uploaded data.")
        
        # Campaign Performance Analysis
        st.subheader("Campaign Performance Analysis")
        
        campaign_lead_data, campaign_conversion_data, campaign_stage_data = analyze_campaigns(df)
        
        if campaign_lead_data is not None:
            tabs = st.tabs(["Top Campaigns by Leads", "Top Campaigns by Conversion", "Campaign Details"])
            
            with tabs[0]:
                # Show top 10 campaigns by lead count
                top_lead_campaigns = campaign_lead_data.sort_values('count', ascending=False).head(10)
                
                # Display as table first
                st.subheader("Campaigns by Lead Count")
                st.dataframe(top_lead_campaigns, hide_index=True)
                
                # Create bar chart
                fig = px.bar(
                    top_lead_campaigns,
                    x='campaign',
                    y='count',
                    text='count',
                    title="Top 10 Campaigns by Leads Generated"
                )
                
                fig.update_layout(
                    xaxis_title="Campaign",
                    yaxis_title="Number of Leads",
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                if campaign_conversion_data is not None:
                    # Show top 10 campaigns by conversion rate (minimum 5 leads)
                    top_conversion_campaigns = campaign_conversion_data[campaign_conversion_data['total'] >= 5].sort_values('conversion_rate', ascending=False).head(10)
                    
                    if not top_conversion_campaigns.empty:
                        # Display as table first
                        st.subheader("Campaigns by Conversion Rate")
                        # Display only campaign, completed, and total columns (no conversion_rate percentage)
                        display_df = top_conversion_campaigns[['campaign', 'completed', 'total']]
                        st.dataframe(display_df, hide_index=True)
                        
                        # Create bar chart
                        fig = px.bar(
                            top_conversion_campaigns,
                            x='campaign',
                            y='conversion_rate',
                            text='conversion_rate',
                            title="Top 10 Campaigns by Conversion Rate (minimum 5 leads)"
                        )
                        
                        fig.update_layout(
                            xaxis_title="Campaign",
                            yaxis_title="Conversion Rate (%)",
                            xaxis_tickangle=-45
                        )
                        
                        fig.update_traces(
                            texttemplate='%{text:.1f}%',
                            textposition='outside',
                            marker_color='green'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data to show conversion rates.")
                else:
                    st.info("Conversion data not available.")
            
            with tabs[2]:
                # Add dropdown to select campaigns
                campaign_list = list(campaign_stage_data.keys())
                
                # Create container with reduced height to ensure visibility
                campaign_details_container = st.container()
                
                with campaign_details_container:
                    # Add dropdown for campaign selection
                    selected_campaign = st.selectbox(
                        "Select Campaign",
                        options=campaign_list,
                        index=campaign_list.index("B2C/INFLUENCERCAMPAIGN/AIJAMAYROCK") if "B2C/INFLUENCERCAMPAIGN/AIJAMAYROCK" in campaign_list else 0
                    )
                    
                    # Create a smaller container for the table
                    table_container = st.container()
                    with table_container:
                        # Get the data for the selected campaign
                        stage_data = campaign_stage_data[selected_campaign]
                        
                        # Create table data
                        table_data = {
                            "Metric": [
                                "Total Leads",
                                "Personal Information",
                                "Academic Information & Extracurriculars",
                                "Essays",
                                "Book TETR Trial",
                                "Conversion Rate"
                            ],
                            "Value": [
                                stage_data['total'],
                                stage_data['PERSONAL INFORMATION'],
                                stage_data['ACADEMIC INFORMATION & EXTRACURRICULARS'],
                                stage_data['ESSAYS'],
                                stage_data['BOOK YOUR TETR TRIAL - APTITUDE TEST'],
                                f"{stage_data['conversion_rate']}%"
                            ]
                        }
                        
                        # Display as dataframe with custom height
                        st.dataframe(pd.DataFrame(table_data), hide_index=True, height=250)
        else:
            st.info("Campaign information not available in the uploaded data.")
        
        # Country & Region Analysis
        st.subheader("Country & Region Analysis")
        
        country_data, region_data = analyze_geography(df)
        
        if country_data is not None or region_data is not None:
            tabs = st.tabs(["Top Countries", "Country Distribution", "Region Distribution"])
            
            with tabs[0]:
                if country_data is not None:
                    # Display top countries table
                    st.dataframe(country_data.head(10), hide_index=True)
            
            with tabs[1]:
                if country_data is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display top 10 countries in a bar chart
                        top_countries = country_data.head(10)
                        
                        # Create bar chart
                        fig = px.bar(
                            top_countries,
                            x='country',
                            y='count',
                            text='count',
                            title="Top 10 Countries by Lead Count"
                        )
                        
                        fig.update_layout(
                            xaxis_title="Country",
                            yaxis_title="Number of Leads",
                            xaxis_tickangle=-45
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Create pie chart for country distribution
                        # Take top 5 countries and group others for better visualization
                        top_5_countries = country_data.head(5)
                        other_countries = pd.DataFrame([{
                            'country': 'Other Countries',
                            'count': country_data['count'][5:].sum() if len(country_data) > 5 else 0,
                            'percentage': country_data['percentage'][5:].sum() if len(country_data) > 5 else 0
                        }])
                        
                        pie_data = pd.concat([top_5_countries, other_countries])
                        
                        fig = px.pie(
                            pie_data, 
                            values='count', 
                            names='country', 
                            title='Lead Distribution by Country',
                            hole=0.4,  # Make it a donut chart for better appearance
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                        
                        fig.update_traces(
                            textposition='inside',
                            textinfo='percent+label'
                        )
                        
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                            margin=dict(l=20, r=20, t=30, b=0),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Country data not available.")
            
            with tabs[2]:
                if region_data is not None:
                    # Create bar chart for region distribution
                    top_regions = region_data.head(10)
                    
                    # Create bar chart
                    fig = px.bar(
                        top_regions,
                        x='region',
                        y='count',
                        text='count',
                        title="Top 10 Regions by Lead Count"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Region",
                        yaxis_title="Number of Leads",
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Region data not available.")
        else:
            st.info("Geographic information not available in the uploaded data.")
        
        # Income Data Analysis
        st.subheader("Income Data Analysis")
        
        income_analysis = analyze_income(df)
        
        if income_analysis is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Display income data as a table
                st.dataframe(income_analysis, hide_index=True)
            
            with col2:
                # Create pie chart for income distribution
                fig = px.pie(
                    income_analysis,
                    values='total_leads',
                    names='income_bracket',
                    title='Lead Distribution by Annual Household Income',
                    hole=0.4,  # Make it a donut chart for better appearance
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label'
                )
                
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                    margin=dict(l=20, r=20, t=30, b=0),
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Income information not available in the uploaded data.")
        st.subheader("Raw Data Viewer")
        show_raw_data = st.checkbox("Show Raw Data")
        
        if show_raw_data:
            st.dataframe(df)
            
            # Allow downloading filtered data
            st.markdown(get_csv_download_link(df, "filtered_leads.csv"), unsafe_allow_html=True)
else:
    # Display sample dashboard with instructions
    st.info("👆 Upload a CSV file to analyze lead data")
    st.write("This dashboard analyzes lead data from application forms. Upload a CSV file containing lead information to get started.")
    st.write("The dashboard works best with CSVs containing these columns:")
    
    # List of expected columns
    expected_columns = [
        "Registered Name", "Registered Email", "Registered Mobile", 
        "Primary Registration Campaign", "User Registration Date", 
        "Region", "Country", "Form stage", "Form Initiated",
        "Program Name", "Annual Household Income (USD)", "Payment Status"
    ]
    
    st.write(", ".join(expected_columns))
    
    st.write("But it will adapt to any CSV with similar lead data structure.")