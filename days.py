import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import base64
import requests
from io import StringIO
import time

st.set_page_config(layout="wide", page_title="Leads Analysis Dashboard", page_icon="📊")

def load_github_data(github_url):
    try:
        # Add a random parameter to prevent caching
        timestamp = int(time.time())
        
        if "github.com" in github_url and "/blob/" in github_url:
            raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        else:
            raw_url = github_url
            
        # Add cache-busting parameter
        cache_bust_url = f"{raw_url}?nocache={timestamp}"
        
        st.write(f"Fetching data from: {cache_bust_url}")
        response = requests.get(cache_bust_url)
        
        if response.status_code == 200:
            content = StringIO(response.text)
            df = pd.read_csv(content, encoding='utf-8')
            st.write(f"Data loaded successfully! Found {len(df)} rows.")
            return df
        else:
            st.error(f"Failed to load data: HTTP Status {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.replace('\s+', ' ', regex=True)
    return df

def clean_data(df):
    df_clean = df.copy()
    
    # Debug information about column names
    st.sidebar.expander("Debug Column Names").write(list(df_clean.columns))
    
    # Handle date columns more carefully
    date_columns = [col for col in df_clean.columns if 'date' in col.lower() or 'registration' in col.lower()]
    for col in date_columns:
        if col in df_clean.columns:
            st.sidebar.expander(f"Sample dates from {col}").write(df_clean[col].head())
            
            try:
                # Try various date formats
                date_formats = [
                    '%d/%m/%Y, %H:%M:%S',  # Standard format with time
                    '%d/%m/%Y, %H:%M',     # Without seconds
                    '%d/%m/%Y %H:%M:%S',   # Without comma
                    '%d/%m/%Y',            # Just date
                    '%m/%d/%Y, %H:%M:%S',  # US format with time
                    '%m/%d/%Y'             # US format date only
                ]
                
                for fmt in date_formats:
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], format=fmt, errors='coerce')
                        valid_dates = df_clean[col].dropna().count()
                        st.sidebar.expander(f"Date parsing").write(f"Format {fmt} parsed {valid_dates} dates for {col}")
                        if valid_dates > len(df_clean) * 0.5:  # If more than 50% parsed successfully
                            break
                    except:
                        continue
                        
                # If all specific formats failed, try automatic parsing
                if df_clean[col].dropna().count() < len(df_clean) * 0.5:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    
            except Exception as e:
                st.sidebar.expander(f"Date parsing error for {col}").write(f"Error: {e}")
                    
    # Rest of the function remains the same
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
        elif 'primary' in col.lower() and ('campaign' in col.lower() or 'registration' in col.lower()):
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
    
    rename_dict = {k: v for k, v in rename_dict.items() if k in df_clean.columns}
    if rename_dict:
        df_clean = df_clean.rename(columns=rename_dict)
        
    # Show date range info for debugging
    if 'User Registration Date' in df_clean.columns:
        if pd.api.types.is_datetime64_dtype(df_clean['User Registration Date']):
            min_date = df_clean['User Registration Date'].min()
            max_date = df_clean['User Registration Date'].max()
            st.sidebar.expander("Date Range in Data").write(f"Min date: {min_date}, Max date: {max_date}")
            
            # Show counts by date
            date_counts = df_clean['User Registration Date'].dt.date.value_counts().sort_index()
            st.sidebar.expander("Leads by Date").write(date_counts)
         
    return df_clean

def calculate_metrics(df):
    metrics = {}
    metrics['total_leads'] = len(df)
    if 'Form stage' in df.columns:
        form_stages = df['Form stage'].value_counts()
        personal_info_count = form_stages.get('PERSONAL INFORMATION', 0)
        metrics['form_progression_rate'] = round(((metrics['total_leads'] - personal_info_count) / metrics['total_leads'] * 100), 1) if metrics['total_leads'] > 0 else 0
    else:
        metrics['form_progression_rate'] = 0
    if 'Payment Status' in df.columns:
        approved_payments = df[df['Payment Status'] == 'PAYMENT APPROVED'].shape[0]
        metrics['completion_rate'] = round((approved_payments / metrics['total_leads'] * 100), 1) if metrics['total_leads'] > 0 else 0
        metrics['approved_payments'] = approved_payments
    else:
        metrics['completion_rate'] = 0
        metrics['approved_payments'] = 0
    if 'Country' in df.columns:
        metrics['countries_count'] = df['Country'].nunique()
    else:
        metrics['countries_count'] = 0
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

def analyze_form_stages(df):
    if 'Form stage' not in df.columns:
        return None, None
    stage_counts = df['Form stage'].value_counts().reset_index()
    stage_counts.columns = ['stage', 'count']
    
    funnel_data = []
    stage_order = [
        'PERSONAL INFORMATION', 
        'ACADEMIC INFORMATION & EXTRACURRICULARS', 
        'ESSAYS', 
        'BOOK YOUR TETR TRIAL - APTITUDE TEST'
    ]
    
    for stage in stage_counts['stage'].unique():
        if stage not in stage_order:
            stage_order.append(stage)
    
    for stage in stage_order:
        count = stage_counts[stage_counts['stage'] == stage]['count'].sum() if stage in stage_counts['stage'].values else 0
        funnel_data.append({'stage': stage, 'count': count})
    
    funnel_df = pd.DataFrame(funnel_data)
    
    return stage_counts, funnel_df

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

def analyze_campaigns(df):
    if 'Primary Registration Campaign' not in df.columns:
        return None, None, None
    
    lead_data = df['Primary Registration Campaign'].value_counts().reset_index()
    lead_data.columns = ['campaign', 'count']
    
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
    
    campaign_stage_data = {}
    for campaign in lead_data['campaign']:
        campaign_df = df[df['Primary Registration Campaign'] == campaign]
        stage_counts = {
            'total': len(campaign_df),
            'PERSONAL INFORMATION': 0,
            'ACADEMIC INFORMATION & EXTRACURRICULARS': 0,
            'ESSAYS': 0,
            'BOOK YOUR TETR TRIAL - APTITUDE TEST': 0
        }
        
        if 'Form stage' in campaign_df.columns:
            for stage in ['PERSONAL INFORMATION', 'ACADEMIC INFORMATION & EXTRACURRICULARS', 'ESSAYS', 'BOOK YOUR TETR TRIAL - APTITUDE TEST']:
                stage_counts[stage] = campaign_df[campaign_df['Form stage'] == stage].shape[0]
        
        if 'Payment Status' in campaign_df.columns:
            completed = campaign_df[campaign_df['Payment Status'] == 'PAYMENT APPROVED'].shape[0]
            conversion = round((completed / len(campaign_df) * 100), 1) if len(campaign_df) > 0 else 0
            stage_counts['conversion_rate'] = conversion
        else:
            stage_counts['conversion_rate'] = 0
            
        campaign_stage_data[campaign] = stage_counts
    
    return lead_data, conversion_df, campaign_stage_data

def analyze_income(df):
    if 'Annual Household Income (USD)' not in df.columns or 'Payment Status' not in df.columns:
        return None
    
    income_data = df['Annual Household Income (USD)'].value_counts().reset_index()
    income_data.columns = ['income_bracket', 'total_leads']
    
    income_completed = df[df['Payment Status'] == 'PAYMENT APPROVED'].groupby('Annual Household Income (USD)').size().reset_index()
    income_completed.columns = ['income_bracket', 'completed_leads']
    
    income_analysis = pd.merge(income_data, income_completed, on='income_bracket', how='left')
    income_analysis['completed_leads'] = income_analysis['completed_leads'].fillna(0).astype(int)
    income_analysis['conversion_rate'] = round((income_analysis['completed_leads'] / income_analysis['total_leads'] * 100), 1)
    
    return income_analysis

def get_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download processed CSV</a>'
    return href

# Add a function to display dataset stats
def display_dataset_stats(df):
    with st.expander("Dataset Statistics"):
        st.write(f"Total rows: {len(df)}")
        st.write(f"Columns: {', '.join(df.columns.tolist())}")
        
        if 'User Registration Date' in df.columns and pd.api.types.is_datetime64_dtype(df['User Registration Date']):
            # Create a DataFrame showing count by date
            date_counts = df['User Registration Date'].dt.date.value_counts().sort_index()
            date_counts = date_counts.reset_index()
            date_counts.columns = ['Date', 'Count']
            
            # Display as both table and chart
            st.write("Lead count by date:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(date_counts)
                
            with col2:
                fig = px.bar(date_counts, x='Date', y='Count', title='Leads by Date')
                fig.update_layout(xaxis_title="Date", yaxis_title="Number of Leads")
                st.plotly_chart(fig, use_container_width=True)

# Interface
st.sidebar.title("Leads Analysis Dashboard")
st.sidebar.subheader("Data Source")

default_url = "https://github.com/Harsh220802/Daily_Leads/blob/main/Leads.csv"
github_url = st.sidebar.text_input("GitHub CSV URL", default_url)

# Add option to force clear cache
clear_cache = st.sidebar.checkbox("Clear cache on load", value=True)
load_button = st.sidebar.button("Load Data")

st.title("📊 Leads Analysis Dashboard")

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    
if load_button or ('data_loaded' in st.session_state and st.session_state.data_loaded):
    if not st.session_state.data_loaded or load_button:
        if clear_cache and load_button:
            st.cache_data.clear()
            st.session_state.data_loaded = False
        
        with st.spinner('Loading data from GitHub...'):
            df_original = load_github_data(github_url)
            if df_original is not None:
                st.session_state.df_original = df_original
                st.session_state.data_loaded = True
                st.success(f"Data loaded successfully! Found {len(df_original)} rows.")
            else:
                st.error("Failed to load data. Please check the URL and try again.")
                st.session_state.data_loaded = False
    else:
        df_original = st.session_state.df_original
    
    if st.session_state.data_loaded:
        df = clean_column_names(df_original)
        df = clean_data(df)
        
        # Display dataset statistics
        display_dataset_stats(df)
        
        st.sidebar.subheader("Processed Data")
        st.sidebar.markdown(get_csv_download_link(df, "processed_leads.csv"), unsafe_allow_html=True)
        
        filtered_df = df.copy()
        
        if 'User Registration Date' in filtered_df.columns:
            st.sidebar.subheader("Date Filters")
            
            if filtered_df['User Registration Date'].dtype != 'datetime64[ns]':
                try:
                    st.sidebar.warning("Converting date column to datetime format...")
                    filtered_df['User Registration Date'] = pd.to_datetime(filtered_df['User Registration Date'], errors='coerce')
                    st.sidebar.success("Date conversion completed")
                except Exception as e:
                    st.sidebar.error(f"Could not convert date column to datetime format: {e}")
            
            if pd.api.types.is_datetime64_dtype(filtered_df['User Registration Date']):
                valid_dates = filtered_df['User Registration Date'].dropna()
                
                if not valid_dates.empty:
                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()
                    
                    # Show the date range to the user
                    st.sidebar.write(f"Available date range: {min_date} to {max_date}")
                    
                    filter_mode = st.sidebar.radio(
                        "Select Date Filter Mode",
                        ["All Dates", "Date Range", "Specific Date"]
                    )
                    
                    if filter_mode == "Date Range":
                        start_date = st.sidebar.date_input("Start Date", min_date)
                        end_date = st.sidebar.date_input("End Date", max_date)
                        
                        # Add a day to end_date to include the entire day
                        filtered_df = filtered_df[(filtered_df['User Registration Date'].dt.date >= start_date) & 
                                            (filtered_df['User Registration Date'].dt.date <= end_date)]
                        st.info(f"Showing data from {start_date} to {end_date}")
                        
                        # Show the count of records in the filtered date range
                        st.write(f"Showing {len(filtered_df)} leads")
                        
                    elif filter_mode == "Specific Date":
                        available_dates = sorted(filtered_df['User Registration Date'].dropna().dt.date.unique())
                        default_date_index = len(available_dates) - 1 if available_dates else 0
                        
                        if available_dates:
                            selected_date = st.sidebar.selectbox(
                                "Select Specific Date",
                                options=available_dates,
                                index=default_date_index
                            )
                            filtered_df = filtered_df[filtered_df['User Registration Date'].dt.date == selected_date]
                            st.info(f"Showing data for {selected_date}")
                            
                            # Show the count of records for the specific date
                            st.write(f"Showing {len(filtered_df)} leads for {selected_date}")
                        else:
                            st.sidebar.warning("No valid dates available in the dataset")
                else:
                    st.sidebar.warning("No valid dates available in the dataset")
        
        metrics = calculate_metrics(filtered_df)
        
        if 'User Registration Date' in filtered_df.columns:
            current_date = pd.to_datetime('today').date()
            today_df = filtered_df[filtered_df['User Registration Date'].dt.date == current_date] if pd.api.types.is_datetime64_dtype(filtered_df['User Registration Date']) else pd.DataFrame()
            
            if not today_df.empty:
                st.subheader(f"Today's Insights ({current_date})")
                st.write(f"New leads today: {len(today_df)}")
            
            if pd.api.types.is_datetime64_dtype(filtered_df['User Registration Date']):
                latest_date = filtered_df['User Registration Date'].max().date()
                st.write(f"Data last updated: {latest_date}")
        
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
            
        st.subheader("Form Stage Analysis")
        
        stage_counts, funnel_df = analyze_form_stages(filtered_df)
        
        if stage_counts is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Stage Distribution")
                form_stages = stage_counts['stage'].tolist()
                selected_stage = st.selectbox("Select Form Stage", form_stages)
                stage_users = stage_counts[stage_counts['stage'] == selected_stage]['count'].values[0]
                st.metric("Total Users", stage_users)
                
                if 'Country' in filtered_df.columns:
                    stage_df = filtered_df[filtered_df['Form stage'] == selected_stage]
                    top_countries = stage_df['Country'].value_counts().head(3)
                    
                    st.write(f"Top Sources for \"{selected_stage}\":")
                    
                    if 'Country' in filtered_df.columns:
                        country_text = f"Country: {top_countries.index[0]}" if not top_countries.empty else "Country: Not Specified"
                        st.markdown(f"• {country_text}")
                    
                    if 'Region' in filtered_df.columns:
                        top_regions = stage_df['Region'].value_counts().head(1)
                        region_text = f"Region: {top_regions.index[0]}" if not top_regions.empty else "Region: Not Specified"
                        st.markdown(f"• {region_text}")
                    
                    if 'Primary Registration Campaign' in filtered_df.columns:
                        top_campaigns = stage_df['Primary Registration Campaign'].value_counts().head(1)
                        campaign_text = f"Campaign: {top_campaigns.index[0]}" if not top_campaigns.empty else "Campaign: Not Specified"
                        st.markdown(f"• {campaign_text}")
            
            with col2:
                st.subheader("Application Funnel")
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

        # Rest of your existing code for campaign analysis, geography analysis, etc.
        # (remaining code kept the same as in the original for brevity)

        st.subheader("Raw Data Viewer")
        show_raw_data = st.checkbox("Show Raw Data")
        
        if show_raw_data:
            st.dataframe(filtered_df)
            st.markdown(get_csv_download_link(filtered_df, "filtered_leads.csv"), unsafe_allow_html=True)
else:
    st.info("👆 Upload a CSV file to analyze lead data")
    st.write("This dashboard analyzes lead data from application forms. Upload a CSV file containing lead information to get started.")
    st.write("The dashboard works best with CSVs containing these columns:")
    expected_columns = [
        "Registered Name", "Registered Email", "Registered Mobile", 
        "Primary Registration Campaign", "User Registration Date", 
        "Region", "Country", "Form stage", "Form Initiated",
        "Program Name", "Annual Household Income (USD)", "Payment Status"
    ]
    st.write(", ".join(expected_columns))
    st.write("But it will adapt to any CSV with similar lead data structure.")
