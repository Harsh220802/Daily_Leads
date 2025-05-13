def clean_data(df):
    df_clean = df.copy()
    date_columns = [col for col in df_clean.columns if 'date' in col.lower()]
    for col in date_columns:
        try:
            # First try parsing timestamps with both date and time in flexible formats
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        except:
            # If the automatic parsing fails, we'll try specific formats
            try:
                # Try to parse dates with time component: 'dd/mm/yyyy, hh:mm:ss'
                df_clean[col] = pd.to_datetime(df_clean[col], format='%d/%m/%Y, %H:%M:%S', errors='coerce')
            except:
                try:
                    # Fallback to just date without time: 'dd/mm/yyyy'
                    df_clean[col] = pd.to_datetime(df_clean[col], format='%d/%m/%Y', errors='coerce')
                except:
                    # Final attempt with a common format
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], dayfirst=True, errors='coerce')
                    except:
                        pass
    
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
    rename_dict = {k: v for k, v in rename_dict.items() if k in df_clean.columns}
    if rename_dict:
        df_clean = df_clean.rename(columns=rename_dict)         
    return df_clean

def load_github_data(github_url):
    try:
        if "github.com" in github_url and "/blob/" in github_url:
            raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        else:
            raw_url = github_url         
        response = requests.get(raw_url)      
        if response.status_code == 200:
            content = StringIO(response.text)
            
            # The issue might be caused by incorrect CSV parsing
            # Try parsing with different options
            try:
                df = pd.read_csv(content, encoding='utf-8')
            except:
                # Reset the StringIO cursor to the beginning
                content.seek(0)
                try:
                    # Try with different delimiter detection
                    df = pd.read_csv(content, encoding='utf-8', delimiter=None, engine='python')
                except:
                    # Reset the StringIO cursor again
                    content.seek(0)
                    # Last attempt with very flexible parsing
                    df = pd.read_csv(content, encoding='utf-8', sep=None, engine='python')
            
            return df
        else:
            st.error(f"Failed to load data: HTTP Status {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
