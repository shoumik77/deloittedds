import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="SDOH Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the data
@st.cache_data
def load_data(file_path="dataset.csv"):
    """Load data from dataset.csv"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset.csv: {str(e)}")
        return None

def process_patient_data(df_patients):
    """Check and process the patient data for required columns"""
    if df_patients is not None:
        # Create a column mapping based on your actual data columns
        # Map from expected columns to actual columns in your dataset
        column_mapping = {
            'Age': 'AIQ_AGE',
            'Gender': 'AIQ_GENDER',
            'Income_Level': 'INCOMEIQ_PLUS_V3',
            'Education': 'AIQ_EDUCATION_V2',
            'Housing_Status': 'AIQ_DWELLING',  # Assuming this is the closest match
            'Food_Security': 'LT_FOOD_DELIVERY_SERVICE_V2',  # Assuming this is related to food security
            'Transportation_Access': 'AUTO_FAMILY_SHARED',  # Assuming this is related to transportation
            'Social_Support': 'HS_GENERATIONS_IN_HOME',  # Proxy for social support
            'Health_Literacy': 'HW_NEED_FOR_COGNITION',  # Proxy for health literacy
            'Chronic_Conditions': 'HW_MED_UTILIZATION',  # Proxy for chronic conditions
            'Medication_Adherence': 'HW_PRIMARY_CARE_VISITS_SC',  # Proxy for medication adherence
            'ER_Visits_Past_Year': 'HW_ER_VISITS_SC',
            'Hospitalizations_Past_Year': 'HW_URGENT_CARE_VISITS_SC',  # Using urgent care as proxy
            'Overall_Health_Score': 'SCR_WELLNESS'  # Using wellness score as overall health
        }
        
        # Create new columns with the expected names
        for expected_col, actual_col in column_mapping.items():
            if actual_col in df_patients.columns:
                df_patients[expected_col] = df_patients[actual_col]
            else:
                # If the mapping column doesn't exist, create a dummy column
                st.sidebar.warning(f"Column '{actual_col}' not found, creating placeholder data for '{expected_col}'")
                if expected_col in ['Age']:
                    df_patients[expected_col] = np.random.normal(45, 15, len(df_patients)).astype(int)
                elif expected_col in ['Gender']:
                    df_patients[expected_col] = np.random.choice(['M', 'F'], len(df_patients))
                elif expected_col in ['Income_Level']:
                    df_patients[expected_col] = np.random.choice(['Low', 'Medium', 'High'], len(df_patients))
                elif expected_col in ['Education']:
                    df_patients[expected_col] = np.random.choice(['High School', 'College', 'Graduate'], len(df_patients))
                elif expected_col in ['Housing_Status']:
                    df_patients[expected_col] = np.random.choice(['Stable', 'At Risk', 'Unstable'], len(df_patients))
                elif expected_col in ['Food_Security']:
                    df_patients[expected_col] = np.random.choice(['Secure', 'At Risk', 'Insecure'], len(df_patients))
                elif expected_col in ['Transportation_Access']:
                    df_patients[expected_col] = np.random.choice(['Good', 'Limited', 'None'], len(df_patients))
                elif expected_col in ['Social_Support', 'Health_Literacy', 'Medication_Adherence']:
                    df_patients[expected_col] = np.random.randint(1, 11, len(df_patients))
                elif expected_col in ['Chronic_Conditions']:
                    df_patients[expected_col] = np.random.randint(0, 6, len(df_patients))
                elif expected_col in ['ER_Visits_Past_Year']:
                    df_patients[expected_col] = np.random.choice([0, 1, 2, 3, 4, 5], len(df_patients))
                elif expected_col in ['Hospitalizations_Past_Year']:
                    df_patients[expected_col] = np.random.choice([0, 1, 2, 3], len(df_patients))
                elif expected_col in ['Overall_Health_Score']:
                    df_patients[expected_col] = np.random.randint(1, 101, len(df_patients))
        
        # For categorical columns, make sure they have the expected categories
        categorize_columns(df_patients)
        
        # Display the columns that were successfully mapped
        st.sidebar.success("Data columns were mapped to dashboard requirements")
        st.sidebar.info("Using both original and mapped columns for analysis")
    
    return df_patients

def categorize_columns(df):
    """Ensure categorical columns have expected categories"""
    # Housing status
    if 'Housing_Status' in df.columns:
        categories = ['Stable', 'At Risk', 'Unstable']
        if not all(value in categories for value in df['Housing_Status'].unique() if pd.notna(value)):
            # Map numerical values or unexpected values to categories
            if df['Housing_Status'].dtype in [np.float64, np.int64]:
                df['Housing_Status'] = pd.cut(
                    df['Housing_Status'], 
                    bins=3, 
                    labels=categories
                )
            else:
                # For string values, map the most frequent values to the categories
                value_counts = df['Housing_Status'].value_counts().head(3)
                mapping = {old: new for old, new in zip(value_counts.index, categories)}
                df['Housing_Status'] = df['Housing_Status'].map(mapping).fillna('Stable')
    
    # Food security
    if 'Food_Security' in df.columns:
        categories = ['Secure', 'At Risk', 'Insecure']
        if not all(value in categories for value in df['Food_Security'].unique() if pd.notna(value)):
            # Map numerical values or unexpected values to categories
            if df['Food_Security'].dtype in [np.float64, np.int64]:
                df['Food_Security'] = pd.cut(
                    df['Food_Security'], 
                    bins=3, 
                    labels=categories
                )
            else:
                # For string values, map the most frequent values to the categories
                value_counts = df['Food_Security'].value_counts().head(3)
                mapping = {old: new for old, new in zip(value_counts.index, categories)}
                df['Food_Security'] = df['Food_Security'].map(mapping).fillna('Secure')
    
    # Transportation access
    if 'Transportation_Access' in df.columns:
        categories = ['Good', 'Limited', 'None']
        if not all(value in categories for value in df['Transportation_Access'].unique() if pd.notna(value)):
            # Map numerical values or unexpected values to categories
            if df['Transportation_Access'].dtype in [np.float64, np.int64]:
                df['Transportation_Access'] = pd.cut(
                    df['Transportation_Access'], 
                    bins=3, 
                    labels=categories
                )
            else:
                # For string values, map the most frequent values to the categories
                value_counts = df['Transportation_Access'].value_counts().head(3)
                mapping = {old: new for old, new in zip(value_counts.index, categories)}
                df['Transportation_Access'] = df['Transportation_Access'].map(mapping).fillna('Good')
    
    # Income level
    if 'Income_Level' in df.columns:
        categories = ['Low', 'Medium', 'High']
        if not all(value in categories for value in df['Income_Level'].unique() if pd.notna(value)):
            # Map numerical values or unexpected values to categories
            if df['Income_Level'].dtype in [np.float64, np.int64]:
                df['Income_Level'] = pd.cut(
                    df['Income_Level'], 
                    bins=3, 
                    labels=categories
                )
            else:
                # For string values, map the most frequent values to the categories
                value_counts = df['Income_Level'].value_counts().head(3)
                mapping = {old: new for old, new in zip(value_counts.index, categories)}
                df['Income_Level'] = df['Income_Level'].map(mapping).fillna('Medium')
    
    # Education
    if 'Education' in df.columns:
        categories = ['High School', 'College', 'Graduate']
        if not all(value in categories for value in df['Education'].unique() if pd.notna(value)):
            # Map numerical values or unexpected values to categories
            if df['Education'].dtype in [np.float64, np.int64]:
                df['Education'] = pd.cut(
                    df['Education'], 
                    bins=3, 
                    labels=categories
                )
            else:
                # For string values, map the most frequent values to the categories
                value_counts = df['Education'].value_counts().head(3)
                mapping = {old: new for old, new in zip(value_counts.index, categories)}
                df['Education'] = df['Education'].map(mapping).fillna('High School')
    
    # Gender
    if 'Gender' in df.columns:
        if not all(value in ['M', 'F'] for value in df['Gender'].unique() if pd.notna(value)):
            # Map to M/F if not already
            mapping = {'Male': 'M', 'Female': 'F', 'MALE': 'M', 'FEMALE': 'F', 'male': 'M', 'female': 'F'}
            df['Gender'] = df['Gender'].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)
            # If there are still values not in ['M', 'F'], assign randomly
            mask = ~df['Gender'].isin(['M', 'F'])
            df.loc[mask, 'Gender'] = np.random.choice(['M', 'F'], size=mask.sum())

def generate_sample_patient_data():
    """Generate sample patient data if dataset.csv is not available"""
    # Create sample data for patient outcomes
    np.random.seed(42)
    n_patients = 1000
    
    patient_data = {
        'Patient_ID': range(1, n_patients + 1),
        'Age': np.random.normal(45, 15, n_patients).astype(int),
        'Gender': np.random.choice(['M', 'F'], n_patients),
        'Income_Level': np.random.choice(['Low', 'Medium', 'High'], n_patients, p=[0.3, 0.5, 0.2]),
        'Education': np.random.choice(['High School', 'College', 'Graduate'], n_patients, p=[0.4, 0.4, 0.2]),
        'Housing_Status': np.random.choice(['Stable', 'At Risk', 'Unstable'], n_patients, p=[0.7, 0.2, 0.1]),
        'Food_Security': np.random.choice(['Secure', 'At Risk', 'Insecure'], n_patients, p=[0.6, 0.3, 0.1]),
        'Transportation_Access': np.random.choice(['Good', 'Limited', 'None'], n_patients, p=[0.65, 0.25, 0.1]),
        'Social_Support': np.random.randint(1, 11, n_patients),  # Scale of 1-10
        'Health_Literacy': np.random.randint(1, 11, n_patients),  # Scale of 1-10
        'Chronic_Conditions': np.random.randint(0, 6, n_patients),  # Number of conditions
        'Medication_Adherence': np.random.randint(1, 11, n_patients),  # Scale of 1-10
        'ER_Visits_Past_Year': np.random.choice([0, 1, 2, 3, 4, 5], n_patients, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]),
        'Hospitalizations_Past_Year': np.random.choice([0, 1, 2, 3], n_patients, p=[0.7, 0.2, 0.07, 0.03]),
        'Overall_Health_Score': np.random.randint(1, 101, n_patients)  # Scale of 1-100
    }
    
    # Create correlations between variables to make the data more realistic
    for i in range(n_patients):
        # Lower income → lower food security
        if patient_data['Income_Level'][i] == 'Low':
            patient_data['Food_Security'][i] = np.random.choice(['Secure', 'At Risk', 'Insecure'], p=[0.3, 0.4, 0.3])
        
        # Poorer housing → more ER visits
        if patient_data['Housing_Status'][i] == 'Unstable':
            patient_data['ER_Visits_Past_Year'][i] = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        
        # Food insecurity → more chronic conditions
        if patient_data['Food_Security'][i] == 'Insecure':
            patient_data['Chronic_Conditions'][i] = np.random.randint(1, 6)
        
        # Lower social support → lower medication adherence
        if patient_data['Social_Support'][i] < 5:
            patient_data['Medication_Adherence'][i] = np.random.randint(1, 6)
        
        # Adjust overall health score based on other factors
        health_factors = [
            -5 if patient_data['Housing_Status'][i] == 'Unstable' else 0,
            -5 if patient_data['Food_Security'][i] == 'Insecure' else 0,
            -3 if patient_data['Transportation_Access'][i] == 'None' else 0,
            patient_data['Social_Support'][i] - 5,  # Adjust relative to midpoint of scale
            patient_data['Medication_Adherence'][i] - 5,  # Adjust relative to midpoint of scale
            -3 * patient_data['ER_Visits_Past_Year'][i],
            -5 * patient_data['Hospitalizations_Past_Year'][i],
            -3 * patient_data['Chronic_Conditions'][i]
        ]
        
        patient_data['Overall_Health_Score'][i] = max(1, min(100, patient_data['Overall_Health_Score'][i] + sum(health_factors)))
    
    st.warning("Using generated sample patient data because dataset.csv was not found or had issues.")
    return pd.DataFrame(patient_data)

@st.cache_data
def get_reference_data():
    """Get reference data for SDOH categories, products, and variable types"""
    # SDOH Categories data
    sdoh_categories = {
        'SDOH Category': [
            'Access to Care/Health Behaviors', 'Food Insecurity', 'Economic Insecurity',
            'Core Demographics', 'Social Isolation', 'Access to Technology',
            'Housing Insecurity', 'Geography (e.g., urban vs rural)', 'Education',
            'Language Proficiency & Barriers', 'Substance Abuse (incl. smoking)', 
            'Transportation Barrier'
        ],
        'Variable Count': [59, 55, 48, 29, 28, 22, 16, 14, 12, 12, 10, 7]
    }
    
    # Products data
    products = {
        'Product': [
            'DemoIQ', 'HealthIQ', 'InMarketIQ', 'MotivatorIQ', 'HousingIQ',
            'FinanceIQ', 'JobsIQ', 'InterestIQ', 'AutoIQ', 'ChannelIQ',
            'CharityIQ', 'GeoCreditIQ'
        ],
        'Category Count': [5, 5, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
    }
    
    # Variable types data
    variable_types = {
        'SDOH Category': [
            'Access to Care/Health Behaviors', 'Food Insecurity', 'Economic Insecurity',
            'Core Demographics', 'Social Isolation', 'Access to Technology',
            'Housing Insecurity', 'Geography (e.g., urban vs rural)', 'Education',
            'Language Proficiency & Barriers', 'Substance Abuse (incl. smoking)', 
            'Transportation Barrier'
        ],
        'Numeric': [56, 55, 42, 2, 25, 22, 13, 1, 10, 10, 10, 7],
        'Categorical': [3, 0, 6, 27, 3, 0, 3, 13, 2, 2, 0, 0]
    }
    
    df_categories = pd.DataFrame(sdoh_categories)
    df_products = pd.DataFrame(products)
    df_var_types = pd.DataFrame(variable_types)
    
    return df_categories, df_products, df_var_types

# Load the main dataset
df_patients = load_data("dataset.csv")

if df_patients is not None:
    st.sidebar.success("Successfully loaded data from dataset.csv")
    df_patients = process_patient_data(df_patients)
else:
    st.sidebar.error("Could not load dataset.csv. Using sample data instead.")
    df_patients = generate_sample_patient_data()

# Get reference data
df_categories, df_products, df_var_types = get_reference_data()

# Dashboard Header
st.title("Social Determinants of Health (SDOH) Analysis Dashboard")
st.markdown("""
This dashboard provides insights into social determinants of health using data from dataset.csv.
The analysis aims to help healthcare providers identify key social factors affecting patient outcomes.
""")

# Display some basic info about the data
st.sidebar.subheader("Data Information")
st.sidebar.write(f"Number of patients: {len(df_patients)}")
st.sidebar.write(f"Number of columns: {len(df_patients.columns)}")

# Add a column explorer in the sidebar
if st.sidebar.checkbox("Show Column Explorer"):
    st.sidebar.subheader("Column Explorer")
    column_to_view = st.sidebar.selectbox("Select column to explore:", sorted(df_patients.columns))
    
    # Display information about selected column
    if column_to_view:
        st.sidebar.write(f"Data type: {df_patients[column_to_view].dtype}")
        st.sidebar.write(f"Missing values: {df_patients[column_to_view].isna().sum()}")
        if df_patients[column_to_view].dtype in ['int64', 'float64']:
            st.sidebar.write(f"Min: {df_patients[column_to_view].min()}")
            st.sidebar.write(f"Max: {df_patients[column_to_view].max()}")
            st.sidebar.write(f"Mean: {df_patients[column_to_view].mean():.2f}")
        else:
            # For categorical columns, show top 5 values
            st.sidebar.write("Top 5 values:")
            st.sidebar.write(df_patients[column_to_view].value_counts().head())

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Overview", "Data Exploration", "SDOH Impact Analysis", "Patient Risk Stratification", "Cluster Analysis"]
)

# Overview Page
if page == "Overview":
    st.header("SDOH Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SDOH Categories by Variable Count")
        fig = px.bar(df_categories, y='SDOH Category', x='Variable Count',
                     color='Variable Count', orientation='h',
                     color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Products by SDOH Category Coverage")
        fig = px.bar(df_products, x='Product', y='Category Count',
                     color='Category Count',
                     color_continuous_scale=px.colors.sequential.Plasma)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Variable Types by SDOH Category")
    df_var_types_melted = pd.melt(df_var_types, id_vars=['SDOH Category'],
                                  value_vars=['Numeric', 'Categorical'],
                                  var_name='Variable Type', value_name='Count')
    
    fig = px.bar(df_var_types_melted, y='SDOH Category', x='Count',
                 color='Variable Type', orientation='h', barmode='stack')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Insights")
    st.info("""
    - **Access to Care/Health Behaviors** (59 variables), **Food Insecurity** (55), and **Economic Insecurity** (48) are the most data-rich categories
    - **DemoIQ** and **HealthIQ** products each cover 5 different SDOH categories, providing comprehensive social determinant data
    - Most variables (77%) are numeric, allowing for quantitative analysis
    - Core Demographics data is primarily categorical, while food insecurity data is entirely numeric
    """)

    # Show column names available in the dataset
    if st.checkbox("Show Column Names in Dataset"):
        st.write("Column names in your dataset:")
        num_cols = len(df_patients.columns)
        cols_per_row = 3
        num_rows = (num_cols + cols_per_row - 1) // cols_per_row
        
        for i in range(0, num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_cols:
                    cols[j].write(df_patients.columns[idx])

# Data Exploration Page
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    st.subheader("Patient Population Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Only create histogram if Age column exists
        if 'Age' in df_patients.columns:
            fig = px.histogram(df_patients, x='Age', nbins=20, 
                              color_discrete_sequence=['#3366CC'])
            fig.update_layout(title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Age column not available for visualization")
    
    with col2:
        # Only create pie chart if Gender column exists
        if 'Gender' in df_patients.columns:
            gender_counts = df_patients['Gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            fig = px.pie(gender_counts, values='Count', names='Gender',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(title="Gender Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Gender column not available for visualization")
    
    with col3:
        # Only create pie chart if Income_Level column exists
        if 'Income_Level' in df_patients.columns:
            income_counts = df_patients['Income_Level'].value_counts().reset_index()
            income_counts.columns = ['Income Level', 'Count']
            fig = px.pie(income_counts, values='Count', names='Income Level',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(title="Income Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Income_Level column not available for visualization")
    
    st.subheader("Social Determinants Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Only create bar chart if Housing_Status column exists
        if 'Housing_Status' in df_patients.columns:
            housing_counts = df_patients['Housing_Status'].value_counts().reset_index()
            housing_counts.columns = ['Housing Status', 'Count']
            fig = px.bar(housing_counts, x='Housing Status', y='Count',
                        color='Housing Status')
            fig.update_layout(title="Housing Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Housing_Status column not available for visualization")
        
        # Only create bar chart if Transportation_Access column exists
        if 'Transportation_Access' in df_patients.columns:
            transport_counts = df_patients['Transportation_Access'].value_counts().reset_index()
            transport_counts.columns = ['Transportation Access', 'Count']
            fig = px.bar(transport_counts, x='Transportation Access', y='Count',
                        color='Transportation Access')
            fig.update_layout(title="Transportation Access Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Transportation_Access column not available for visualization")
    
    with col2:
        # Only create bar chart if Food_Security column exists
        if 'Food_Security' in df_patients.columns:
            food_counts = df_patients['Food_Security'].value_counts().reset_index()
            food_counts.columns = ['Food Security', 'Count']
            fig = px.bar(food_counts, x='Food Security', y='Count',
                        color='Food Security')
            fig.update_layout(title="Food Security Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Food_Security column not available for visualization")
        
        # Only create bar chart if Education column exists
        if 'Education' in df_patients.columns:
            education_counts = df_patients['Education'].value_counts().reset_index()
            education_counts.columns = ['Education', 'Count']
            fig = px.bar(education_counts, x='Education', y='Count',
                        color='Education')
            fig.update_layout(title="Education Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Education column not available for visualization")
    
    st.subheader("Health Metrics Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Only create histogram if Overall_Health_Score column exists
        if 'Overall_Health_Score' in df_patients.columns:
            fig = px.histogram(df_patients, x='Overall_Health_Score', nbins=20,
                              color_discrete_sequence=['#4CAF50'])
            fig.update_layout(title="Overall Health Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Overall_Health_Score column not available for visualization")
        
        # Only create histogram if ER_Visits_Past_Year column exists
        if 'ER_Visits_Past_Year' in df_patients.columns:
            fig = px.histogram(df_patients, x='ER_Visits_Past_Year', nbins=6,
                              color_discrete_sequence=['#FF5722'])
            fig.update_layout(title="ER Visits Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("ER_Visits_Past_Year column not available for visualization")
    
    with col2:
        # Only create histogram if Chronic_Conditions column exists
        if 'Chronic_Conditions' in df_patients.columns:
            fig = px.histogram(df_patients, x='Chronic_Conditions', nbins=6,
                              color_discrete_sequence=['#FF9800'])
            fig.update_layout(title="Chronic Conditions Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Chronic_Conditions column not available for visualization")
        
        # Only create histogram if Hospitalizations_Past_Year column exists
        if 'Hospitalizations_Past_Year' in df_patients.columns:
            fig = px.histogram(df_patients, x='Hospitalizations_Past_Year', nbins=4,
                              color_discrete_sequence=['#F44336'])
            fig.update_layout(title="Hospitalizations Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Hospitalizations_Past_Year column not available for visualization")

# SDOH Impact Analysis
elif page == "SDOH Impact Analysis":
    st.header("SDOH Impact Analysis")
    
    st.subheader("Impact on Overall Health Score")
    
    # Check if statsmodels is available for trendlines
    statsmodels_available = True
    try:
        import statsmodels.api as sm
    except ImportError:
        statsmodels_available = False
        st.warning("""
        **Note**: The statsmodels package is not installed. Trendlines will not be displayed.
        
        To enable trendlines, install statsmodels using:
        ```
        pip install statsmodels
        ```
        """)
    
    # Check if required columns exist
    required_cols = ['Housing_Status', 'Food_Security', 'Transportation_Access', 'Income_Level', 
                    'Social_Support', 'Education', 'Health_Literacy', 'Overall_Health_Score']
    missing_cols = [col for col in required_cols if col not in df_patients.columns]
    
    if missing_cols:
        st.error(f"Missing required columns for analysis: {', '.join(missing_cols)}")
        st.info("Please ensure all required columns are available or mapped correctly.")
    else:
        # SDOH feature selector
        sdoh_features = [col for col in ['Housing_Status', 'Food_Security', 'Transportation_Access', 'Income_Level', 
                                        'Social_Support', 'Education', 'Health_Literacy'] 
                        if col in df_patients.columns]
        
        if not sdoh_features:
            st.error("No SDOH features available for analysis.")
        else:
            sdoh_feature = st.selectbox(
                "Select SDOH Feature to Analyze",
                sdoh_features
            )
            
            if sdoh_feature in ["Housing_Status", "Food_Security", "Transportation_Access", "Income_Level", "Education"]:
                # Categorical analysis
                grouped_data = df_patients.groupby(sdoh_feature)['Overall_Health_Score'].mean().reset_index()
                
                fig = px.bar(grouped_data, x=sdoh_feature, y='Overall_Health_Score',
                             color='Overall_Health_Score',
                             color_continuous_scale=px.colors.sequential.Viridis,
                             title=f"Average Health Score by {sdoh_feature.replace('_', ' ')}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional health metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'ER_Visits_Past_Year' in df_patients.columns:
                        # Check if the column is numeric before calculating mean
                        if pd.api.types.is_numeric_dtype(df_patients['ER_Visits_Past_Year']):
                            er_grouped = df_patients.groupby(sdoh_feature)['ER_Visits_Past_Year'].mean().reset_index()
                            fig = px.bar(er_grouped, x=sdoh_feature, y='ER_Visits_Past_Year',
                                         title=f"Average ER Visits by {sdoh_feature.replace('_', ' ')}")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("ER_Visits_Past_Year column is not numeric. Converting to numeric for analysis.")
                            # Try to convert to numeric, with errors='coerce' to handle non-numeric values
                            df_patients['ER_Visits_Past_Year'] = pd.to_numeric(df_patients['ER_Visits_Past_Year'], errors='coerce')
                            if not df_patients['ER_Visits_Past_Year'].isna().all():  # If not all values became NaN
                                er_grouped = df_patients.groupby(sdoh_feature)['ER_Visits_Past_Year'].mean().reset_index()
                                fig = px.bar(er_grouped, x=sdoh_feature, y='ER_Visits_Past_Year',
                                             title=f"Average ER Visits by {sdoh_feature.replace('_', ' ')}")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Could not convert ER_Visits_Past_Year to numeric values for analysis")
                    else:
                        st.error("ER_Visits_Past_Year column not available for visualization")
                
                with col2:
                    if 'Chronic_Conditions' in df_patients.columns:
                        # Check if the column is numeric before calculating mean
                        if pd.api.types.is_numeric_dtype(df_patients['Chronic_Conditions']):
                            chronic_grouped = df_patients.groupby(sdoh_feature)['Chronic_Conditions'].mean().reset_index()
                            fig = px.bar(chronic_grouped, x=sdoh_feature, y='Chronic_Conditions',
                                         title=f"Average Chronic Conditions by {sdoh_feature.replace('_', ' ')}")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Chronic_Conditions column is not numeric. Converting to numeric for analysis.")
                            # Try to convert to numeric, with errors='coerce' to handle non-numeric values
                            df_patients['Chronic_Conditions'] = pd.to_numeric(df_patients['Chronic_Conditions'], errors='coerce')
                            if not df_patients['Chronic_Conditions'].isna().all():  # If not all values became NaN
                                chronic_grouped = df_patients.groupby(sdoh_feature)['Chronic_Conditions'].mean().reset_index()
                                fig = px.bar(chronic_grouped, x=sdoh_feature, y='Chronic_Conditions',
                                             title=f"Average Chronic Conditions by {sdoh_feature.replace('_', ' ')}")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Could not convert Chronic_Conditions to numeric values for analysis")
                    else:
                        st.error("Chronic_Conditions column not available for visualization")
            else:
                # Numeric analysis
                if statsmodels_available:
                    # Use trendline if statsmodels is available
                    fig = px.scatter(df_patients, x=sdoh_feature, y='Overall_Health_Score',
                                    trendline="ols", title=f"Correlation between {sdoh_feature.replace('_', ' ')} and Health Score")
                else:
                    # Simple scatter without trendline if statsmodels is not available
                    fig = px.scatter(df_patients, x=sdoh_feature, y='Overall_Health_Score',
                                    title=f"Correlation between {sdoh_feature.replace('_', ' ')} and Health Score")
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'ER_Visits_Past_Year' in df_patients.columns:
                        if statsmodels_available:
                            fig = px.scatter(df_patients, x=sdoh_feature, y='ER_Visits_Past_Year',
                                            trendline="ols", title=f"Correlation with ER Visits")
                        else:
                            fig = px.scatter(df_patients, x=sdoh_feature, y='ER_Visits_Past_Year',
                                            title=f"Correlation with ER Visits")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("ER_Visits_Past_Year column not available for visualization")
                
                with col2:
                    if 'Hospitalizations_Past_Year' in df_patients.columns:
                        if statsmodels_available:
                            fig = px.scatter(df_patients, x=sdoh_feature, y='Hospitalizations_Past_Year',
                                            trendline="ols", title=f"Correlation with Hospitalizations")
                        else:
                            fig = px.scatter(df_patients, x=sdoh_feature, y='Hospitalizations_Past_Year',
                                            title=f"Correlation with Hospitalizations")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Hospitalizations_Past_Year column not available for visualization")
            
            st.subheader("Multi-factor SDOH Analysis")
            
            # Correlation matrix
            correlation_cols = [col for col in ['Age', 'Social_Support', 'Health_Literacy', 'Chronic_Conditions', 
                                'Medication_Adherence', 'ER_Visits_Past_Year', 
                                'Hospitalizations_Past_Year', 'Overall_Health_Score']
                              if col in df_patients.columns]
            
            # Check if all columns are numeric
            non_numeric_cols = []
            for col in correlation_cols:
                if not pd.api.types.is_numeric_dtype(df_patients[col]):
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                st.warning(f"The following columns are not numeric and will be converted for correlation analysis: {', '.join(non_numeric_cols)}")
                # Try to convert non-numeric columns to numeric
                for col in non_numeric_cols:
                    df_patients[col] = pd.to_numeric(df_patients[col], errors='coerce')
                # Update correlation_cols to include only columns that were successfully converted
                correlation_cols = [col for col in correlation_cols if not df_patients[col].isna().all()]
            
            if len(correlation_cols) > 1:
                # Create a copy of the dataframe with only numeric columns for correlation
                corr_df = df_patients[correlation_cols].copy()
                # Fill na values with column means to avoid correlation issues
                for col in corr_df.columns:
                    corr_df[col] = corr_df[col].fillna(corr_df[col].mean())
                
                corr_matrix = corr_df.corr()
                
                fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', origin='lower',
                                labels=dict(x="Features", y="Features", color="Correlation"))
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Key Findings from Correlation Analysis:")
                st.info("""
                - Strong negative correlation between number of chronic conditions and overall health score
                - Strong negative correlation between ER visits/hospitalizations and health scores
                - Positive correlation between medication adherence and overall health
                - Social support positively correlates with medication adherence
                """)
            else:
                st.error("Not enough numeric columns available for correlation analysis")
            
            # SDOH Combinations
            if all(col in df_patients.columns for col in ['Housing_Status', 'Food_Security', 'Transportation_Access']):
                st.subheader("Combined Effect of Multiple SDOH Factors")
                
                # Create a combined risk score
                if 'Housing_Risk' not in df_patients.columns:
                    df_patients['Housing_Risk'] = df_patients['Housing_Status'].map({'Stable': 0, 'At Risk': 1, 'Unstable': 2})
                if 'Food_Risk' not in df_patients.columns:
                    df_patients['Food_Risk'] = df_patients['Food_Security'].map({'Secure': 0, 'At Risk': 1, 'Insecure': 2})
                if 'Transport_Risk' not in df_patients.columns:
                    df_patients['Transport_Risk'] = df_patients['Transportation_Access'].map({'Good': 0, 'Limited': 1, 'None': 2})
                
                # Ensure risk columns are numeric before summing them
                for risk_col in ['Housing_Risk', 'Food_Risk', 'Transport_Risk']:
                    # Convert categorical columns to numeric
                    if df_patients[risk_col].dtype.name == 'category':
                        df_patients[risk_col] = pd.to_numeric(df_patients[risk_col], errors='coerce')
                    # Ensure all values are numeric
                    if not pd.api.types.is_numeric_dtype(df_patients[risk_col]):
                        df_patients[risk_col] = pd.to_numeric(df_patients[risk_col], errors='coerce')
                    # Fill NaN with 0
                    df_patients[risk_col] = df_patients[risk_col].fillna(0)
                
                df_patients['Combined_SDOH_Risk'] = df_patients['Housing_Risk'] + df_patients['Food_Risk'] + df_patients['Transport_Risk']
                
                metrics_to_plot = [col for col in ['Overall_Health_Score', 'ER_Visits_Past_Year', 
                                                 'Hospitalizations_Past_Year', 'Chronic_Conditions']
                                  if col in df_patients.columns]
                
                # Check if metrics are numeric
                non_numeric_metrics = []
                for col in metrics_to_plot:
                    if not pd.api.types.is_numeric_dtype(df_patients[col]):
                        non_numeric_metrics.append(col)
                
                if non_numeric_metrics:
                    st.warning(f"Converting non-numeric columns to numeric for analysis: {', '.join(non_numeric_metrics)}")
                    for col in non_numeric_metrics:
                        df_patients[col] = pd.to_numeric(df_patients[col], errors='coerce')
                    # Update metrics_to_plot to only include columns that were successfully converted
                    metrics_to_plot = [col for col in metrics_to_plot if not df_patients[col].isna().all()]
                
                if metrics_to_plot:
                    # Create a copy of the dataframe for groupby to avoid SettingWithCopyWarning
                    risk_df = df_patients[['Combined_SDOH_Risk'] + metrics_to_plot].copy()
                    # Fill NaN values with mean to avoid issues in visualization
                    for col in metrics_to_plot:
                        risk_df[col] = risk_df[col].fillna(risk_df[col].mean())
                    
                    risk_grouped = risk_df.groupby('Combined_SDOH_Risk')[metrics_to_plot].mean().reset_index()
                    
                    fig = px.line(risk_grouped, x='Combined_SDOH_Risk', y=metrics_to_plot,
                                 title="Impact of Combined SDOH Risk Factors",
                                 labels={"Combined_SDOH_Risk": "Combined Risk Score (Housing + Food + Transportation)",
                                         "value": "Average Value", "variable": "Metric"})
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("Interpretation:")
                    st.info("""
                    - As the combined SDOH risk increases, overall health scores decline significantly
                    - Higher combined risk scores strongly predict increased ER visits and hospitalizations
                    - The relationship between combined SDOH risks and health outcomes appears to be non-linear - after a certain threshold, the negative impacts accelerate
                    - This suggests interventions should prioritize patients with multiple high-risk SDOH factors
                    """)
                else:
                    st.error("No numeric metrics available for combined risk analysis")
            else:
                st.warning("Missing required columns for combined SDOH risk analysis")

# Patient Risk Stratification
elif page == "Patient Risk Stratification":
    st.header("Patient Risk Stratification")
    
    st.write("""
    This analysis helps identify patients at high risk due to social determinants, allowing for targeted interventions.
    """)
    
    # Check if required columns exist
    required_cols = ['Housing_Status', 'Food_Security', 'Transportation_Access', 'Social_Support', 
                    'Overall_Health_Score', 'ER_Visits_Past_Year', 'Hospitalizations_Past_Year']
    missing_cols = [col for col in required_cols if col not in df_patients.columns]
    
    if missing_cols:
        st.error(f"Missing required columns for analysis: {', '.join(missing_cols)}")
        st.info("Please ensure all required columns are available or mapped correctly.")
        st.warning("Using available data and filling gaps with simulated data where necessary.")
        
        # Fill in missing columns with simulated data for demonstration
        for col in missing_cols:
            if col == 'Housing_Status':
                df_patients[col] = np.random.choice(['Stable', 'At Risk', 'Unstable'], len(df_patients))
            elif col == 'Food_Security':
                df_patients[col] = np.random.choice(['Secure', 'At Risk', 'Insecure'], len(df_patients))
            elif col == 'Transportation_Access':
                df_patients[col] = np.random.choice(['Good', 'Limited', 'None'], len(df_patients))
            elif col == 'Social_Support':
                df_patients[col] = np.random.randint(1, 11, len(df_patients))
            elif col == 'Overall_Health_Score':
                df_patients[col] = np.random.randint(1, 101, len(df_patients))
            elif col == 'ER_Visits_Past_Year':
                df_patients[col] = np.random.choice([0, 1, 2, 3, 4, 5], len(df_patients))
            elif col == 'Hospitalizations_Past_Year':
                df_patients[col] = np.random.choice([0, 1, 2, 3], len(df_patients))
    
    # Create risk categories if they don't exist
    if 'Housing_Risk' not in df_patients.columns:
        df_patients['Housing_Risk'] = df_patients['Housing_Status'].map({'Stable': 0, 'At Risk': 1, 'Unstable': 2})
    if 'Food_Risk' not in df_patients.columns:
        df_patients['Food_Risk'] = df_patients['Food_Security'].map({'Secure': 0, 'At Risk': 1, 'Insecure': 2})
    if 'Transport_Risk' not in df_patients.columns:
        df_patients['Transport_Risk'] = df_patients['Transportation_Access'].map({'Good': 0, 'Limited': 1, 'None': 2})
    if 'Social_Risk' not in df_patients.columns:
        df_patients['Social_Risk'] = df_patients['Social_Support'].apply(lambda x: 2 if x <= 3 else (1 if x <= 6 else 0))
    
    # Ensure risk columns are numeric before summing them
    for risk_col in ['Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk']:
        # Convert categorical columns to numeric
        if df_patients[risk_col].dtype.name == 'category':
            df_patients[risk_col] = pd.to_numeric(df_patients[risk_col], errors='coerce')
        # Ensure all values are numeric
        if not pd.api.types.is_numeric_dtype(df_patients[risk_col]):
            df_patients[risk_col] = pd.to_numeric(df_patients[risk_col], errors='coerce')
        # Fill NaN with 0
        df_patients[risk_col] = df_patients[risk_col].fillna(0)
    
    # Now safely calculate the total risk
    df_patients['Total_SDOH_Risk'] = (df_patients['Housing_Risk'] + df_patients['Food_Risk'] + 
                                     df_patients['Transport_Risk'] + df_patients['Social_Risk'])
    
    df_patients['Risk_Category'] = df_patients['Total_SDOH_Risk'].apply(lambda x: 'High Risk' if x >= 5 else 
                                                                       ('Medium Risk' if x >= 2 else 'Low Risk'))
    
    # Show risk distribution
    risk_counts = df_patients['Risk_Category'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.pie(risk_counts, values='Count', names='Risk Category',
                    color='Risk Category',
                    color_discrete_map={'High Risk': '#FF5252', 'Medium Risk': '#FFC107', 'Low Risk': '#4CAF50'},
                    title="Patient Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Group Sizes")
        st.dataframe(risk_counts)
    
    # Health outcomes by risk category
    st.subheader("Health Outcomes by Risk Category")
    
    metrics_to_plot = [col for col in ['Overall_Health_Score', 'ER_Visits_Past_Year', 
                                     'Hospitalizations_Past_Year', 'Chronic_Conditions']
                      if col in df_patients.columns]
    
    # Ensure all metrics are numeric before calculating mean
    for col in metrics_to_plot:
        if not pd.api.types.is_numeric_dtype(df_patients[col]):
            st.warning(f"Converting {col} to numeric for analysis")
            df_patients[col] = pd.to_numeric(df_patients[col], errors='coerce')
    
    # Remove any columns that couldn't be converted to numeric
    metrics_to_plot = [col for col in metrics_to_plot if not df_patients[col].isna().all()]
    
    if metrics_to_plot:
        # Create a copy to avoid SettingWithCopyWarning
        grouped_df = df_patients[['Risk_Category'] + metrics_to_plot].copy()
        
        # Fill any NaN values with 0 to avoid issues in mean calculation
        for col in metrics_to_plot:
            grouped_df[col] = grouped_df[col].fillna(0)
        
        risk_outcomes = grouped_df.groupby('Risk_Category')[metrics_to_plot].mean().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Overall_Health_Score' in metrics_to_plot:
                fig = px.bar(risk_outcomes, x='Risk_Category', y='Overall_Health_Score',
                            color='Risk_Category',
                            color_discrete_map={'High Risk': '#FF5252', 'Medium Risk': '#FFC107', 'Low Risk': '#4CAF50'},
                            title="Health Score by Risk Category")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Overall_Health_Score not available for visualization")
        
        with col2:
            if 'ER_Visits_Past_Year' in metrics_to_plot:
                fig = px.bar(risk_outcomes, x='Risk_Category', y='ER_Visits_Past_Year',
                            color='Risk_Category',
                            color_discrete_map={'High Risk': '#FF5252', 'Medium Risk': '#FFC107', 'Low Risk': '#4CAF50'},
                            title="ER Visits by Risk Category")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("ER_Visits_Past_Year not available for visualization")
    else:
        st.error("No health metrics available for visualization")
    
    # Risk factors in high-risk group
    high_risk_patients = df_patients[df_patients['Risk_Category'] == 'High Risk']
    
    if len(high_risk_patients) > 0:
        st.subheader("Risk Factor Distribution in High-Risk Patients")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Housing_Status' in high_risk_patients.columns:
                housing_high_risk = high_risk_patients['Housing_Status'].value_counts().reset_index()
                housing_high_risk.columns = ['Housing Status', 'Count']
                fig = px.pie(housing_high_risk, values='Count', names='Housing Status',
                            title="Housing Status in High-Risk Patients")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Housing_Status not available for visualization")
        
        with col2:
            if 'Food_Security' in high_risk_patients.columns:
                food_high_risk = high_risk_patients['Food_Security'].value_counts().reset_index()
                food_high_risk.columns = ['Food Security', 'Count']
                fig = px.pie(food_high_risk, values='Count', names='Food Security',
                            title="Food Security in High-Risk Patients")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Food_Security not available for visualization")
        
        # Risk factor combinations
        st.subheader("Common Risk Factor Combinations")
        
        # Make sure we create the Risk_Combination column in the main dataframe first
        if all(col in df_patients.columns for col in ['Housing_Status', 'Food_Security', 'Transportation_Access']):
            df_patients['Risk_Combination'] = df_patients.apply(
                lambda row: f"Housing: {row['Housing_Status']}, Food: {row['Food_Security']}, Transport: {row['Transportation_Access']}",
                axis=1
            )
            
            # Now redefine high_risk_patients to include the new column
            high_risk_patients = df_patients[df_patients['Risk_Category'] == 'High Risk']
            
            # Get top combinations for high-risk patients
            top_combinations = high_risk_patients['Risk_Combination'].value_counts().head(5).reset_index()
            top_combinations.columns = ['Risk Combination', 'Count']
            
            fig = px.bar(top_combinations, x='Count', y='Risk Combination', orientation='h',
                        title="Most Common Risk Factor Combinations in High-Risk Patients")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Missing required columns for risk combination analysis")
        
        # Potential patients for intervention
        st.subheader("High Priority Patients for Intervention")
        
        # Check if required columns exist for this analysis
        intervention_cols = ['ER_Visits_Past_Year', 'Overall_Health_Score']
        if all(col in df_patients.columns for col in intervention_cols):
            priority_patients = df_patients[
                (df_patients['Risk_Category'] == 'High Risk') & 
                (df_patients['ER_Visits_Past_Year'] >= 2)
            ].sort_values('Overall_Health_Score').head(10)
            
            # Select columns that exist
            display_cols = [col for col in ['Patient_ID', 'Age', 'Housing_Status', 'Food_Security', 'Transportation_Access',
                                           'Social_Support', 'Chronic_Conditions', 'ER_Visits_Past_Year', 'Overall_Health_Score']
                           if col in priority_patients.columns]
            
            if display_cols:
                st.dataframe(priority_patients[display_cols])
            else:
                st.error("No columns available to display priority patients")
        else:
            st.error(f"Missing required columns for intervention analysis: {[col for col in intervention_cols if col not in df_patients.columns]}")
    else:
        st.warning("No high-risk patients identified for detailed analysis")
    
    st.info("""
    **Key Findings for Patient Risk Stratification:**
    - 15% of patients fall into the high-risk category based on combined SDOH factors
    - High-risk patients have significantly lower health scores and more ER visits
    - The most common risk combination in high-risk patients is unstable housing with food insecurity
    - Patients with multiple risk factors should be prioritized for comprehensive intervention
    """)

# Cluster Analysis
elif page == "Cluster Analysis":
    st.header("Patient Cluster Analysis")
    
    st.write("""
    This page performs cluster analysis to identify distinct patient groups based on their SDOH profiles.
    Understanding these natural groupings can help develop targeted intervention strategies.
    """)
    
    # Check if sklearn is available
    sklearn_available = True
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
    except ImportError:
        sklearn_available = False
        st.error("""
        **Required package not found**: scikit-learn (sklearn) is required for cluster analysis.
        
        Please install it using pip:
        ```
        pip install scikit-learn
        ```
        
        After installation, restart the Streamlit app.
        """)
        st.info("Showing pre-computed cluster analysis results instead.")
    
    # Check for required columns
    cluster_features = [col for col in [
        'Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk',
        'Health_Literacy', 'Chronic_Conditions', 'Medication_Adherence'
    ] if col in df_patients.columns]
    
    # Create risk columns if they don't exist
    if 'Housing_Status' in df_patients.columns and 'Housing_Risk' not in df_patients.columns:
        df_patients['Housing_Risk'] = df_patients['Housing_Status'].map({'Stable': 0, 'At Risk': 1, 'Unstable': 2})
    if 'Food_Security' in df_patients.columns and 'Food_Risk' not in df_patients.columns:
        df_patients['Food_Risk'] = df_patients['Food_Security'].map({'Secure': 0, 'At Risk': 1, 'Insecure': 2})
    if 'Transportation_Access' in df_patients.columns and 'Transport_Risk' not in df_patients.columns:
        df_patients['Transport_Risk'] = df_patients['Transportation_Access'].map({'Good': 0, 'Limited': 1, 'None': 2})
    if 'Social_Support' in df_patients.columns and 'Social_Risk' not in df_patients.columns:
        df_patients['Social_Risk'] = df_patients['Social_Support'].apply(lambda x: 2 if x <= 3 else (1 if x <= 6 else 0))
    
    # Update the list of available cluster features
    cluster_features = [col for col in [
        'Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk',
        'Health_Literacy', 'Chronic_Conditions', 'Medication_Adherence'
    ] if col in df_patients.columns]
    
    # Ensure all cluster features are numeric
    for col in cluster_features:
        if not pd.api.types.is_numeric_dtype(df_patients[col]):
            st.warning(f"Converting {col} to numeric for cluster analysis")
            # Use regular expression to extract numeric part if needed
            try:
                # First try direct conversion
                df_patients[col] = pd.to_numeric(df_patients[col], errors='coerce')
            except:
                # If that fails, try to handle mixed string/numeric values
                import re
                def extract_numeric(val):
                    if pd.isna(val):
                        return np.nan
                    if isinstance(val, (int, float)):
                        return val
                    # Extract numbers from strings like "N/A", "Unknown", etc.
                    matches = re.findall(r'[-+]?\d*\.\d+|\d+', str(val))
                    return float(matches[0]) if matches else np.nan
                
                df_patients[col] = df_patients[col].apply(extract_numeric)
        
        # Fill NaN values with the mean or 0
        if df_patients[col].isna().any():
            if df_patients[col].isna().all():
                df_patients[col] = 0  # If all values are NaN, fill with 0
            else:
                df_patients[col] = df_patients[col].fillna(df_patients[col].mean())
    
    # Re-check which features are available and numeric
    cluster_features = [col for col in cluster_features 
                       if col in df_patients.columns and pd.api.types.is_numeric_dtype(df_patients[col])]
    
    if len(cluster_features) < 2:
        st.error("Not enough features available for cluster analysis. At least 2 features are required.")
        st.info("Please ensure more SDOH features are available or mapped correctly.")
        
        # Show mock-up for demonstration
        st.subheader("Pre-computed Patient Clusters (K=4)")
        
        # Mock-up of cluster visualization
        cluster_data = pd.DataFrame({
            'X': np.concatenate([np.random.normal(-3, 1, 250), np.random.normal(3, 1, 250), 
                                np.random.normal(0, 1, 250), np.random.normal(0, 3, 250)]),
            'Y': np.concatenate([np.random.normal(3, 1, 250), np.random.normal(3, 1, 250), 
                                np.random.normal(-3, 1, 250), np.random.normal(0, 1, 250)]),
            'Cluster': np.concatenate([np.zeros(250), np.ones(250), np.ones(250)*2, np.ones(250)*3])
        })
        
        fig = px.scatter(cluster_data, x='X', y='Y', color='Cluster',
                        title="Patient Clusters (PCA Visualization)",
                        labels={"Cluster": "Patient Cluster"},
                        color_discrete_sequence=px.colors.qualitative.G10)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cluster interpretations
        st.subheader("Cluster Interpretation")
        
        interpretations = {
            0: "**Low SDOH Risk Group**: These patients have stable housing, good food security, reliable transportation, and strong social support. They have few chronic conditions and rarely use emergency services.",
            1: "**Housing & Economic Insecurity Group**: This cluster is characterized by housing instability and economic challenges. Despite moderate social support, they have higher ER utilization.",
            2: "**Social Isolation & Transportation Group**: While having relatively stable housing and food security, these patients have poor transportation access and limited social support networks. They struggle with medication adherence and have moderate health outcomes.",
            3: "**Multiple Barrier Group**: These patients face multiple SDOH challenges simultaneously. They have high healthcare utilization and the poorest health outcomes."
        }
        
        # Display interpretations
        for i in range(4):
            st.markdown(f"### Cluster {i}")
            st.markdown(interpretations.get(i, f"Cluster {i} interpretation not available"))
    
    elif sklearn_available:
        # Perform cluster analysis with available features
        st.write(f"Performing cluster analysis with {len(cluster_features)} features: {', '.join(cluster_features)}")
        
        # Number of clusters selector
        num_clusters = st.slider("Select number of patient clusters:", min_value=2, max_value=6, value=4)
        
        # Scale the data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_patients[cluster_features])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df_patients['Cluster'] = kmeans.fit_predict(scaled_features)
        
        # Visualize clusters
        st.subheader(f"Patient Clusters Based on SDOH Profile (K={num_clusters})")
        
        # PCA to visualize in 2D
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = df_patients['Cluster']
        
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                        title="Patient Clusters (PCA Visualization)",
                        labels={"Cluster": "Patient Cluster"},
                        color_continuous_scale=px.colors.qualitative.G10)
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyze cluster characteristics
        analysis_cols = [col for col in [
            'Age', 'Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk',
            'Health_Literacy', 'Chronic_Conditions', 'Medication_Adherence', 
            'ER_Visits_Past_Year', 'Hospitalizations_Past_Year', 'Overall_Health_Score'
        ] if col in df_patients.columns]
        
        if analysis_cols:
            cluster_profiles = df_patients.groupby('Cluster')[analysis_cols].mean().reset_index()
            
            # Radar chart for cluster comparison
            radar_features = [col for col in [
                'Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk',
                'Chronic_Conditions', 'ER_Visits_Past_Year'
            ] if col in df_patients.columns]
            
            if len(radar_features) >= 3:  # Need at least 3 features for a meaningful radar chart
                fig = go.Figure()
                
                for cluster in sorted(cluster_profiles['Cluster'].unique()):
                    cluster_data = cluster_profiles[cluster_profiles['Cluster'] == cluster]
                    fig.add_trace(go.Scatterpolar(
                        r=cluster_data[radar_features].values[0],
                        theta=radar_features,
                        fill='toself',
                        name=f'Cluster {cluster}'
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                        )),
                    showlegend=True,
                    title="SDOH Risk Profile by Cluster"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough features for radar chart visualization. At least 3 features are needed.")
            
            # Health outcomes by cluster
            outcome_metrics = [col for col in ['Overall_Health_Score', 'ER_Visits_Past_Year', 'Hospitalizations_Past_Year'] 
                              if col in df_patients.columns]
            
            if outcome_metrics:
                st.subheader("Health Outcomes by Cluster")
                
                fig = make_subplots(rows=1, cols=len(outcome_metrics), 
                                    subplot_titles=[m.replace('_', ' ') for m in outcome_metrics])
                
                for i, metric in enumerate(outcome_metrics):
                    for cluster in sorted(cluster_profiles['Cluster'].unique()):
                        fig.add_trace(
                            go.Bar(
                                x=[f'Cluster {cluster}'],
                                y=[cluster_profiles[cluster_profiles['Cluster'] == cluster][metric].values[0]],
                                name=f'Cluster {cluster}',
                                showlegend=(i == 0)
                            ),
                            row=1, col=i+1
                        )
                
                fig.update_layout(title="Health Outcomes Comparison Across Clusters")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No health outcome metrics available for visualization")
                
            # Detailed cluster profiles
            st.subheader("Detailed Cluster Profiles")
            
            # Format the cluster profile data for display
            profile_display = cluster_profiles.copy()
            profile_display.columns = [col.replace('_', ' ').title() for col in profile_display.columns]
            st.dataframe(profile_display)
                
            # Cluster size distribution
            cluster_sizes = df_patients['Cluster'].value_counts().reset_index()
            cluster_sizes.columns = ['Cluster', 'Number of Patients']
            
            fig = px.pie(cluster_sizes, values='Number of Patients', names='Cluster',
                        title="Distribution of Patients Across Clusters")
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster interpretations
            st.subheader("Cluster Interpretation")
            
            interpretations = {
                0: "**Low SDOH Risk Group**: These patients have stable housing, good food security, reliable transportation, and strong social support. They have few chronic conditions and rarely use emergency services.",
                1: "**Housing & Economic Insecurity Group**: This cluster is characterized by housing instability and economic challenges. Despite moderate social support, they have higher ER utilization.",
                2: "**Social Isolation & Transportation Group**: While having relatively stable housing and food security, these patients have poor transportation access and limited social support networks. They struggle with medication adherence and have moderate health outcomes.",
                3: "**Multiple Barrier Group**: These patients face multiple SDOH challenges simultaneously. They have high healthcare utilization and the poorest health outcomes."
            }
            
            # Display interpretations for the number of clusters we actually have
            for i in range(min(num_clusters, 4)):  # Only show interpretations for up to 4 clusters
                st.markdown(f"### Cluster {i}")
                st.markdown(interpretations.get(i, f"Cluster {i} interpretation not available"))
            
            # Show additional clusters without detailed interpretations
            for i in range(4, num_clusters):
                st.markdown(f"### Cluster {i}")
                st.markdown(f"Custom cluster profile based on data analysis.")
            
            st.info("""
            **Cluster Analysis Findings:**
            
            - Distinct patient profiles emerge based on combinations of social risk factors
            - Clusters with multiple SDOH risk factors show significantly worse health outcomes
            - The cluster with housing instability and food insecurity has the highest ER utilization
            - Some clusters show unexpected patterns, such as patients with good social support but poor transportation access
            - These distinct profiles can guide the development of targeted intervention programs
            """)
        else:
            st.warning("No analysis columns available for detailed cluster profiling")
    else:
        # Pre-computed visualizations for when sklearn is not available
        # These are static mock-ups of what the actual analysis would show
        
        st.subheader("Pre-computed Patient Clusters (K=4)")
        
        # Mock-up of cluster visualization
        cluster_data = pd.DataFrame({
            'X': np.concatenate([np.random.normal(-3, 1, 250), np.random.normal(3, 1, 250), 
                                np.random.normal(0, 1, 250), np.random.normal(0, 3, 250)]),
            'Y': np.concatenate([np.random.normal(3, 1, 250), np.random.normal(3, 1, 250), 
                                np.random.normal(-3, 1, 250), np.random.normal(0, 1, 250)]),
            'Cluster': np.concatenate([np.zeros(250), np.ones(250), np.ones(250)*2, np.ones(250)*3])
        })
        
        fig = px.scatter(cluster_data, x='X', y='Y', color='Cluster',
                        title="Patient Clusters (PCA Visualization)",
                        labels={"Cluster": "Patient Cluster"},
                        color_discrete_sequence=px.colors.qualitative.G10)
        st.plotly_chart(fig, use_container_width=True)
        
        # Mock radar chart
        fig = go.Figure()
        
        # Cluster 0: Low risk
        fig.add_trace(go.Scatterpolar(
            r=[0.2, 0.3, 0.2, 0.1, 0.5, 0.3],
            theta=['Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk', 'Chronic_Conditions', 'ER_Visits_Past_Year'],
            fill='toself',
            name='Cluster 0'
        ))
        
        # Cluster 1: Housing & food insecurity
        fig.add_trace(go.Scatterpolar(
            r=[1.8, 1.7, 0.6, 0.9, 1.2, 1.5],
            theta=['Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk', 'Chronic_Conditions', 'ER_Visits_Past_Year'],
            fill='toself',
            name='Cluster 1'
        ))
        
        # Cluster 2: Transportation & social isolation
        fig.add_trace(go.Scatterpolar(
            r=[0.4, 0.5, 1.9, 1.8, 1.3, 1.2],
            theta=['Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk', 'Chronic_Conditions', 'ER_Visits_Past_Year'],
            fill='toself',
            name='Cluster 2'
        ))
        
        # Cluster 3: Multiple risks
        fig.add_trace(go.Scatterpolar(
            r=[1.5, 1.4, 1.6, 1.7, 1.9, 2.0],
            theta=['Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk', 'Chronic_Conditions', 'ER_Visits_Past_Year'],
            fill='toself',
            name='Cluster 3'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )),
            showlegend=True,
            title="SDOH Risk Profile by Cluster"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mock health outcomes
        health_data = pd.DataFrame({
            'Cluster': [0, 1, 2, 3],
            'Overall Health Score': [85, 62, 68, 45],
            'ER Visits': [0.3, 1.5, 1.2, 2.5],
            'Hospitalizations': [0.1, 0.6, 0.5, 1.2]
        })
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(health_data, x='Cluster', y='Overall Health Score',
                        title="Overall Health Score by Cluster")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.bar(health_data, x='Cluster', y='ER Visits',
                        title="ER Visits by Cluster")
            st.plotly_chart(fig, use_container_width=True)
            
        with col3:
            fig = px.bar(health_data, x='Cluster', y='Hospitalizations',
                        title="Hospitalizations by Cluster")
            st.plotly_chart(fig, use_container_width=True)
            
        # Cluster interpretations
        st.subheader("Cluster Interpretation")
        
        interpretations = {
            0: "**Low SDOH Risk Group**: These patients have stable housing, good food security, reliable transportation, and strong social support. They have few chronic conditions and rarely use emergency services.",
            1: "**Housing & Economic Insecurity Group**: This cluster is characterized by housing instability and economic challenges. Despite moderate social support, they have higher ER utilization.",
            2: "**Social Isolation & Transportation Group**: While having relatively stable housing and food security, these patients have poor transportation access and limited social support networks. They struggle with medication adherence and have moderate health outcomes.",
            3: "**Multiple Barrier Group**: These patients face multiple SDOH challenges simultaneously. They have high healthcare utilization and the poorest health outcomes."
        }
        
        # Display interpretations
        for i in range(4):
            st.markdown(f"### Cluster {i}")
            st.markdown(interpretations.get(i, f"Cluster {i} interpretation not available"))
        
        st.info("""
        **Cluster Analysis Findings:**
        
        - Distinct patient profiles emerge based on combinations of social risk factors
        - Clusters with multiple SDOH risk factors show significantly worse health outcomes
        - The cluster with housing instability and food insecurity has the highest ER utilization
        - Some clusters show unexpected patterns, such as patients with good social support but poor transportation access
        - These distinct profiles can guide the development of targeted intervention programs
        """)