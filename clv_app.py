import streamlit as st
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='Year 2009-2010')
        st.write("Column names in the dataset:", df.columns.tolist())
        required_columns = ['Quantity', 'InvoiceDate', 'Customer ID', 'Invoice', 'Price']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Required column '{col}' is missing. Please check the dataset.")
                return None
        df.dropna(inplace=True)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['Price']
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to compute RFM
def compute_rfm(df):
    now = dt.datetime(2011, 12, 10)
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (now - x.max()).days,
        'Invoice': 'count',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalPrice': 'Monetary'})
    return rfm

# Load custom CSS
load_css("style.css")

# Load data
file_path = 'online_retail.xlsx'
df = load_data(file_path)

if df is not None:
    # Compute RFM
    rfm = compute_rfm(df)

    # Split the data
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    y = rfm['Monetary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Streamlit app
    st.title('Customer Lifetime Value Prediction')
    st.markdown("""
        <div style="background-color: green; padding: 10px; border-radius: 5px;">
            <h1 style="color: black; text-align: center;">CLV Prediction App</h1>
        </div>
    """, unsafe_allow_html=True)
    st.write('This app predicts the Customer Lifetime Value (CLV) based on Recency, Frequency, and Monetary value (RFM) metrics.')

    # Display data and metrics
    st.sidebar.header('Data Overview')
    st.sidebar.write(df.head())

    st.sidebar.header('RFM Table')
    st.sidebar.write(rfm.head())

    st.sidebar.header('Model Performance')
    st.sidebar.write(f'Mean Squared Error: {mse:.2f}')
    st.sidebar.write(f'Mean Absolute Error: {mae:.2f}')
    st.sidebar.write(f'R^2 Score: {r2:.2f}')

    # Visualizations
    st.header('Data Visualizations')
    st.subheader('Recency vs Monetary')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Recency', y='Monetary', data=rfm, ax=ax, color="#00796b")
    st.pyplot(fig)

    st.subheader('Frequency vs Monetary')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Frequency', y='Monetary', data=rfm, ax=ax, color="#00796b")
    st.pyplot(fig)

    # User input for new prediction
    st.header('Predict CLV for a New Customer')
    user_name = st.text_input('Name')
    recency = st.number_input('Recency (days)', min_value=0, max_value=1000, value=30)
    frequency = st.number_input('Frequency (number of purchases)', min_value=0, max_value=100, value=10)
    monetary_inr = st.number_input('Monetary (total spending in INR)', min_value=0.0, value=500.0)

    # Exchange rate for conversion
    exchange_rate = 74.0  # Example exchange rate from INR to original currency

    # Make prediction
    if st.button('Predict'):
        monetary = monetary_inr / exchange_rate
        new_data = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})
        prediction = model.predict(new_data)

        # Convert prediction to INR
        prediction_inr = prediction[0] * exchange_rate

        st.success(f'{user_name}, the predicted CLV is ${prediction[0]:.2f}')
        st.success(f'{user_name}, the predicted CLV in INR is ₹{prediction_inr:.2f}')
