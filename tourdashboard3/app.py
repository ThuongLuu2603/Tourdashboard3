import streamlit as st
from src.data.loader import load_data
from src.utils.helpers import process_data

# Set the title of the app
st.title("Tour Dashboard")

# Load data
data = load_data()

# Process data
processed_data = process_data(data)

# Display data
st.write("Processed Data:")
st.dataframe(processed_data)

# Add more functionality as needed
st.sidebar.header("Settings")
option = st.sidebar.selectbox("Select an option", ["Option 1", "Option 2"])

if option == "Option 1":
    st.write("You selected Option 1")
elif option == "Option 2":
    st.write("You selected Option 2")