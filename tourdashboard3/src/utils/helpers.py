def load_data(file_path):
    """Load data from a specified file path."""
    import pandas as pd
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data for analysis."""
    # Example preprocessing steps
    data.dropna(inplace=True)
    return data

def format_output(data):
    """Format the output for display in the Streamlit app."""
    return data.to_dict(orient='records')

def calculate_statistics(data):
    """Calculate basic statistics for the dataset."""
    return {
        'mean': data.mean(),
        'median': data.median(),
        'std_dev': data.std()
    }