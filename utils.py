def format_percentage(value):
    """
    Format a decimal as a percentage string
    """
    return f"{value * 100:.1f}%"

def calculate_survival_rate(df, column):
    """
    Calculate survival rate for different groups in a column
    """
    return df.groupby(column)['Survived'].mean()

def get_basic_stats(df, column):
    """
    Get basic statistical information about a column
    """
    return {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }

def clean_text_response(response):
    """
    Clean and format text responses
    """
    response = response.strip()
    response = response.replace('\n\n', '\n')
    return response
