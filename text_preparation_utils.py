import re
from rapidfuzz import fuzz

def sanitize_text(text):
    """
    Sanitize a text string by removing non-readable characters, 
    excessive whitespace, and newline signs.

    Args:
        text (str): The input text to sanitize.

    Returns:
        str: The sanitized text.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Remove non-printable characters (e.g., control characters)
    sanitized = ''.join(char for char in text if char.isprintable())
    
    # Replace newlines and tabs with a single space
    sanitized = re.sub(r'[\r\n\t]', ' ', sanitized)
    
    # Remove excessive spaces
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Strip leading and trailing spaces
    sanitized = sanitized.strip()
    
    return sanitized
    
# Define a function to drop similar rows
def drop_similar_rows(df, column, threshold=90):
    """
    Remove rows from a pandas DataFrame that contain text in a specified column
    similar to the text in other rows, based on a similarity threshold.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        column (str): The column name containing the text to compare.
        threshold (int): The similarity threshold (0–100) to determine 
                         whether two rows are "similar". Higher values mean stricter similarity.

    Returns:
        pandas.DataFrame: A new DataFrame with rows containing similar text removed.
    """
    
    to_drop = set()
    for i, text1 in enumerate(df[column]):
        for j, text2 in enumerate(df[column]):
            if i != j and fuzz.ratio(text1, text2) > threshold:
                to_drop.add(j)
    return df.drop(to_drop)