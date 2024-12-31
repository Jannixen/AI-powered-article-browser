import re
from rapidfuzz import fuzz

def sanitize_text(text):
    text = re.sub('\n', ' ', text)
    return text
    
# Define a function to drop similar rows
def drop_similar_rows(df, column, threshold=90):
    to_drop = set()
    for i, text1 in enumerate(df[column]):
        for j, text2 in enumerate(df[column]):
            if i != j and fuzz.ratio(text1, text2) > threshold:
                to_drop.add(j)
    return df.drop(to_drop)