import pandas as pd
import re

def load_and_clean_data(file_path):
    """
    Loads raw Twitter sentiment data, removes irrelevant/null/duplicate rows, 
    and normalizes URLs and mentions.
    """
    # 1. Load data
    df = pd.read_csv(file_path, header=None, names=["tweet_id", "entity", "sentiment", "tweet_content"])
    
    # 2. Filter and clean
    df = df[df['sentiment'] != 'Irrelevant']
    df = df.dropna(subset=['tweet_content'])
    df = df.drop_duplicates(subset=['tweet_content'])
    
    # 3. Normalize noise (URLs and Mentions)
    df['tweet_content'] = df['tweet_content'].str.replace(r'http\S+|www\S+', '<url>', regex=True, case=False)
    df['tweet_content'] = df['tweet_content'].str.replace(r'@\w+', '<mention>', regex=True)
    
    # 4. Return only needed columns
    return df[['tweet_content', 'sentiment']]

# Example usage for your validation set:
# val_df = load_and_clean_data("data/raw/twitter_validation.csv")
# val_df.to_csv("data/processed/val_cleaned.csv", index=False)