import pandas as pd
from fuzzywuzzy import fuzz
import re

# ''' 
# This program processes an input string to find the top 15 most relevant matches 
# from a dataset based on brand names, category tokens, and product similarity. 
# It uses fuzzy matching and token-based exact matching to filter and rank results.
# Input: A string entered by the user.
# Output: A CSV file containing the top 15 matches with relevant details and scores.
# '''

def load_data(file_path):
    """
    Load and preprocess the dataset.
    Ensures that the 'CategoryNameTokens' column is in a consistent string format.
    """
    data = pd.read_csv(file_path)
    data['CategoryNameTokens'] = data['CategoryNameTokens'].fillna('').astype(str)
    return data

def calculate_fuzzy_score(input_string, target_string):
    """
    Calculate a fuzzy match score between the input string and a target string.
    Uses `fuzz.partial_ratio` for substring-based similarity.
    """
    if not isinstance(target_string, str):
        return 0
    return fuzz.partial_ratio(input_string.lower(), target_string.lower())

def exact_word_match(word, target_string):
    """
    Check if a word exactly matches any token in the target string.
    Tokens are split by '.' to ensure consistency with 'CategoryNameTokens'.
    """
    if not isinstance(target_string, str):
        return False
    target_tokens = target_string.lower().split('.')
    return word in target_tokens

def find_best_matches(input_string, data, brands, tokens):
    """
    Find the best matches for the input string based on brand names, tokens, and product similarity.
    Filters and ranks data based on the identified brand, tokens, and fuzzy similarity scores.
    """
    input_string_lower = input_string.lower()
    words_in_input = re.findall(r'\w+', input_string_lower)  # Extract words from the input string

    # Identify brand by checking input words against brand list
    best_brand = None
    for word in words_in_input:
        if word in brands:
            best_brand = word
            break

    # Check tokens for matches
    matched_tokens = []
    for word in words_in_input:
        for token in tokens:
            if exact_word_match(word, token):
                matched_tokens.append(token)

    # Filter data by identified brand
    filtered_data = data.copy()
    if best_brand:
        brand_condition = (
            (data['BrandName'].str.lower().str.contains(best_brand, na=False)) |
            (data['BrandNameEn'].str.lower().str.contains(best_brand, na=False))
        )
        filtered_data = filtered_data[brand_condition]

    # Further filter by matched tokens
    if matched_tokens:
        token_condition = filtered_data['CategoryNameTokens'].apply(
            lambda x: any(token in x.split('.') for token in matched_tokens)
        )
        filtered_data = filtered_data[token_condition]

    # If no tokens match, fallback to model similarity search
    if filtered_data.empty:
        filtered_data = data.copy()

    # Calculate similarity scores for FaModel and Name
    filtered_data['Score'] = filtered_data.apply(
        lambda row: max(
            calculate_fuzzy_score(input_string_lower, row['FaName'].lower()) if isinstance(row['FaName'], str) else 0,
            calculate_fuzzy_score(input_string_lower, row['Name'].lower()) if isinstance(row['Name'], str) else 0,
            100 if exact_word_match(input_string_lower, row['FaName']) or exact_word_match(input_string_lower, row['Name']) else 0
        ),
        axis=1
    )

    # Add columns for matched tokens and brand
    filtered_data['IdentifiedBrand'] = best_brand if best_brand else ''
    filtered_data['MatchedTokens'] = ', '.join(matched_tokens) if matched_tokens else ''

    # Sort by score and return top matches
    filtered_data = filtered_data.sort_values(by='Score', ascending=False).head(15)
    return filtered_data

if __name__ == "__main__":

    """
    Main function to load data, process input, and find the best matches.
    Saves the top 15 matches to a CSV file.
    """
    # Load the dataset
    data_file = 'gsProductData_corrected_lowercase.csv'
    data = load_data(data_file)

    # Prepare unique brand names and tokens
    unique_brands = list(data['BrandName'].dropna().str.lower().unique()) + list(data['BrandNameEn'].dropna().str.lower().unique())
    unique_tokens = [token.strip() for sublist in data['CategoryNameTokens'].str.split('.').dropna() for token in sublist]

    # Accept user input
    input_string = input("Enter a string to find best matches (e.g., product name, brand, or category): ").strip()

    # Find best matches
    best_matches = find_best_matches(input_string, data, unique_brands, unique_tokens)

    # Extract required columns
    required_columns = ['ProductCode', 'Name', 'FaName', 'BrandName', 'BrandNameEn', 'BrandId', 'CategoryName', 'CategoryId', 'Score', 'IdentifiedBrand', 'MatchedTokens']
    best_matches = best_matches[[col for col in required_columns if col in best_matches.columns]]

    # Save the results
    output_file_path = 'top_15_matches_with_debug_data.csv'
    best_matches.to_csv(output_file_path, index=False, encoding='utf-8-sig')

    print(f"Top 15 most related records for '{input_string}' saved to: {output_file_path}")

# Run the main function

