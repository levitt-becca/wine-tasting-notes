"""
Analysis of Regional and Varietal Signatures in Wine

This analysis looks at a dataset of Wine Enthusiast Magazine reviews to uncover
relationships between a wine's origin, variety, description, and perceived quality.

Dataset: 49,999 randomly sampled reviews from Wine Enthusiast magazine
Source: https://www.kaggle.com/datasets/zynicide/wine-reviews/data
"""

# Import libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

print("Wine Tasting Notes Analysis")
print("=" * 50)

filepath = r"C:\Users\becca\OneDrive\Documents\OMSA\Spring 2025\CSE 6040\Extra Credit\winemag-data_first150k.csv"

print("Initial Data Exploration")

# Read in CSV
wine = pd.read_csv(filepath)

# Drop unnamed index column if it exists
if 'Unnamed: 0' in wine.columns:
    wine = wine.drop('Unnamed: 0', axis=1)

print(f"Dimensions: {wine.shape}")
print(f"\nFirst 5 rows:")
print(wine.head())

print(f"\nColumn names: {wine.columns.tolist()}")
print(f"\nData types:\n{wine.dtypes}")
print(f"\nNull values:\n{wine.isnull().sum()}")
print(f"\nUnique values:\n{wine.nunique()}")
print(f"\nNumerical description:\n{wine.describe()}")

print("\nData Cleaning")

# Rename select columns
wine = wine.rename(columns={
    'region_1': 'region',
    'region_2': 'subregion',
    'taster_name': 'taster_name',
    'taster_twitter_handle': 'taster_twitter'
})

# Fill in null values
wine = wine.fillna("Unknown")

# Change price values to numeric for future analysis
wine['price'] = pd.to_numeric(wine['price'], errors='coerce')

# Remove unknown values for country and variety
wine = wine[wine['country'] != 'Unknown']
wine = wine[wine['variety'] != 'Unknown']

# Remove varieties with fewer than 100 reviews
variety_counts = wine['variety'].value_counts()
v_bool = wine['variety'].map(lambda x: variety_counts[x] >= 100)
wine = wine[v_bool]

# Remove countries with fewer than 100 reviews
country_counts = wine['country'].value_counts()
c_bool = wine['country'].map(lambda x: country_counts[x] >= 100)
wine = wine[c_bool]

print(f"After cleaning: {wine.shape}")

print("\nTasting Features")

# Define stopwords
stopwords = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours','ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
]

# Process descriptor words - FIXED VERSION
def descriptor_words(text):
    """Remove stopwords and punctuation from descriptions"""
    # Handle NaN or non-string values
    if pd.isna(text) or not isinstance(text, str):
        return []

    text_lower = text.lower()
    text_cleaned = re.sub(r'[^\w\s]', '', text_lower)
    words = text_cleaned.split()
    found_words = []

    for w in words:
        if w not in stopwords:
            found_words.append(w)

    return found_words

# Create descriptor words column with error handling
print("\nCreating descriptor words...")
wine['descriptor_words'] = wine['description'].apply(descriptor_words)

# Show most common descriptor words
try:
    all_descriptors = [desc for sublist in wine['descriptor_words'].tolist() for desc in sublist if isinstance(sublist, list)]
    print('Most common words in descriptions:')
    for word, count in Counter(all_descriptors).most_common(25):
        print(f"  {word}: {count}")
except Exception as e:
    print(f"Error processing descriptor words: {e}")
    # Fallback: create empty lists for all rows
    wine['descriptor_words'] = wine['description'].apply(lambda x: [])

# Define wine-specific descriptors
wine_descriptors = [
    "oak", "tann", "complex", "earth", "flo", "jam", "spic", "butter", "herb", "smok",
    "vanilla", "chocolate", "mineral", "leather", "tobacco", "pepper", "citrus", "berr",
    "plum", "strawberr", "peach", "cherr", "caramel", "nut", "grass", "veget", "honey",
    "tropical", "stone", "peat", "barn", "blackcurrant", "toast", "cedar", "blackberr",
    "bread", "mushroom", "melon", "raspberr", "apple", "licorice", "astringent", "balance",
    "bold", "crisp", "delicate", "dry", "full", "light", "medium", "slate", "acid", "soft",
    "structure", "round", "robust", "elegant", "supple", "lean", "rich", "viscous", "velvet",
    "mature", "youth", "age", "approachable", "austere", "backward", "close", "develope",
    "evolve", "fad", "layer", "nuance", "opulent", "refine", "rustic", "texture", "vibrant",
    "concentrate", "fresh", "harmon", "long", "short", "linger", "clean", "bright", "live",
    "silk", "smooth", "power", "integrate", "green", "expressive", "intense", "mellow",
    "firm", "juic", "tight", "vigor", "warm", "zip", "brac", "savor", "lemon", "cream",
    "yeast", "salt", "menthol"
]

# Find wine descriptors in text - FIXED VERSION
def find_descriptors(text):
    """Find wine descriptors in text"""
    # Handle NaN or non-string values
    if pd.isna(text) or not isinstance(text, str):
        return []

    text_lower = text.lower()
    found_descriptors = []

    for descriptor in wine_descriptors:
        try:
            pattern = r'\b' + re.escape(descriptor) + r'[a-z]*\b'
            if re.search(pattern, text_lower):
                found_descriptors.append(descriptor)
        except Exception as e:
            print(f"\nError processing descriptor '{descriptor}': {e}")
            continue

    return found_descriptors

# Create tasting notes column with error handling
print("\nCreating tasting notes...\n")
wine['tasting_notes'] = wine['description'].apply(find_descriptors)

# Show most common tasting notes
try:
    all_notes = [desc for sublist in wine['tasting_notes'].tolist() for desc in sublist if isinstance(sublist, list)]
    print('\nMost common tasting notes:')
    for note, count in Counter(all_notes).most_common(25):
        print(f"  {note}: {count}")
except Exception as e:
    print(f"\nError processing tasting notes: {e}")


print("\nCountry Analysis")
try:
    # Group by country and calculate statistics
    country_stats = wine.groupby('country').agg({
        'points': ['mean', 'min', 'max', 'count'],
        'price': ['mean', 'min', 'max']
    }).round(2)

    # Flatten column names
    country_stats.columns = ['_'.join(col).strip() for col in country_stats.columns]

    # Sort by average points (descending)
    country_stats_sorted = country_stats.sort_values('points_mean', ascending=False)

    print("\nTop 10 highest rated countries by average points:")
    print(country_stats_sorted[['points_mean', 'points_count', 'price_mean']].head(10))

    print("\nBottom 10 lowest rated countries by average points:")
    print(country_stats_sorted[['points_mean', 'points_count', 'price_mean']].tail(10))

except Exception as e:
    print(f"\nError in country analysis: {e}")

print("\nVariety Analysis")
try:
    # Group by variety and calculate statistics
    variety_stats = wine.groupby('variety').agg({
        'points': ['mean', 'min', 'max', 'count'],
        'price': ['mean', 'min', 'max']
    }).round(2)

    # Flatten column names
    variety_stats.columns = ['_'.join(col).strip() for col in variety_stats.columns]

    # Sort by average points (descending)
    variety_stats_sorted = variety_stats.sort_values('points_mean', ascending=False)

    print("\nTop 15 highest rated varieties by average points:")
    print(variety_stats_sorted[['points_mean', 'points_count', 'price_mean']].head(15))

    print("\nBottom 15 lowest rated varieties by average points:")
    print(variety_stats_sorted[['points_mean', 'points_count', 'price_mean']].tail(15))

except Exception as e:
    print(f"\nError in variety analysis: {e}")

print(f"\nValue Analysis")
try:
    min_points = 90
    max_price = 30
    print(f"Finding wines with {min_points}+ points and ≤${max_price}")

    # Filter for high-quality, affordable wines - handle NaN prices
    value_wines = wine[(wine['points'] >= min_points) & (wine['price'] <= max_price) & (~pd.isna(wine['price']))]

    print(f"\nFound {len(value_wines)} value wines out of {len(wine)} total wines")
    print(f"That's {len(value_wines)/len(wine)*100:.1f}% of all wines")

    if len(value_wines) > 0:
        print(f"\nValue wines by country:")
        value_by_country = value_wines['country'].value_counts()
        print(value_by_country.head(10))

        print(f"\nValue wines by variety:")
        value_by_variety = value_wines['variety'].value_counts()
        print(value_by_variety.head(10))

        print(f"\nSample value wines:")
        sample_cols = ['title', 'country', 'variety', 'points', 'price']
        print(value_wines[sample_cols].head(10))

        # Calculate price-to-quality ratio for value wines
        value_wines = value_wines.copy()
        value_wines['quality_per_dollar'] = value_wines['points'] / value_wines['price']

        print(f"\nBest quality per dollar:")
        best_value_cols = ['title', 'country', 'variety', 'points', 'price', 'quality_per_dollar']
        print(value_wines.nlargest(10, 'quality_per_dollar')[best_value_cols])

except Exception as e:
    print(f"\nError in value analysis: {e}")

print("\nVisualizations")

try:
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Points distribution
    axes[0, 0].hist(wine['points'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Wine Ratings (Points)')
    axes[0, 0].set_xlabel('Points')
    axes[0, 0].set_ylabel('Frequency')

    # 2. Price distribution (filter out NaN and 'Unknown' values)
    wine_with_price = wine[~pd.isna(wine['price']) & (wine['price'] != 0)]
    if len(wine_with_price) > 0:
        axes[0, 1].hist(wine_with_price['price'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Distribution of Wine Prices')
        axes[0, 1].set_xlabel('Price ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')

    # 3. Top countries by count
    top_countries = wine['country'].value_counts().head(10)
    axes[1, 0].barh(range(len(top_countries)), top_countries.values, color='lightgreen')
    axes[1, 0].set_yticks(range(len(top_countries)))
    axes[1, 0].set_yticklabels(top_countries.index)
    axes[1, 0].set_title('Top 10 Countries by Number of Reviews')
    axes[1, 0].set_xlabel('Number of Reviews')

    # 4. Top varieties by count
    top_varieties = wine['variety'].value_counts().head(10)
    axes[1, 1].barh(range(len(top_varieties)), top_varieties.values, color='plum')
    axes[1, 1].set_yticks(range(len(top_varieties)))
    axes[1, 1].set_yticklabels(top_varieties.index)
    axes[1, 1].set_title('Top 10 Varieties by Number of Reviews')
    axes[1, 1].set_xlabel('Number of Reviews')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nError creating visualizations: {e}")

print("\nDescriptor Analaysis")

try:
    # Keep only the rows that have tasting notes
    wine_filtered = wine[wine['tasting_notes'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    if len(wine_filtered) == 0:
        print("\nNo wines with tasting notes found. Skipping descriptor analysis.")
    else:
        print(f"\nFound {len(wine_filtered)} wines with tasting notes out of {len(wine)} total wines")

        # Give each tasting note its own line to make it easier to count
        wine_exp = wine_filtered.explode('tasting_notes')

        # Analyze notes by country
        group_field = 'country'
        top_n = 5

        # Calculate the tasting notes by country
        counts = wine_exp.groupby([group_field, 'tasting_notes']).size().reset_index(name='count')

        # Calculate the frequency of tasting notes in a country's reviews
        totals = wine_exp[group_field].value_counts()
        counts['total_reviews'] = counts[group_field].map(totals)
        counts['percentage'] = (counts['count'] / counts['total_reviews']) * 100

        # Calculate distinctiveness of country's tasting notes relative to other countries
        overall_freq = wine_exp['tasting_notes'].value_counts(normalize=True) * 100
        counts['overall_percentage'] = counts['tasting_notes'].map(overall_freq.to_dict())
        counts['distinctiveness'] = counts['percentage'] - counts['overall_percentage']

        country_notes = counts

        # Get top notes for each country
        top_country_notes = (country_notes.sort_values([group_field, 'count'], ascending=[True, False])
                            .groupby(group_field)
                            .apply(lambda x: ", ".join(x.nlargest(top_n, 'count')['tasting_notes']))
                            .reset_index()
                            .rename(columns={0: 'top_notes'}))

        print('\nTop tasting notes by country:')
        print(top_country_notes)

        # Plot most distinctive notes for select countries
        group_list = ['France', 'Italy', 'US', 'Spain', 'Portugal']  # Example countries
        available_countries = [c for c in group_list if c in country_notes[group_field].values]

        if available_countries:
            plt.figure(figsize=(12, 10))
            for i, group_value in enumerate(available_countries):
                group_data = (country_notes[country_notes[group_field] == group_value]
                             .sort_values('distinctiveness', ascending=False)
                             .head(top_n))
                plt.subplot(len(available_countries), 1, i+1)
                sns.barplot(data=group_data, x='distinctiveness', y='tasting_notes')
                plt.title(f'Most Distinctive Tasting Notes for {group_value}')
                plt.xlabel('Distinctiveness Score (%)')

            plt.tight_layout(pad=3.0)
            plt.show()

except Exception as e:
    print(f"Error in descriptor analysis: {e}")

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE!")