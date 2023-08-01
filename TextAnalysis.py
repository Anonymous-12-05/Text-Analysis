import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import openpyxl
# Download the 'punkt' package if not already downloaded
nltk.download('punkt')

# Function to extract article text from the URL
def extract_article_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        article_title = soup.title.text.strip()
        article_text = ' '.join([p.text.strip() for p in soup.find_all('p')])
        return article_title, article_text
    else:
        print(f"Failed to extract article from URL: {url}")
        return None, None

# Function to calculate the number of syllables in a word
def count_syllables(word):
    vowels = 'aeiouAEIOU'
    count = 0
    prev_char = None

    for char in word:
        if char in vowels and prev_char not in vowels:
            count += 1
        prev_char = char

    if word.endswith(('es', 'ed')):
        count -= 1

    return max(count, 1)

# Function to calculate the FOG Index
def calculate_fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

# Function to perform text analysis and compute variables
def perform_text_analysis(article_text, stopwords_folder, positive_words, negative_words):
    # Cleaning using Stop Words Lists
    stop_words = set()
    for file_name in os.listdir(stopwords_folder):
        with open(os.path.join(stopwords_folder, file_name), 'r', encoding='utf-8') as stopwords_file:
            stop_words.update(stopwords_file.read().splitlines())
             

    words = word_tokenize(article_text)
    cleaned_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # Creating a dictionary of Positive and Negative words
    positive_score = sum(1 for word in cleaned_words if word in positive_words)
    negative_score = sum(1 for word in cleaned_words if word in negative_words)

    # Polarity Score
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)

    # Subjectivity Score
    subjectivity_score = (positive_score + negative_score) / (len(cleaned_words) + 0.000001)

    # Analysis of Readability
    sentences = sent_tokenize(article_text)
    total_words = len(cleaned_words)
    total_sentences = len(sentences)
    avg_sentence_length = total_words / total_sentences

    complex_words = [word for word in cleaned_words if count_syllables(word) > 2]
    complex_word_count = len(complex_words)
    percentage_complex_words = complex_word_count / total_words

    fog_index = calculate_fog_index(avg_sentence_length, percentage_complex_words)

    # Average Number of Words Per Sentence
    avg_words_per_sentence = total_words / total_sentences

    # Syllable Count Per Word
    syllable_count_per_word = sum(count_syllables(word) for word in cleaned_words) / total_words

    # Personal Pronouns
    personal_pronouns_count = len(re.findall(r'\b(I|we|my|ours|us)\b', article_text, re.IGNORECASE))

    # Average Word Length
    avg_word_length = sum(len(word) for word in cleaned_words) / total_words

    return (
        positive_score,
        negative_score,
        polarity_score,
        subjectivity_score,
        avg_sentence_length,
        percentage_complex_words,
        fog_index,
        avg_words_per_sentence,
        complex_word_count,
        total_words,
        syllable_count_per_word,
        personal_pronouns_count,
        avg_word_length,
    )

# Main function to perform data extraction and analysis
def main():
    # Read input data from Excel
    input_file = "Input.xlsx"
    output_file = "Output Data Structure.xlsx"
    df_input = pd.read_excel(input_file)

    encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'cp1252']
    positive_words = None
    negative_words = None

    for encoding in encodings_to_try:
        try:
            with open("positive-words.txt", "r", encoding=encoding) as positive_file:
                positive_words = positive_file.read().splitlines()

            with open("negative-words.txt", "r", encoding=encoding) as negative_file:
                negative_words = negative_file.read().splitlines()

            # If successfully read, break out of the loop
            break
        except UnicodeDecodeError:
            continue

    if positive_words is None or negative_words is None:
        print("Failed to read the positive and/or negative words dictionaries.")
        return

    # Create an empty DataFrame to store the output
    columns = df_input.columns.tolist() + [
        "POSITIVE SCORE",
        "NEGATIVE SCORE",
        "POLARITY SCORE",
        "SUBJECTIVITY SCORE",
        "AVG SENTENCE LENGTH",
        "PERCENTAGE OF COMPLEX WORDS",
        "FOG INDEX",
        "AVG NUMBER OF WORDS PER SENTENCE",
        "COMPLEX WORD COUNT",
        "WORD COUNT",
        "SYLLABLE PER WORD",
        "PERSONAL PRONOUNS",
        "AVG WORD LENGTH",
    ]
    df_output = pd.DataFrame(columns=columns)

    # Process each row (article) in the input DataFrame
    for index, row in df_input.iterrows():
        url_id = row["URL_ID"]
        url = row["URL"]

        # Extract article text from the URL
        article_title, article_text = extract_article_text(url)

        if article_text:
            # Perform text analysis
            variables = [url_id] + perform_text_analysis(article_text, "StopWords", positive_words, negative_words)

            # Append the computed variables to the output DataFrame
            df_output = df_output.append(pd.Series(variables, index=df_output.columns), ignore_index=True)

            # Save the extracted article text to a file
            with open(f"{url_id}.txt", "w", encoding="utf-8") as file:
                file.write(f"{article_title}\n\n{article_text}")

    # Save the output DataFrame to an Excel file
    df_output.to_excel(output_file, index=False)

if __name__ == "__main__":
    main()
