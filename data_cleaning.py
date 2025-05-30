import kagglehub
import pandas as pd
from pathlib import Path
import nltk
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

nltk.download('stopwords')
# Set of common english stop words to compare the email text to
stop_words = set(stopwords.words('english'))

# Helper function to remove punctuation
# Removes anything that isn't a word, whitespace
# or '@' symbol
def remove_punctuation(text):
    return re.sub(r'[^\w\s@]', '', text)

# Helper function to remove stop words
def remove_stop_words(text):
    words = text.split()
    words_filtered = [word for word in words if word.lower() not in stop_words]
    return " ".join(words_filtered)

def clean_html(text):
   def clean_html(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    
    # Simple check if string contains HTML tags
    if not re.search(r'<[^>]+>', text):
        # No HTML tags found, return text as is
        return text.strip()
    
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

# Download latest version of multi source email dataset (Enron, Ling, CEAS, Nazario, Nigerian Fraud, and SpamAssassin)
path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")

email_data = pd.read_csv(Path(path).joinpath("phishing_email.csv"))

# Isolate the safe emails into their own dataframe
# The unsafe emails in this dataset will not be used as they
# do not distinguish between regular spam and phishing emails
safe_email_data = email_data[email_data['label'] == 0].copy()
unsafe_email_data = email_data[email_data['label'] == 1].copy()

# Obtain the phishing emails from the Nigerian Prince and Nazario 
# datasets specifically as those contain only emails confirmed to
# be phishing 
nigerian_emails = pd.read_csv(Path(path).joinpath("Nigerian_Fraud.csv"))
nazario_emails = pd.read_csv(Path(path).joinpath("Nazario.csv"))

# Combine the email body with the subject, sender, recipient, dates, 
# and subject
nigerian_emails["Email Text"] = nigerian_emails[["sender", "receiver", "date", "subject", "body"]].astype(str).agg(''.join, axis=1)
nazario_emails["Email Text"] = nazario_emails[["sender", "receiver", "date", "subject", "body"]].astype(str).agg(''.join, axis=1)

# Add label column to label all emails in this dataset as phishing (1)
nigerian_emails["label"] = 1
nazario_emails["label"] = 1 

# Drop columns that are no longer needed
nigerian_emails = nigerian_emails.drop(columns=["sender", "receiver", "date", "subject", "body", "urls"])
nazario_emails = nazario_emails.drop(columns=["sender", "receiver", "date", "subject", "body", "urls"])

# Remove html tags from the email contents
nigerian_emails["Email Text"] = nigerian_emails["Email Text"].apply(clean_html)
nazario_emails["Email Text"] = nazario_emails["Email Text"].apply(clean_html)

# Download latest version of phishing email specific dataset
path = kagglehub.dataset_download("subhajournal/phishingemails")

data = pd.read_csv(Path(path).joinpath("Phishing_Email.csv"))

# Isolate the phishing emails from this dataset to add to our
# legitimate emails from the first dataset since this dataset
# records phishing emails specifically.
phish_data = data[data["Email Type"] == "Phishing Email"].copy()

# Isolate the phishing emails from this dataset to add to our
# legitimate emails from the first dataset since this dataset
# records phishing emails specifically. Using phishing emails from
# 2 datasets to help make the ratio of legitimate to phishing emails
# more representative of what would be seen in a real setting.
df = pd.read_csv(Path(".").joinpath("Phishing_validation_emails.csv"))


# Verify all email text portions of the dataframes are in lowercase
safe_email_data["text_combined"] = safe_email_data["text_combined"].str.lower()
unsafe_email_data["text_combined"] = unsafe_email_data["text_combined"].str.lower()
phish_data["Email Text"] = phish_data["Email Text"].str.lower()
nigerian_emails["Email Text"] = nigerian_emails["Email Text"].str.lower()
nazario_emails["Email Text"] = nazario_emails["Email Text"].str.lower()

# Add label column to phish_data to label them as containing exclusively phishing emails
# 1 indicates phishing while 0 represents legitimate
phish_data["label"] = 1

# Drop columns that are no longer needed
phish_data = phish_data.drop(columns=["Unnamed: 0", "Email Type"])

# Create a cleaned dataset that combines the legitimate emails with those classified as phishing
new_column_names = {
    "text_combined": "Email Text"
}
# Rename the text_combined column to Email Text like the other two dataframes use
safe_email_data = safe_email_data.rename(columns=new_column_names)
unsafe_email_data = unsafe_email_data.rename(columns=new_column_names)

# Concatenate all the dataframes together
combo_data = pd.concat([safe_email_data, unsafe_email_data, nigerian_emails, nazario_emails, phish_data], ignore_index=True)

# Remove punctuation (if I decide to do sentiment analysis add this back or use the combo_data df)
clean_data = combo_data.copy()
clean_data["Email Text"] = clean_data["Email Text"].astype(str)
clean_data['cleaned text'] = clean_data["Email Text"].apply(remove_punctuation)
clean_data = clean_data.drop(columns=["Email Text"])

print(f"Total number of emails: {len(clean_data)}")

# Find duplicate rows based on a specific column, e.g., 'email_body'
duplicates_mask = clean_data.duplicated(subset=['cleaned text'], keep=False)

# Select all duplicate rows (including all copies, not just the second and beyond)
duplicate_records = df[duplicates_mask]

print(duplicate_records)

# Calculate the number of duplicates
duplicates = clean_data.duplicated(subset=['cleaned text']).sum()
print(f'Duplicate emails: {duplicates}')

# Drop any duplicate rows we may have introduced
# when combining these datasets
clean_data = clean_data.drop_duplicates()

safe_count = clean_data['label'].value_counts().get(0)
unsafe_count = clean_data['label'].value_counts().get(1)

print(f"Total number of emails after dropping duplicates: {len(clean_data)}")
print(f"Number of safe emails: {safe_count}")
print(f"Number of phishing emails: {unsafe_count}")

# Missing data count
missing_emails = clean_data['cleaned text'].isnull().sum()
print(f'Missing email bodies: {missing_emails}')

# Create another dataset with stop words removed, compare how the model performs with and without stop words
clean_data_no_stop = clean_data.copy()
clean_data_no_stop["cleaned text"] = clean_data_no_stop["cleaned text"].apply(remove_stop_words)

# Write cleaned data to csvs
clean_data.to_csv("clean_data.csv", index=False)
clean_data_no_stop.to_csv("clean_data_no_stop.csv", index=False)
combo_data.to_csv("clean_data_with_punc.csv", index=False)

# Email length in words
clean_data_no_stop['word_count'] = clean_data_no_stop['cleaned text'].apply(lambda x: len(str(x).split()))

print('Word count statistics:')
print(clean_data_no_stop.groupby('label')['word_count'].describe())

# Email length in characters
clean_data_no_stop['char_count'] = clean_data_no_stop['cleaned text'].apply(lambda x: len(str(x)))
print('Character count statistics:')
print(clean_data_no_stop.groupby('label')['char_count'].describe())