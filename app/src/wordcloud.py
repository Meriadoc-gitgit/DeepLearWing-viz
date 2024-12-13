import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pdfplumber

import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Get English stopwords
english_stopwords = set(stopwords.words('english'))


def read_pdf_pdfplumber(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


def remove_stop_words(corpus) : 
  for text, val in corpus.items() : 
    tmp = val.split(' ')
    # print(tmp)
    for stop_word in english_stopwords:
      if stop_word in tmp:
        # print(stop_word)
        tmp.remove(stop_word)
    tmp = " ".join(tmp)
    # print(tmp)
    corpus[text] = tmp
  return corpus


def extract_sections(text):
    # Regex to match headers
    header_pattern = r"(?<=\n)([A-Za-z0-9 .]+):(?!\S)|(?<=\n)(\d+ [A-Za-z0-9 ]+)"
    matches = list(re.finditer(header_pattern, text))

    sections = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = match.group().strip(': ')
        sections[header] = text[start:end].strip()
    
    return sections

@st.cache_data
def visualize(ax, label):
    
    wordcloud = WordCloud(width=500, height=400).generate(ax[label])
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(label)
