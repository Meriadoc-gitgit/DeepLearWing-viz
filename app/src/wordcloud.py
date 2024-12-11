import re
import io
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def shape_wordcloud(df, theme):
    word_freq = defaultdict(int)
    for entry in df[theme]:
        pairs = re.findall(r'\[([^,]+), (\d+)\]', entry)    # Extraire les paires mot-fréquence
        for word, freq in pairs:
            word_freq[word] += int(freq)

    wordcloud = WordCloud(width=600, height=600, background_color='white').generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.imshow(wordcloud, interpolation='bilinear')
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    plt.axis('off')

    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    plt.close()

    return image_stream