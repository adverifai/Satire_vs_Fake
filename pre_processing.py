import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup


def remove_url(text):
    """
    removing url
    :param text: input text
    :return: text without url
    """
    text = ' '.join(x for x in text.split() if x.startswith('http') == False and x.startswith('www') == False)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^www?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # extra step to make sure html tags are completely removed
    clean = re.compile('<.*>|<.*\"')
    result = re.sub(clean, '', text)
    return result


def stop_word_removal(text, stemming, punc_removal):
    """
    removing stops words
    :param text: input text
    :param stemming: if 1, apply stemming on input text, otherwise, don't
    :param punc_removal: if 0, remove punctuations, otherwise, don't
    """
    # the following line should be uncommented if the nltk data is not downloaded yet
    # nltk.download('stopwords')
    punc_list = list(string.punctuation)
    porter_stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for w in word_tokens:
        if w.lower() not in stop_words and ((punc_removal == 0 and w.lower() not in punc_list) or (punc_removal == 1)):
            if stemming == 1:
                filtered_sentence.append(porter_stemmer.stem(w.lower()))
            else:
                filtered_sentence.append(w.lower())
    result = ' '.join(filtered_sentence)
    return result


def text_clean(text, url_removal, tag_removal, stem_stop_punc, punc_removal):
    """
    cleaning a text
    :param text: input text of any length
    :param url_removal: flag for removing url from text
    :param tag_removal: flag for removing tags from text
    :param stem_stop_punc: flag for removing stop words, stemming tokens, and removing punctuations
    :param punc_removal: flag for removing punctuations
    :return: cleaned text
    """

    # removing urls from text
    if url_removal is True:
        text = remove_url(text)

    # removing HTML tags
    if tag_removal is True:
        text = BeautifulSoup(text, "lxml").text

    # stop word removal, stemming the tokens, and punctuations removal
    if stem_stop_punc is True:
        text = stop_word_removal(text, 1, punc_removal)

    # removing new line characters
    text = text.replace('\n', ' ')

    # filtering non-printable characters
    text = ''.join([i if ord(i) < 128 or i in ["'", "`"] else ' ' for i in text])

    # removing more than one space
    text = ' '.join(text.split())

    return text.strip()
