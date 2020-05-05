"""
parsers.py
Gets closest match from Lemotif subjects and emotions.
"""
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


class SemanticSim():

    def __init__(self, model):
        self.model = spacy.load(model)
        self.parser = spacy.lang.en.English()
        self.lemotif = {
            'subjects': ['exercise', 'family', 'food', 'friends', 'god', 'health', 'love', 'recreation', 'school',
                         'sleep', 'work'],
            'emotions': ['afraid', 'angry', 'anxious', 'ashamed', 'awkward', 'bored', 'calm', 'confused',
                         'disgusted', 'excited', 'frustrated', 'happy', 'jealous', 'nostalgic', 'proud',
                         'sad', 'satisfied', 'surprised'],
        }
        self.lemotif_vectorized = self.model(' '.join(self.lemotif['subjects'] + self.lemotif['emotions']))

    def clean_text(self, text):
        """Lemmatize and omit punctuation and stopwords."""
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        punctuations = string.punctuation

        tokens = self.parser(text)
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
        tokens = [word for word in tokens if word not in stop_words and word not in punctuations]

        return list(set(tokens))

    def compare_word(self, text):
        """Return closest Lemotif label for each token."""
        tokens = self.model(text)
        best = {}
        for token in tokens:
            best_score, best_match = float('-inf'), None
            for concept in self.lemotif_vectorized:
                current_score = token.similarity(concept)
                if current_score > best_score:
                    best_score, best_match = current_score, concept
            best[token.text] = (best_match, best_score)

        return best

    def evaluate_text(self, text):
        """Calculate sum of similarities across all Lemotif labels."""
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.clean_text)
        frequencies = tfidf_vectorizer.fit_transform([text])
        features = tfidf_vectorizer.get_feature_names()
        evaluation = {}
        for idx, feature in enumerate(features):
            match, score = self.compare_word(feature)[feature]
            if match in evaluation.keys():
                evaluation[match] += score*frequencies.data[idx]
            else:
                evaluation[match] = score*frequencies.data[idx]

        return sorted(evaluation.items(), key=lambda x: x[1], reverse=True)