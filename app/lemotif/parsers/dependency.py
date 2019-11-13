"""
dependency.py
Extract dependencies from text input.
"""

import string
import spacy

class Dependency():

    def __init__(self):
        self.model = spacy.load('en')
        self.lemotif = {
            'subjects': ['exercise', 'family', 'food', 'friends', 'god', 'health', 'love', 'recreation', 'school',
                         'sleep', 'work'],
            'emotions': ['afraid', 'angry', 'anxious', 'ashamed', 'awkward', 'bored', 'calm', 'confused',
                         'disgusted', 'excited', 'frustrated', 'happy', 'jealous', 'nostalgic', 'proud',
                         'sad', 'satisfied', 'surprised'],
        }
        self.lemotif_vectorized = self.model(' '.join(self.lemotif['subjects'] + self.lemotif['emotions']))

    def extract_dependencies(self, text):
        doc = self.model(text)
        relationships = {}
        for token in doc:
            subject = [w for w in token.head.lefts if w.dep_ == "nsubj"]
            print(token, subject, token.dep_)
            if subject and subject[0] != token and token.dep_ in ('acomp', 'ccomp', 'conj', 'amod'):
                subject = subject[0].lemma_.lower().strip() if subject[0].lemma_ != '-PRON-' else subject[0].lower_
                if subject in relationships.keys():
                    relationships[subject].append(token.lemma_.lower().strip())
                else:
                    relationships[subject] = [token.lemma_.lower().strip()]
        return relationships

if __name__ == '__main__':
    extractor = Dependency()
    print(extractor.extract_dependencies('Today was a terrible day. Work was stressful.'
                                         'School made me cry. I feel extremely sad and defeated.'))
    print(extractor.extract_dependencies('I had an amazing day with friends and family. We went swimming on'
                                         'a beautiful lake and I spent all afternoon in a state of bliss.'))