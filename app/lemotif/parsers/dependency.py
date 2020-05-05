"""
dependency.py
Extract dependencies from text input.
"""

import spacy

class Dependency():
    """
    Use spacy to extract dependencies from text input.
    """

    def __init__(self):
        # Load English model
        self.model = spacy.load('en')

    def extract_dependencies(self, text):
        """
        Return the relationship between subject nouns and modifier words in input text.

        :param text: Input text (str).
        :return: Subjects and their modifiers (dict).
        """
        doc = self.model(text)
        relationships = {}
        for token in doc:
            subject = [w for w in token.head.lefts if w.dep_ == "nsubj"]
            if subject and subject[0] != token and token.dep_ in ('acomp', 'ccomp', 'conj', 'amod'):
                subject = subject[0].lemma_.lower().strip() if subject[0].lemma_ != '-PRON-' else subject[0].lower_
                if subject in relationships.keys():
                    relationships[subject].append(token.lemma_.lower().strip())
                else:
                    relationships[subject] = [token.lemma_.lower().strip()]
        return relationships