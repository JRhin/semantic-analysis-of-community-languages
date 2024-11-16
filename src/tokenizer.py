"""A special tokenizer based on SpaCy.
"""

import spacy
import numpy as np
import unicodedata
from tqdm.auto import tqdm
from collections import defaultdict

from spacy import Language
from spacy.tokens.doc import Doc


# ==============================================================================
#
#                           TOKENIZER SETTINGS
#
# ==============================================================================

# Function to check if a character is an emoji
def is_emoji(token):
    for char in token.text:
        if unicodedata.category(char).startswith("So"):
            return True
    return False

@Language.component("handling_negation")
def handling_negation(doc: Doc) -> Doc:
    """This components handles the negation of in a phrase by saving a negated version of the lemma verb.

    Args:
        doc : Doc
            The "Doc" that currently SpaCy is handling.

    Returns:
        doc : Doc
            The same "Doc" with the verb lemma modified.
    """
    for token in doc:
        if token.pos_ == "VERB" and any(child.dep_ == "neg" for child in token.children):
            token.lemma_ = f"not_{token.lemma_}"
    return doc


@Language.component("handling_propnames")
def handling_propnames(doc: Doc) -> Doc:
    """This components handles the presence of propnames in a phrase by saving a particular version of the lemma.

    Args:
        doc : Doc
            The "Doc" that currently SpaCy is handling.

    Returns:
        doc : Doc
            The same "Doc" with the modified lemma.
    """
    for token in doc:
        if token.pos_ == "PROPN":
            token.lemma_ = f"propn_{token.lemma_}"
    return doc


class Tokenizer():
    """A special tokenizer based on SpaCy.

    Args:
        model : str
            The SpaCy language model name. Default "en_core_web_sm".
        
    Attributes:
        self.nlp : Language
            A SpaCy model built used the passed model name.
    """
    def __init__(self,
                 model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

        # Add the custom processing components of the pipeline
        self.nlp.add_pipe("merge_entities")
        self.nlp.add_pipe("handling_negation", after="lemmatizer")
        self.nlp.add_pipe("handling_propnames", after="merge_entities")
        
    def tokenize(self,
                 corpora: list[str],
                 desc: str = None,
                 ids: list[str] = None, 
                 keep_tokens: list[str] = ["NOUN", "ADJ", "VERB", "PRON", "PROPN"],
                 n_process: int = 1,
                 batch_size: int = 1000) -> list[list[str]]:
        """It tokenizes the passed corpora and retrieve the adjectives.

        Args:
            corpora : list[str]
                A list of sentences.
            desc : str
                The description for tqdm. Default None.
            keep_tokens : list[str]
                Tokens to keep. Default ["NOUN", "ADJ", "VERB", "PRON"].
            n_process : int
                The number of process to use to run nlp.pipe in parallel. Default 1.
            batch_size : int
                The batch size. Default 1000.

        Returns:
            tokenized_texts : list[list[str]]
                The list of tokenized version of the texts in the corpora.

            adj : dict[str, int]
                The dictionary with the adjectives and their frequencies.
        """
        tokenized_texts = []
        final_ids = []
        adj = defaultdict(lambda: 0)
        for i, text in enumerate(tqdm(self.nlp.pipe(corpora, n_process=n_process, batch_size=batch_size), total=len(corpora), desc=desc)):
            tokens = []
            for token in text:
                # Check the nature of the tokens
                if token.pos_ in keep_tokens and not token.is_stop and not is_emoji(token) and not token.like_url and not token.is_space:

                    lemma = token.lemma_.lower().strip()

                    # Check if adj
                    if token.pos_ == "ADJ":
                        adj[lemma] += 1

                    # Save the lemmas
                    tokens.append(lemma)

            # Save the tokens
            if len(tokens) != 0:
                tokenized_texts.append(tokens)
                if ids: final_ids.append(ids[i])

        return tokenized_texts, dict(adj), final_ids



# ========================================================================
#
#                                MAIN LOOP
#
# ========================================================================

def main() -> None:
    """Some sanity checks.
    """
    from spacy.lang.en.examples import sentences

    print("Tryng some texts...")
    
    # Make spacy use the GPU if available
    spacy.prefer_gpu()
    
    nlp = Tokenizer()
    nlp.tokenize(sentences, ids=list(range(len(sentences))), n_process=1)

    print("Done.")


if __name__ == "__main__":
    main()
