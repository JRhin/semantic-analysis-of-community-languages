"""The script to tokenize the texts of each topic.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import spacy
import polars as pl
from tqdm.auto import tqdm

from src.tokenizer import Tokenizer


# ========================================================================
#
#                                MAIN LOOP
#
# ========================================================================

def main() -> None:
    """The main loop.
    """
    # Defining paths
    CURRENT: Path = Path('.')
    DATA_DIR: Path = CURRENT / 'data'
    PARQUET_PATH: Path = DATA_DIR / "voat_labeled_data_unified.parquet"
    
    # Variables
    MODEL: Path = CURRENT / "model/original_model.model"
    column = "text"
    
    # Make spacy use the GPU if available
    spacy.prefer_gpu()
    
    # Get the topics
    df = pl.scan_parquet(PARQUET_PATH)
    topics = df.select('topic').unique().sort('topic').collect().get_column('topic').to_list()

    print(f'Retrieving the texts from {PARQUET_PATH}...')
    corporas = dict()
    for topic in tqdm(topics):
        # Get the corporas for each topic
        corporas[topic] = df.filter(pl.col('topic') == topic).select(column).drop_nulls().collect().get_column(column).to_list()

    print()
    print('Starting tokenizing the texts...')
    nlp = Tokenizer()
    tokenized_texts = dict()
    for topic in corporas:
        # Perform the texts tokenization
        tokenized_texts[topic] = nlp.tokenize(corporas[topic], topic, batch_size=15000)

        # Define the path where to cache the results
        path: Path = DATA_DIR / "tokenized text"
        path.mkdir(exist_ok=True)

        # Save the tokenized texts as a parquet
        pl.DataFrame({'Texts': tokenized_texts[topic]}).write_parquet(path/f'{topic.lower()}.parquet')

    return None

if __name__ == "__main__":
    main()
