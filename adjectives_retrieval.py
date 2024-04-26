"""The script to retrieve the adjectives for each topic.
"""

import spacy
import polars as pl
from pathlib import Path
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
    adjectives = dict()
    for topic in corporas:
        # Perform the adjective retrieval
        adjectives[topic] = nlp.adjectives(corporas[topic], topic, batch_size=15000)

        # Define the path where to cache the results
        path: Path = DATA_DIR / "adjectives"
        path.mkdir(exist_ok=True)

        # Save the adjectives as a parquet
        pl.DataFrame({'Token': list(adjectives[topic].keys()),
                      'Count': list(adjectives[topic].values())}).write_parquet(path/f'{topic.lower()}_adjectives.parquet')

    return None

if __name__ == "__main__":
    main()
