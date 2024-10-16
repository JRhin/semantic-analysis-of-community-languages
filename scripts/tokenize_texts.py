"""The script to tokenize the texts of each language.
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
    platform: str = "Voat"
    CURRENT: Path = Path('.')
    DATA_DIR: Path = CURRENT / 'data'
    PARQUET_PATH: Path = DATA_DIR / f"{platform}.parquet"
    
    # Variables
    column = "text"
    
    # Make spacy use the GPU if available
    spacy.prefer_gpu()
    
    # Get the languages
    df = pl.scan_parquet(PARQUET_PATH)
    languages = df.select('language').unique().sort('language').collect().get_column('language').to_list()

    print(f'Retrieving the texts from {PARQUET_PATH}...')
    corporas = dict()
    for language in tqdm(languages):
        # Get the corporas for each language
        corporas[language] = df.filter(pl.col('language') == language).select(column).drop_nulls().collect().get_column(column).to_list()

    print()
    print('Starting tokenizing the texts...')
    nlp = Tokenizer()
    for language in corporas:
        # Perform the texts tokenization
        tokenized_texts, adjectives = nlp.tokenize(corporas[language], language, batch_size=15000)

        # Define the path where to cache the results
        path: Path = DATA_DIR / f"{platform}"
        path.mkdir(exist_ok=True)

        # Save the resuls in parquet files
        pl.DataFrame({'Token': list(adjectives.keys()),
                      'Count': list(adjectives.values())}).write_parquet(path/f'{language.lower()}_adjectives.parquet')
        del adjectives
        pl.DataFrame({'Texts': tokenized_texts}).write_parquet(path/f'{language.lower()}_tokens.parquet')
        del tokenized_texts

    return None

if __name__ == "__main__":
    main()
