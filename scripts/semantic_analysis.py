"""The script for computing the semantic analysis.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import polars as pl
from tqdm.auto import tqdm

from gensim.models import Word2Vec
from gensim.corpora import Dictionary

from src.mappings import relative_representation, semantic_differences


# ========================================================================
#
#                                MAIN LOOP
#
# ========================================================================

def main() -> None:
    """The main loop.
    """
    platform: str = "reddit"
    CURRENT: Path = Path('.')
    DATA_DIR: Path = CURRENT / "data"
    MODELS_DIR: Path = CURRENT / "models"
    PARQUET_PATH: Path = DATA_DIR / f"{platform}.parquet"

    # Variables
    prob: float  = 0.9
    adj_qt: float  = 0.1
    min_dfs: int = 150
  
    # Get the languages
    languages = [language.lower() for language in pl.scan_parquet(PARQUET_PATH).select('language').unique().sort('language').collect().get_column('language').to_list()]
   
    # Read the tokenized texts for each language
    print("Read the tokenized texts for each language...")
    tokenized_texts = pl.DataFrame()
    for language in tqdm(languages):
        tokenized_texts = tokenized_texts.vstack(pl.read_parquet(DATA_DIR / f'{platform}/tokens/{language}_tokens.parquet').with_columns(language=pl.lit(language)))

    print()
    print("Number of documents for each language:")
    print(tokenized_texts.group_by('language').len().sort('len'))

    # Create Gensim Dictionaries
    print()
    print("Create a Dictionary for each language...")
    dictionaries = dict()
    for language in tqdm(languages):
        dictionaries[language] = Dictionary(tokenized_texts.filter(pl.col('language')==language)['Texts'].to_list())        

    print()
    print("Some info:")
    for language in dictionaries:
        print(f'- {language}:')
        print("\tNumber of tokens:", len(dictionaries[language]))
        print('\tThe 5 most common words:', dictionaries[language].most_common(5))
                
    # Retrieve models
    print()
    print("Retrieving the models...")
    models = dict()
    for language in tqdm(languages):
        models[language] = Word2Vec.load(str(MODELS_DIR / f'{platform}/{platform}_{language}_w2v.model'))

    # Create vocabularies
    print()
    print("Creating vocabularies...")
    vocabularies = dict()
    for language in tqdm(languages):
        vocabularies[language] = list(models[language].wv.key_to_index.keys())

    common_vocab = set.intersection(*list(map(set, vocabularies.values())))

    # Returning the number of tokens for each language
    print()
    print(f"Number of common tokens between the vocabularies: {len(common_vocab)} tokens.")
    print("Number of tokens for each language (the ones that appeared in the corpora at least 'min_count' of the original model):")
    for language in vocabularies:
        print(f'\t{language}: {len(vocabularies[language])} tokens of which {round(len(common_vocab)/len(vocabularies[language])*100, 2)}% are in common.')
        
    # Creating a summary dataframe
    print()
    print("Creating a summary dataframe...")
    summary = pl.DataFrame()
    for language in languages:
        tokens = []
        dfs = []
        cfs = []
        embeddings = []
        
        for token in tqdm(vocabularies[language], desc=language):
            token_id = dictionaries[language].token2id[token]
            tokens.append(token)
            dfs.append(dictionaries[language].dfs[token_id])
            cfs.append(dictionaries[language].cfs[token_id])
            embeddings.append(models[language].wv[token])

        summary = summary.vstack(pl.DataFrame({
                                     'language': language,
                                     'Token': tokens,
                                     'Frequency in Documents': dfs,
                                     'Frequency': cfs,
                                     'Embedding': embeddings
                                 }).filter(pl.col("Frequency in Documents") >= min_dfs).with_row_index())
    
    # Update the vocabularies
    print()
    print("Update the vocabularies...")
    for language in tqdm(vocabularies):
        vocabularies[language] = summary.filter(pl.col('language')==language)['Token'].to_list()

    common_vocab = set.intersection(*list(map(set, vocabularies.values())))

    # Returning the number of tokens for each language, after romoving the ones with dfs less than min_dfs
    print()
    print(f"Number of common tokens between the vocabularies: {len(common_vocab)} tokens.")
    print(f"Number of tokens for each language (with dfs >= {min_dfs}):")
    for language in vocabularies:
        print(f'\t{language}: {len(vocabularies[language])} tokens of which {round(len(common_vocab)/len(vocabularies[language])*100, 2)}% are in common.')

    # Get the common adjectives as anchors
    print()
    print("Retrieving the common adjectives as anchors...")
    adjectives_df = pl.DataFrame()
    for language in tqdm(languages):
        adjectives_df = adjectives_df.vstack(pl.read_parquet(DATA_DIR / f'{platform}/adjectives/{language}_adjectives.parquet').with_columns(language=pl.lit(language)))

    adjectives_df = (
                        adjectives_df
                        .join(summary, on=['language', 'Token'])
                        .with_columns(Prob=pl.col('Count')/pl.col('Frequency'))
                        .filter(pl.col('Prob')>=prob)
                        .filter(pl.col("Frequency in Documents") >= pl.col("Frequency in Documents").quantile(adj_qt))
                        .group_by('Token').len()
                        .filter(pl.col('len') == len(languages))
                    )

    anchors = adjectives_df['Token'].to_list()
    
    print()
    print(f"Get the relative representation for {len(anchors)} anchors...")
    rel_repr = relative_representation(summary, languages, anchors)
    print(rel_repr)

    print()
    print("Get the semantic differences...")
    sem_diff_df = pl.DataFrame()
    sem_diff = semantic_differences(rel_repr, languages, anchors)
    for couple in sem_diff:
        sem_diff_df.vstack(sem_diff[couple].with_columns(Combination=pl.lit(couple)).select(['Combination', 'Token', 'Semantic Distance']),
                           in_place=True)
        print(couple)
        print(sem_diff[couple])
        print()

    print()
    print(sem_diff_df)

    sem_diff_df.write_parquet(f'{platform}_results.parquet')
        
    return None


if __name__ == "__main__":
    main()
