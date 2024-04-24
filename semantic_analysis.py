"""
"""

import polars as pl
from pathlib import Path
from tqdm.auto import tqdm

from gensim.models import Word2Vec
from gensim.corpora import Dictionary

from src.mappings import relative_representation


# ========================================================================
#
#                                MAIN LOOP
#
# ========================================================================

def main() -> None:
    """The main loop.
    """
    CURRENT: Path = Path('.')
    DATA_DIR: Path = CURRENT / "data"
    MODELS_DIR: Path = CURRENT / "models"
    PARQUET_PATH: Path = DATA_DIR / "voat_labeled_data_unified.parquet"

    # Variables
    min_dfs = 150
  
    # Get the topics
    topics = [topic.lower() for topic in pl.scan_parquet(PARQUET_PATH).select('topic').unique().sort('topic').collect().get_column('topic').to_list()]
   
    # Read the tokenized texts for each topic
    print("Read the tokenized texts for each topic...")
    tokenized_texts = pl.DataFrame()
    for topic in tqdm(topics):
        tokenized_texts = tokenized_texts.vstack(pl.read_parquet(DATA_DIR / f'tokenized texts/{topic}.parquet').with_columns(Topic=pl.lit(topic)))

    print()
    print("Number of documents for each topic:")
    print(tokenized_texts.group_by('Topic').len().sort('len'))

    # Create Gensim Dictionaries
    print()
    print("Create a Dictionary for each topic...")
    dictionaries = dict()
    for topic in tqdm(topics):
        dictionaries[topic] = Dictionary(tokenized_texts.filter(pl.col('Topic')==topic)['Texts'].to_list())        

    print()
    print("Some info:")
    for topic in dictionaries:
        print(f'- {topic}:')
        print("\tNumber of tokens:", len(dictionaries[topic]))
        print('\tThe 5 most common words:', dictionaries[topic].most_common(5))
                
    # Retrieve models
    print()
    print("Retrieving the models...")
    models = dict()
    for topic in tqdm(topics):
        models[topic] = Word2Vec.load(str(MODELS_DIR / f'{topic}_w2v.model'))

    # Create vocabularies
    print()
    print("Creating vocabularies...")
    vocabularies = dict()
    for topic in tqdm(topics):
        vocabularies[topic] = list(models[topic].wv.key_to_index.keys())

    common_vocab = set.intersection(*list(map(set, vocabularies.values())))

    # Returning the number of tokens for each topic
    print()
    print(f"Number of common tokens between the vocabularies: {len(common_vocab)} tokens.")
    print("Number of tokens for each topic (the ones that appeared in the corpora at least 'min_count' of the original model):")
    for topic in vocabularies:
        print(f'\t{topic}: {len(vocabularies[topic])} tokens of which {round(len(common_vocab)/len(vocabularies[topic])*100, 2)}% are in common.')
        
    # Creating a summary dataframe
    print()
    print("Creating a summary dataframe...")
    summary = pl.DataFrame()
    for topic in topics:
        tokens = []
        dfs = []
        cfs = []
        embeddings = []
        
        for token in tqdm(vocabularies[topic], desc=topic):
            token_id = dictionaries[topic].token2id[token]
            tokens.append(token)
            dfs.append(dictionaries[topic].dfs[token_id])
            cfs.append(dictionaries[topic].dfs[token_id])
            embeddings.append(models[topic].wv[token])

        summary = summary.vstack(pl.DataFrame({
                                     'Topic': topic,
                                     'Token': tokens,
                                     'Frequency in Documents': dfs,
                                     'Frequency': cfs,
                                     'Embeddings': embeddings
                                 }).filter(pl.col("Frequency in Documents") >= min_dfs).with_row_index())

    # Update the vocabularies
    print()
    print("Update the vocabularies...")
    for topic in tqdm(vocabularies):
        vocabularies[topic] = summary.filter(pl.col('Topic')==topic)['Token'].to_list()

    common_vocab = set.intersection(*list(map(set, vocabularies.values())))

    # Returning the number of tokens for each topic, after romoving the ones with dfs less than min_dfs
    print()
    print(f"Number of common tokens between the vocabularies: {len(common_vocab)} tokens.")
    print(f"Number of tokens for each topic (with dfs >= {min_dfs}):")
    for topic in vocabularies:
        print(f'\t{topic}: {len(vocabularies[topic])} tokens of which {round(len(common_vocab)/len(vocabularies[topic])*100, 2)}% are in common.')
   
    return None


if __name__ == "__main__":
    main()