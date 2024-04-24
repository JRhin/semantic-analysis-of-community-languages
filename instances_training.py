"""This python module train an istance of the original model for each topic.
"""

import polars as pl
from pathlib import Path
from tqdm.auto import tqdm
from gensim.models import Word2Vec
from multiprocessing import cpu_count


# ========================================================================
#
#                        FUNCTIONS DEFINITION
#
# ========================================================================

def define_original_model(path: Path,
                          min_count: int = 10,
                          window: int = 7,
                          alpha: float = 1e-1,
                          min_alpha: float = 1e-4,
                          negative: int = 20,
                          workers: int = 1,
                          vector_size: int = 100,
                          epochs: int = 300,
                          seed: int = 42) -> None:
    """Saves a Word2Vec model to a specific path.

    Args:
        path : Path
            The path where the model will be saved.
        min_count : int
            A wrapper to the min_count argument of the Gensim Word2Vec class. Default 10.    
        window : int
            A wrapper to the window argument of the Gensim Word2Vec class. Default 7.    
        alpha : float
            A wrapper to the alpha argument of the Gensim Word2Vec class. Default 1e-1.
        min_alpha : float
            A wrapper to the min_alpha argument of the Gensim Word2Vec class. Default 1e-4.
        negative : int
            A wrapper to the negative argument of the Gensim Word2Vec class. Default 1.
        workers : int
            A wrapper to the workers argument of the Gensim Word2Vec class. Default 1.
        vector_size : int
            A wrapper to the vector_size argument of the Gensim Word2Vec class. Default 100.
        epochs : int
            A wrapper to the epochs argument of the Gensim Word2Vec class. Default 300.
        seed : int
            A wrapper to the seed argument of the Gensim Word2Vec class. Default 42.

    Returns:
        None
    """
    original_model = Word2Vec(min_count=min_count,
                              window=window,
                              alpha=alpha,
                              min_alpha=min_alpha,
                              negative=negative,
                              workers=workers,
                              epochs=epochs,
                              vector_size=vector_size,
                              seed=seed)

    original_model.save(str(path))

    return None


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
    PARQUET_PATH: Path = DATA_DIR / "voat_labeled_data_unified.parquet"
    MODELS_PATH: Path = CURRENT / "models"

    MODELS_PATH.mkdir(exist_ok=True)

    # Variables
    min_len = 10

    # Get the topics
    topics = [topic.lower() for topic in pl.scan_parquet(PARQUET_PATH).select('topic').unique().sort('topic').collect().get_column('topic').to_list()]

    # Read the tokenized texts for each topic
    print("Read the tokenized texts for each topic...")
    df = pl.DataFrame()
    for topic in tqdm(topics):
        df = df.vstack(pl.read_parquet(DATA_DIR / f'tokenized texts/{topic}.parquet').with_columns(Topic=pl.lit(topic)))

    # Define and save the original model
    original_model = MODELS_PATH / "original_w2v.model"
    define_original_model(path = original_model,
                          workers=cpu_count())
    
    # Train an original model istance for each topic
    print()
    print("Training an original model instance for each topic...")
    w2v_models = dict()
    for topic in tqdm(topics):
        texts = df.filter((pl.col('Topic')==topic)&(pl.col('Texts').list.len()>= min_len)).get_column('Texts').to_list()

        # Load the original model configuration
        w2v_models[topic] = Word2Vec.load(str(original_model))

        # Create the vocabulary
        w2v_models[topic].build_vocab(texts)

        # Train the model over the topic corpora
        w2v_models[topic].train(texts, total_examples=w2v_models[topic].corpus_count, epochs=w2v_models[topic].epochs)

        w2v_models[topic].save(str(MODELS_PATH / f"{topic}_w2v.model"))

    return None

    
if __name__ == "__main__":
    main()
