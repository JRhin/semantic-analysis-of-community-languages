"""This python module is used to train an istance of the original model for each language.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import polars as pl
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
    platform: str = "reddit"
    CURRENT: Path = Path('.')
    DATA_DIR: Path = CURRENT / "data"
    PARQUET_PATH: Path = DATA_DIR / f"{platform}.parquet"
    MODELS_PATH: Path = CURRENT / "models"

    MODELS_PATH.mkdir(exist_ok=True)

    # Variables
    min_len = 10

    # Get the languages
    languages = [language.lower() for language in pl.scan_parquet(PARQUET_PATH).select('language').unique().sort('language').collect().get_column('language').to_list()]

    # Read the tokenized texts for each language
    # print("Read the tokenized texts for each language...")
    # df = pl.DataFrame()
    # for language in tqdm(languages):
    #     df = df.vstack(pl.read_parquet(DATA_DIR / f'{platform}/{language}_tokens.parquet').with_columns(language=pl.lit(language)))

    # Define and save the original model
    original_model = MODELS_PATH / "original_w2v.model"
    define_original_model(path = original_model,
                          workers=cpu_count())
    
    # Train an original model istance for each language
    print()
    print("Training an original model instance for each language...")
    w2v_models = dict()
    for language in tqdm(languages):
        # Read the tokenized texts for each language
        # print(pl.read_parquet(DATA_DIR/f'{platform}/{language}_tokens.parquet'))
        texts = pl.scan_parquet(DATA_DIR/f'{platform}/{language}_tokens.parquet').select(pl.col("Texts")).filter(pl.col("Texts").list.len()>=min_len).collect().get_column("Texts").to_list()
        # df.filter((pl.col('language')==language)&(pl.col('texts').list.len()>= min_len)).get_column('Texts').to_list()

        # Load the original model configuration
        w2v_models[language] = Word2Vec.load(str(original_model))

        # Create the vocabulary
        w2v_models[language].build_vocab(texts)

        # Train the model over the language corpora
        w2v_models[language].train(texts, total_examples=w2v_models[language].corpus_count, epochs=w2v_models[language].epochs)

        # Save the model
        w2v_models[language].save(str(MODELS_PATH / f"{platform}_{language}_w2v.model"))

    return None

    
if __name__ == "__main__":
    main()
