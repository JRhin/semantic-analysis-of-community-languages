"""In this python module there are the core functions to retrieve the relative representation of a latent/semantic space.
"""

import polars as pl
from tqdm.auto import tqdm
from itertools import combinations

import numpy as np
from numpy.linalg import norm


# ========================================================================
#
#                        FUNCTIONS DEFINITION
#
# ========================================================================

def similarity(a: np.ndarray,
               b: np.ndarray) -> float:
    """A similarity/distance function.

    Args:
        a : np.ndarray
            First array.
        b : np.ndarray
            Second array. 

    Returns:
        float
            The output of the similarity/distance function.
    """
    return norm(a-b)


def relative_representation(dataframe: pl.DataFrame,
                            languages: list[str],
                            anchors: list[str] | set[str]) -> pl.DataFrame:
    """Defined a set of anchors, maps the absolute representation to a relative one.

    Args:
        dataframe : pl.DataFrame
            The dataframe containing the absolute representetion for each language token.
        languages : list[str]
            A list of languages for which we perform the relative representation.
        anchors : list[str] | set[str]
            A set of anchors.

    Returns:
        relative_reps : pl.DataFrame
            The pl.DataFrame with the relative representations.
    """
    # Order the set of anchors as a list
    anchors = list(anchors)
    anchors.sort()

    relative_repr = pl.DataFrame()
    for language in languages:
        # Filter the dataframe for the current language
        df = dataframe.filter(pl.col('language')==language)

        # Get the tokens
        tokens = df['Token'].to_list()
        
        embeddings = []
        for anchor in tqdm(anchors, desc=language):
            # Get the anchors absolute representation and use it as an axis
            axis = df.filter(pl.col('Token')==anchor)['Embedding'].item().to_numpy()

            # Remap the tokens to the new axis
            embeddings.append(list(map(lambda x: similarity(axis, np.array(x)),
                                       df['Embedding'].to_list())))

        # Save the results in which the anchors are columns
        current = pl.concat([pl.DataFrame({'language': language,
                                           'Token': tokens}),
                             pl.DataFrame(dict(zip(anchors, list(np.array(embeddings)))))],
                            how="horizontal")

        relative_repr.vstack(current,
                             in_place=True)

    return relative_repr


def semantic_differences(dataframe: pl.DataFrame,
                         languages: list[str],
                         anchors: list[str] | set[str]) -> dict[str, pl.DataFrame]:
    """Get the pairwise semantic difference between languages.
    
    Args:
        dataframe : pl.DataFrame
            The dataframe containing the absolute representetion for each language token.
        languages : list[str]
            A list of languages for which we perform the relative representation.
        anchors : list[str] | set[str]
            A set of anchors.

    Returns:
        relative_reps : dict[str, pl.DataFrame]
            The dictionary with the pl.DataFrame containing the semantic difference for each combination of languages.
    """
    results = dict()
    for t1, t2 in list(combinations(languages, 2)):
        # Get the sub-dataframe for each language
        t1_df = dataframe.filter(pl.col('language')==t1)
        t2_df = dataframe.filter(pl.col('language')==t2)
        
        # Get the common vocabulary
        common_vocab = list(set.intersection(*list(map(set, [t1_df['Token'].to_list(),
                                                             t2_df['Token'].to_list()])))-set(anchors))
        common_vocab.sort()

        # Get the relative representation
        t1_repr = t1_df.filter(pl.col('Token').is_in(common_vocab)).sort('Token', descending=False).select(anchors).to_numpy()
        t2_repr = t2_df.filter(pl.col('Token').is_in(common_vocab)).sort('Token', descending=False).select(anchors).to_numpy()

        # Get the semantic difference
        diff = []
        for t1_emb, t2_emb in tqdm(zip(t1_repr, t2_repr)):
            diff.append(norm(t1_emb-t2_emb))

        results[f'{t1} - {t2}'] = pl.DataFrame({
                                               'Token': common_vocab,
                                               t1: t1_repr,
                                               t2: t2_repr,
                                               'Semantic Distance': diff
                                            }).sort('Semantic Distance')
    return results


# ========================================================================
#
#                                MAIN LOOP
#
# ========================================================================

def main() -> None:
    """
    """
    return None


if __name__ == "__main__":
    main()
