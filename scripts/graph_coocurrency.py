"""This python module handles the training of the language model.

    To check the available parameters run 'python /path/to/graph_coocurrency.py --help'.
"""

# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import polars as pl
from tqdm.auto import tqdm
from itertools import combinations


# ========================================================================
#
#                         FUNCTIONS DEFINITION
#
# ========================================================================

def generate_edges(document: list[str]):
    """Generator that yields each possible pair combination of tokens given a document.

    Args:
        document : list[str]
            The document defined as a list of tokens.

    Returns:
        tuple[str, str, int]
            The tuple (t1, t2, 1).
    """
    for pair in combinations(document, 2):
        yield *pair, 1


# ========================================================================
#
#                                MAIN LOOP
#
# ========================================================================

def main() -> None:
    """The main loop.
    """
    import argparse

    description = """
    This python module handles the training of the language model.

    To check the available parameters run 'python /path/to/graph_coocurrency.py --help'.
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-p',
                        '--platform',
                        help='The platform.',
                        type=str,
                        required=True)

    args = parser.parse_args()

    # Define some paths
    CURRENT: Path = Path(".")
    DATA_PATH: Path = CURRENT / "data"
    DUMP_PATH: Path = CURRENT / f"edges/{args.platform}"

    # Control over paths
    DUMP_PATH.mkdir(parents=True, exist_ok=True)

    # Variables
    communities: list[str] = [language.lower() for language in pl.scan_parquet(DATA_PATH / f'{args.platform}/{args.platform}.parquet').select("language").unique().sort('language').collect().get_column('language').to_list()]

    for community in tqdm(communities):
        corpus = pl.scan_parquet(DATA_PATH / (args.platform + "/tokens/" + community + "_tokens.parquet")).select(pl.col("Texts")).collect()["Texts"].to_list()

        df = pl.DataFrame(schema={"from":pl.String, "to":pl.String, "weight":pl.Int64})
        from_list = []
        to_list = []
        weight_list = []
        for i, document in enumerate(tqdm(corpus, desc=community)):
            for t1, t2, weight in generate_edges(document):
                from_list.append(t1)
                to_list.append(t2)
                weight_list.append(weight)

            if i % 10000 == 0:
                # Collect the edges in batches
                current = pl.DataFrame({"from": from_list,
                                        "to": to_list,
                                        "weight": weight_list}).group_by(["from", "to"]).agg(pl.sum("weight"))
                df.vstack(current, in_place=True).group_by(["from", "to"]).agg(pl.sum("weight"))
                
                from_list = []
                to_list = []
                weight_list = []

            if i % 50000 == 0:
                # Dump the current edges, clean the dataframe and continue
                df.write_parquet(DUMP_PATH / f"{i}.parquet")
                del df
                from_list = []
                to_list = []
                weight_list = []

                df = pl.DataFrame(schema={"from":pl.String, "to":pl.String, "weight":pl.Int64})

        # Save the last batch of edges
        current = pl.DataFrame({"from": from_list,
                                "to": to_list,
                                "weight": weight_list}).group_by(["from", "to"]).agg(pl.sum("weight"))
        df.vstack(current, in_place=True).group_by(["from", "to"]).agg(pl.sum("weight"))
    
    return None


if __name__ == "__main__":
    main()
