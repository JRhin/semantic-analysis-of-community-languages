"""
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import polars as pl
from tqdm.auto import tqdm

from src.utils import create_token_cooccurrence_graph

def main() -> None:
    """The main loop.
    """
    # Define some paths
    CURRENT: Path = Path(".")
    DATA_PATH: Path = CURRENT / "data"

    # Variables
    platform: str = "reddit"
    communities: list[str] = ["conspiracy", "news", "politics"]

    for community in tqdm(communities):

        corpus = pl.scan_parquet(DATA_PATH / (platform + "/tokens/" + community + "_tokens.parquet")).select(pl.col("Texts")).collect()["Texts"].to_list()

        graph = create_token_cooccurrence_graph(corpus)
        graph.write_gml(platform + community + ".gml")
    
    return None

if __name__ == "__main__":
    main()
