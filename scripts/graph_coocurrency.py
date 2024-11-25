"""
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import csv
import polars as pl
from tqdm.auto import tqdm
from itertools import combinations

from src.utils import create_token_cooccurrence_graph

# Generator function to yield edges from documents
def generate_edges(document):
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
    # Define some paths
    CURRENT: Path = Path(".")
    DATA_PATH: Path = CURRENT / "data"

    # Variables
    platform: str = "reddit"
    communities: list[str] = ["conspiracy", "news", "politics"]

    for community in tqdm(communities):

        corpus = pl.scan_parquet(DATA_PATH / (platform + "/tokens/" + community + "_tokens.parquet")).select(pl.col("Texts")).collect()["Texts"].to_list()

        # graph = create_token_cooccurrence_graph(corpus)
        # graph.write_gml(platform + community + ".gml")
        
        # Open CSV file in write mode and create a writer object
        # with open(f'{platform}_{community}_edges.csv', mode='a', newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file)

        #     # Write the header if the file is empty (optional)
        #     writer.writerow(['from', 'to', 'weight'])

        #     # Iterate through each document and write each edge
        #     for document in tqdm(corpus, desc=community):
        #         for t1, t2, weight in generate_edges(document):
        #             # Write a single row to the CSV file for each edge
        #             writer.writerow([t1, t2, weight])
        pl.DataFrame(schema={"from":pl.String, "to":pl.String, "weight":pl.Int64}).write_csv(f"{platform}_{community}_edges.csv")
        from_list = []
        to_list = []
        weight_list = []
        for i, document in enumerate(tqdm(corpus, desc=community)):
            for t1, t2, weight in generate_edges(document):
                from_list.append(t1)
                to_list.append(t2)
                weight_list.append(weight)

            if i % 500 == 0:
                df = pl.DataFrame({"from": from_list,
                                   "to": to_list,
                                   "weight": weight_list}).group_by(["from", "to"]).agg(pl.sum("weight"))


                df = pl.concat([
                                   pl.read_csv(f"{platform}_{community}_edges.csv", schema={"from":pl.String, "to":pl.String, "weight":pl.Int64}),
                                   df
                               ], how="vertical").group_by(["from", "to"]).agg(pl.sum("weight"))
                
                df.write_csv(f"{platform}_{community}_edges.csv")

                del df
                
                from_list = []
                to_list = []
                weight_list = []
        
                
    
    return None

if __name__ == "__main__":
    main()
