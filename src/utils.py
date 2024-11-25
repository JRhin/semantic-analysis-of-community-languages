"""General utils functions.
"""
import polars as pl
import igraph as ig
from igraph import Graph
from tqdm.auto import tqdm
from collections import defaultdict


# ========================================================================
#
#                        FUNCTIONS DEFINITION
#
# ========================================================================

def create_token_cooccurrence_graph(corpus: list[list[str]]) -> Graph:
    """Create a directed graph from a corpus where tokens are nodes,
    and edges represent co-occurrence counts in the same document.

    Args:
        corpus : list[list[str]]:
            A corpus of documents represented as a list of lists of tokens.

    Returns:
        graph : igraph.Graph
            A directed graph where edges are weighted by co-occurrence counts.
    """
    # Dictionary to store co-occurrence counts
    cooccurrence_counts = defaultdict(int)

    # Unique tokens
    unique_tokens = set()

    # Compute co-occurrence counts
    for document in tqdm(corpus):
        # Consider unique tokens per document
        tokens = set(document)
        unique_tokens.update(tokens)
        for token1 in tokens:
            for token2 in tokens:
                # Ignore self-loops
                if token1 != token2:
                    cooccurrence_counts[(token1, token2)] += 1

    # Create mapping of tokens to indices
    token_to_index = {token: idx for idx, token in enumerate(unique_tokens)}

    # Create graph
    graph = Graph(directed=True)
    graph.add_vertices(len(unique_tokens))
    graph.vs["name"] = list(unique_tokens)

    # Add edges with weights
    edges = []
    weights = []
    for (token1, token2), count in cooccurrence_counts.items():
        edges.append((token_to_index[token1], token_to_index[token2]))
        weights.append(count)

    graph.add_edges(edges)
    graph.es["weight"] = weights

    return graph
                                                                                                                                                                                                                                                        

def simplify_edgelist(edgelist: pl.LazyFrame) -> pl.LazyFrame:
  """Simplifies the edgelist by removing self-loops, aggregating weights, 
  and preserving the first value of other attributes for each unique edge.

  This function is useful for preprocessing a network's edge list to 
  ensure that:
  - Self-loops (edges where the source and target nodes are the same) are excluded.
  - Duplicate edges (edges with the same source and target nodes) are aggregated:
    - The weights are summed.
    - Other attributes are reduced by taking their first occurrence.

  Args:
    edgelist : pl.LazyFrame: 
      A Polars LazyFrame containing the edge list. 
      Expected columns include:
      - "from": Source node of the edge.
      - "to": Target node of the edge.
      - "weight" (optional): Numeric weights associated with edges.
      Additional columns can also be present.

  Returns:
    pl.LazyFrame
      A simplified edge list as a Polars LazyFrame. The resulting LazyFrame includes:
      - "from": Source node of the edge.
      - "to": Target node of the edge.
      - "weight" (if present): Aggregated sum of weights for the edge.
      - Other attributes reduced to their first occurrence.

  Example:
    ```python
    # Example edgelist
    edgelist = pl.DataFrame({
        "from": [1, 2, 1, 3],
        "to": [2, 3, 2, 3],
        "weight": [0.5, 1.0, 0.8, 2.0],
        "attribute": ["a", "b", "c", "d"]
    })

    # Simplify the edgelist
    simplified = simplify_edgelist(edgelist.lazy())
    print(simplified.collect())
    ```
  """
  return (
      edgelist
      .filter(pl.col("from") != pl.col("to"))
      .group_by(["from", "to"])
      .agg([
          pl.col("weight").sum(),
          pl.all().exclude("weight").first()
      ])
  )


def get_backbone(edgelist: pl.LazyFrame | pl.DataFrame,
                 mode: str,
                 from_col: str, 
                 to_col: str,
                 weight_col: str = None,
                 alpha: float = 0.05) -> pl.LazyFrame:
  """Compute the statistical backbone of a network for both directed and undirected modes 
  using the disparity filter method. This method identifies significant edges based on 
  their weights relative to their connected nodes.

  Args:
    edgelist : pl.LazyFrame | pl.DataFrame 
      The input edgelist represented as a Polars LazyFrame or DataFrame.
      Must contain at least the `from_col` and `to_col` columns.
    mode : str 
      Specifies the type of network. Accepted values:
      - 'undirected': Treat the network as undirected.
      - 'directed': Treat the network as directed.
      - 'in': Compute the backbone based on incoming connections only.
      - 'out': Compute the backbone based on outgoing connections only.
    from_col : str 
      The column name representing the source node of each edge.
    to_col : str 
      The column name representing the target node of each edge.
    weight_col : str, optional 
      The column name representing the edge weight. If `None`, weights are assumed 
      to be 1 for all edges. Defaults to `None`.
    alpha : float, optional 
      The significance level for filtering edges. Edges with a calculated 
      significance score (`a`) less than `alpha` are retained. Defaults to 0.05.

  Returns:
    pl.LazyFrame: 
      A Polars LazyFrame representing the backbone of the network. The result will 
      contain only statistically significant edges with columns:
      - `from`: The source node.
      - `to`: The target node.
      - `weight` (if `weight_col` is provided): The edge weight.

  Raises:
    ValueError:
      - If `alpha` is not in the range (0, 1].
      - If `mode` is not one of the accepted values ('undirected', 'directed', 'in', 'out').

  Notes:
    - In the undirected case, the method ensures that edges (u, v) and (v, u) are treated 
      as identical and removes duplicates.
    - The backbone is computed using the disparity filter model, which analyzes the 
      distribution of edge weights for each node to identify statistically significant edges.
    - For directed networks, the backbone can be computed separately for incoming ('in') 
      or outgoing ('out') edges, or both ('directed').

  Example:
    ```python
    # Example edgelist
    edgelist = pl.DataFrame({
        "from": [1, 2, 3, 2, 4],
        "to": [2, 3, 1, 4, 5],
        "weight": [0.5, 1.2, 0.8, 1.1, 0.9]
    })

    # Compute undirected backbone
    backbone = get_backbone(edgelist, mode='undirected', from_col='from', to_col='to', weight_col='weight', alpha=0.05)
    print(backbone.collect())
    ```
  """
  def compute_backbone(edgelist: pl.LazyFrame,
                       col: str):
    """
    Compute the statistical backbone of a network using the disparity filter method.

    This function calculates the backbone of a network by identifying statistically significant 
    edges based on their weights relative to their connected nodes. The significance is determined 
    using a probabilistic model that compares the edge's weight to the total weight of all edges 
    connected to the node specified by the `col` parameter.

    Args:
      edgelist : pl.LazyFrame
        A Polars LazyFrame representing the network's edge list. 
        The DataFrame must include the following columns:
        - "from": Source node of the edge.
        - "to": Target node of the edge.
        - "weight": Numeric weight of the edge.
      col : str 
        The column name to compute the backbone for. It determines the perspective 
        of the calculation. Accepted values:
        - "from": Compute the backbone based on outgoing edges (source node perspective).
        - "to": Compute the backbone based on incoming edges (target node perspective).

    Returns:
      pl.LazyFrame: 
        A Polars LazyFrame representing the backbone of the network. The result includes:
        - "from": Source node of the edge.
        - "to": Target node of the edge.
        - "weight": Weight of the retained edge.

    Raises:
      AssertionError: 
        If `col` is not one of the accepted values ("from", "to").

    Notes:
      - For each node in the specified column (`col`), the function computes:
          - `k`: The number of edges connected to the node (degree).
          - `p`: The proportion of the edge's weight relative to the total weight 
            of all edges connected to the node.
          - `a`: A statistical significance score derived from the disparity filter model.
      - Edges with a significance score (`a`) less than a predefined threshold (`alpha`) 
        are retained in the backbone.
      - Ensure that the `alpha` value is defined globally or passed as a parameter before 
        calling this function.

    Example:
      ```python
      # Example edgelist
      edgelist = pl.DataFrame({
          "from": [1, 2, 1, 3],
          "to": [2, 3, 2, 3],
          "weight": [0.5, 1.0, 0.8, 2.0]
      })

      # Compute the backbone for outgoing edges
      backbone = compute_backbone(edgelist.lazy(), col="from")
      print(backbone.collect())
      ```
    """
    # Validate arguments
    assert col in ["from", "to"], "The 'col' parameter must be in {'from', 'to'}"

    edgelist = (
        edgelist
        .with_columns(
            k = pl.len().over(col),
            p = pl.col("weight") / pl.sum("weight").over(col))
        .with_columns(
            a = (1 - pl.col("p")).pow(pl.col("k")-1)
        )
    )

    edgelist_bb = (
        edgelist
        .filter(pl.col("a") < alpha)
        .select(["from", "to", "weight"])
    )
    return edgelist_bb


  # Validate arguments
  assert 0 < alpha <= 1, "The 'alpha' parameter must be in (0, 1]."
  assert mode in {"undirected", "directed", "in", "out"}, "The 'mode' parameter must be in {'undirected', 'directed', 'in', 'out'}"

  # Rename columns and add missing weight column if needed
  columns = {from_col: "from", to_col: "to"}
  if weight_col:
      columns[weight_col] = "weight"
  else:
      edge_list = edge_list.with_columns(weight=pl.lit(1))

  # Rename the columns, filter the weights and cast them to the right type
  edgelist = (
      edgelist
      .lazy()
      .rename(columns)
      .filter(pl.col("weight") > 0)
      .with_columns(
          pl.col("weight").cast(pl.Float64)
      )
  )

  # Simplify the network by removing self-loops, summing weights,
  # and keeping the first value for other attributes.
  edgelist = simplify_edgelist(edgelist)

  # Handle each case of the 'mode' parameter
  match mode:
    case "undirected":
      # We need to futher simplify the network
      # by removing duplicate undirected edges
      # (i.e., (u,v) and (v,u) is the same)
      edgelist = (
          edgelist
          .with_columns(
              from_sorted = pl.when(pl.col("from") < pl.col("to")).then(pl.col("from")).otherwise(pl.col("to")),
              to_sorted = pl.when(pl.col("from") < pl.col("to")).then(pl.col("to")).otherwise(pl.col("from"))
          )
          .select(["from_sorted", "to_sorted", "weight"])
          .rename({"from_sorted": "from", "to_sorted": "to"})
      )

      # Simplify the network by removing self-loops, summing weights,
      # and keeping the first value for other attributes.
      edgelist = simplify_edgelist(edgelist)

      # take the edgelist and transform it so that
      # each edge appears both as (u,v) and (v,u)
      edgelist = (
          pl.concat([
                      edgelist,
                      edgelist.rename({"from": "to", "to": "from"})
                    ],
                    how="vertical")
      )

      # Get the backbone
      edgelist_bb = (
          compute_backbone(edgelist, "from")
          .with_columns(
              from_sorted = pl.when(pl.col("from") < pl.col("to")).then(pl.col("from")).otherwise(pl.col("to")),
              to_sorted = pl.when(pl.col("from") < pl.col("to")).then(pl.col("to")).otherwise(pl.col("from"))
          )
          .select(["from_sorted", "to_sorted", "weight"])
          .rename({"from_sorted": "from", "to_sorted": "to"})
          .unique()
      )

    case "directed":
      # Concat both "in" and "out" cases
      edgelist_bb = pl.concat([
                                compute_backbone(edgelist, "to"),
                                compute_backbone(edgelist, "from"),
                              ],
                              how="vertical")

    case "in":
      edgelist_bb = compute_backbone(edgelist, "to")

    case "out":
      edgelist_bb = compute_backbone(edgelist, "from")

  return edgelist_bb



# ========================================================================
#
#                                MAIN LOOP
#
# ========================================================================

def main() -> None:
    """The main loop.
    """
    corpus = [
            ["apple", "banana", "apple", "cherry"],
            ["banana", "cherry", "cherry", "date"],
            ["apple", "date", "banana", "apple"]
        ]

    edgelist = pl.DataFrame({
        "from": [1, 2, 3, 2, 4],
        "to": [2, 3, 1, 4, 5],
        "weight": [0.5, 1.2, 0.8, 1.1, 0.9]
    })

    print("First test...")
    graph = create_token_cooccurrence_graph(corpus)
    print("[DONE]")

    print()
    print("Second test...", end="\t")
    backbone = get_backbone(edgelist, mode='undirected', from_col='from', to_col='to', weight_col='weight', alpha=0.5)
    print("[DONE]")
    
    return None


if __name__ == "__main__":
    main()
