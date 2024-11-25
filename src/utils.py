"""General utils functions.
"""
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

    graph = create_token_cooccurrence_graph(corpus)

    # Print edges with weights
    for edge in graph.es:
        source = graph.vs[edge.source]["name"]
        target = graph.vs[edge.target]["name"]
        weight = edge["weight"]
        print(f"{source} -> {target} [weight: {weight}]")
    return None


if __name__ == "__main__":
    main()
