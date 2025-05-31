import pandas as pd
import networkx as nx


def build_movie_interaction_graphs(users_csv: str, utterances_csv: str):
    """
    Build a conversation network for each movie.
    Nodes represent characters and edges represent interactions
    between characters, weighted by the number of adjacent utterances.
    """
    # Load data
    users = pd.read_csv(users_csv)
    utterances = pd.read_csv(utterances_csv)

    # Merge to associate utterances with movie and character names
    users_df = users[['user_id', 'character_name', 'movie_idx']]
    utterances = utterances.merge(
        users_df,
        left_on='speaker',
        right_on='user_id',
        how='left'
    )

    # Initialize dictionary to hold graphs
    movie_graphs = {}

    # Process each movie
    for movie_idx, group in utterances.groupby('movie_idx', sort=False):
        # Create an undirected graph
        G = nx.Graph()

        # Add nodes for each character
        for name in group['character_name'].unique():
            G.add_node(name)

        # Count interactions based on adjacent utterances
        sequence = group['character_name'].tolist()
        weights = {}
        for u, v in zip(sequence, sequence[1:]):
            if u != v:
                edge = tuple(sorted((u, v)))
                weights[edge] = weights.get(edge, 0) + 1

        # Add weighted edges to the graph
        for (u, v), w in weights.items():
            G.add_edge(u, v, weight=w)

        movie_graphs[movie_idx] = G

    return movie_graphs


if __name__ == '__main__':
    # Example usage
    graphs = build_movie_interaction_graphs(
        'movie_users.csv',
        'movie_utterances.csv'
    )

    # Print summary for each movie
    for movie, G in graphs.items():
        print(f"Movie {movie}: {G.number_of_nodes()} characters, {G.number_of_edges()} interactions")
