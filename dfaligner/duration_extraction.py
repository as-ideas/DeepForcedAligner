import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def to_node_index(i, j, cols):
    return cols * i + j


def from_node_index(node_index, cols):
    return node_index // cols, node_index % cols


def to_adj_matrix(mat):
    rows = mat.shape[0]
    cols = mat.shape[1]

    row_ind = []
    col_ind = []
    data = []

    for i in range(rows):
        for j in range(cols):

            node = to_node_index(i, j, cols)

            if j < cols - 1:
                right_node = to_node_index(i, j + 1, cols)
                weight_right = mat[i, j + 1]
                row_ind.append(node)
                col_ind.append(right_node)
                data.append(weight_right)

            if i < rows - 1 and j < cols:
                bottom_node = to_node_index(i + 1, j, cols)
                weight_bottom = mat[i + 1, j]
                row_ind.append(node)
                col_ind.append(bottom_node)
                data.append(weight_bottom)

            if i < rows - 1 and j < cols - 1:
                bottom_right_node = to_node_index(i + 1, j + 1, cols)
                weight_bottom_right = mat[i + 1, j + 1]
                row_ind.append(node)
                col_ind.append(bottom_right_node)
                data.append(weight_bottom_right)

    adj_mat = coo_matrix((data, (row_ind, col_ind)), shape=(rows * cols, rows * cols))
    return adj_mat.tocsr()


def extract_durations_with_dijkstra(tokens: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Extracts durations from the attention matrix by finding the shortest monotonic path from
    top left to bottom right.
    """

    pred_max = pred[:, tokens]
    path_probs = 1.0 - pred_max
    adj_matrix = to_adj_matrix(path_probs)
    dist_matrix, predecessors = dijkstra(
        csgraph=adj_matrix, directed=True, indices=0, return_predecessors=True
    )
    path = []
    pr_index = predecessors[-1]
    while pr_index != 0:
        path.append(pr_index)
        pr_index = predecessors[pr_index]
    path.reverse()

    # append first and last node
    path = [0] + path + [dist_matrix.size - 1]
    cols = path_probs.shape[1]
    mel_text = {}
    durations = np.zeros(tokens.shape[0], dtype=np.int32)

    # collect indices (mel, text) along the path
    for node_index in path:
        i, j = from_node_index(node_index, cols)
        mel_text[i] = j

    for j in mel_text.values():
        durations[j] += 1

    return durations


def extract_durations_beam(
    tokens: np.ndarray, pred: np.ndarray, k: int
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    data = pred[:, tokens]
    sequences = [[[0], -np.log(data[0, 0])]]  # always start on first position
    for row in data[1:]:
        all_candidates = []
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in [seq[-1], seq[-1] + 1]:  # only allow 2 possible moves
                if j < data.shape[-1]:
                    candidate = [seq + [j], score - np.log(row[j])]
                else:
                    candidate = [seq + [j], np.inf]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    durations = [np.bincount(sequence[0]) for sequence in sequences]
    return durations, sequences
