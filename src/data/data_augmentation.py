import torch

def apply_interpol(inputs : torch.Tensor, scores : torch.Tensor, alpha  : float = 0.5) -> (torch.Tensor,torch.Tensor):
    idx_a,idx_b = get_lowest_score_diff(scores)
    inputs[idx_b] = torch.lerp(inputs[idx_a], inputs[idx_b], alpha)
    scores[0][idx_b] = torch.lerp(scores[0][idx_a],scores[0][idx_b],alpha)
    return inputs,scores


def get_lowest_score_diff(scores : torch.Tensor)-> (int,int):
    differences = torch.abs(scores.view(-1, 1) - scores.view(1, -1))
    differences = differences.masked_fill(torch.eye(scores.numel()) == 1, float('inf'))
    min_indices = torch.argmin(differences)

    row_index = min_indices // differences.size(0)
    col_index = min_indices % differences.size(0)

    return row_index.item(),col_index.item()



