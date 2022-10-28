import torch
import torch.nn as nn
from entmax import sparsemax


_EPSILON = 1e-6

def _vector_norms(v:torch.Tensor)->torch.Tensor:
    """ Computes the vector norms
    Args:
        v: The vector from which there must be calculated the norms

    Returns:
            A tensor containing the norms of input vector v
    """

    squared_norms = torch.sum(v * v, dim=1, keepdim=True)
    return torch.sqrt(squared_norms + _EPSILON)

def _distance(x:torch.Tensor , y:torch.Tensor, type:str='cosine')->torch.Tensor:
        """ Compute distances (or other similarity scores) between
        two sets of samples. Adapted from https://github.com/oscarknagg/few-shot/blob/672de83a853cc2d5e9fe304dc100b4a735c10c15/few_shot/utils.py#L45

        Args:
            x (torch.Tensor):  A tensor of shape (a, b) where b is the embedding dimension. In our paper a=1
            y (torch.Tensor):  A tensor of shape (m, b) where b is the embedding dimension. In our paper m is the number of samples in support set.
            type (str, optional): Type of distance to use. Defaults to 'cosine'. Possible values: cosine, l2, dot

        Raises:
            NameError: if the name of similarity is unknown

        Returns:
            torch.Tensor: A vector contining the distance of each sample in the vector y from vector x
        """
        if type == 'cosine':
            x_norm = x / _vector_norms(x)
            y_norm = y / _vector_norms(y)
            d = 1 - torch.mm(x_norm,y_norm.transpose(0,1))
        elif type == 'l2':
            d = (
                x.unsqueeze(1).expand(x.shape[0], y.shape[0], -1) -
                y.unsqueeze(0).expand(x.shape[0], y.shape[0], -1)
        ).pow(2).sum(dim=2)
        elif type == 'dot':
            expanded_x = x.unsqueeze(1).expand(x.shape[0], y.shape[0], -1)
            expanded_y = y.unsqueeze(0).expand(x.shape[0], y.shape[0], -1)
            d = -(expanded_x * expanded_y).sum(dim=2)
        else:
            raise NameError('{} not recognized as valid distance. Acceptable values are:[\'cosine\',\'l2\',\'dot\']'.format(type))
        return d



        
def baseline_cosine(encoder_output:torch.Tensor, memory_set:torch.Tensor)->torch.Tensor:

        dist = _distance(encoder_output,memory_set,'cosine')
        content_weights = sparsemax(-dist,dim=1)
        memory_vector = torch.matmul(content_weights,memory_set)
 
        return memory_vector
