from typing import Tuple


def example_labeling_function(v: int) -> int:
    class IncorrectIdentifierError(Exception):
        """
        This exception is thrown if you specify negative identifier
        """

        def __init__(self):
            pass

    if 0 <= v <= 16379:
        return 1
    elif v > 0:
        return 2
    else:
        raise IncorrectIdentifierError()


def labeling_function_50k(v: int) -> int:
    class IncorrectIdentifierError(Exception):
        """
        This exception is thrown if you specify negative identifier
        """

        def __init__(self):
            pass

    n = [171491, 174088, 167485, 157935, 175332, 131226, 183846, 170124, 174188, 176775]
    if v in n:
        return 2
    elif v >= 0:
        return 1
    else:
        print('No! v='+str(v))
        raise IncorrectIdentifierError()


class EdgesMatcher:
    def __init__(self, eps):
        self.eps = eps

    # Define the function that compares two edges:
    def edges_matching(self, e1_type, e1_weight, e2_type, e2_weight):
        return e1_type == e2_type and abs(e1_weight - e2_weight) < self.eps

    # Define the function that returns edge type
    @staticmethod
    def edge_type(v1_label, v2_label):
        if v1_label == 1:
            if v2_label == 1:
                return 1
            else:
                return 2
        else:
            if v2_label == 1:
                return 3
            else:
                return 2

    # Define the simpler-to-call edges matching function:
    def e_matching(self, e1: Tuple[int, int, float], e2: Tuple[int, int, float]) -> bool:
        return self.edges_matching(EdgesMatcher.edge_type(e1[0], e1[1]), e1[2],
                                   EdgesMatcher.edge_type(e2[0], e2[1]), e2[2])
