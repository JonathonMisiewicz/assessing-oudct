from . import main
from .classes import Operator, Diagram

from collections import Counter

def is_tensorially_connected(diagram: Diagram) -> bool:
    """ Return whether the diagram is a connected tensor. The final contraction pattern is connected.
    If it wasn't, we wouldn't have a complete contraction. But because the central RDM operator has
    a coefficient of 1, that doesn't guarantee that the final tensor is connected. Connected contractions
    meaning a connected term only holds if all your coefficients are size-extensive.
    """
    # Assume term 0 is the only one without a coefficient tensor.
    connected_terms = {0, 1}
    unprocessed_terms = [1] 
    while unprocessed_terms:
        operator = diagram.operators[unprocessed_terms.pop(0)]
        for row in [operator.upper, operator.lower]:
            for key in row:
                connected_term = int(key[1:])
                if connected_term not in connected_terms:
                    connected_terms.add(connected_term)
                    unprocessed_terms.append(connected_term)
        if len(connected_terms) == len(diagram):
            return True
    return False

def VUCC_class(diagram: Diagram) -> str:
    """ Given a diagram with variational stationarity, determine its class.

    Input
    -----
    diagram: list of [Counter, Counter]

    Output
    ------
    string
    """
    return "Connected" if is_tensorially_connected(diagram) else "Disconnected"

def PUCC_class(diagram: Diagram) -> str:
    """ Given a diagram with projective stationarity, determine its class.

    Input
    -----
    diagram: list of [Counter, Counter]

    Output
    ------
    string
    """
    return str(abs(diagram.excitation_rank()))

def variational(counter):
    """ Given the central RDM operator, compute the exponential
    similarity transform with variational stationarity.

    Input
    -----
    counter: [Counter, Counter]

    Output
    ------
    list of list of Tensor
        The outer list is the commutator number.
    """

    results = main.exponential_similarity_transform(counter, "variational", VUCC_class)
    return [x.get("Connected", []) for x in results]

def projective(counter):
    """ Given the central RDM operator, compute the exponential
    similarity transform with projective stationarity.

    Input
    -----
    counter: [Counter, Counter]

    Output
    ------
    None
    """
    main.exponential_similarity_transform(counter, "projective", PUCC_class)

OOOO = Operator(Counter({"o": 2}), Counter({"o": 2}))
VVVV = Operator(Counter({"v": 2}), Counter({"v": 2}))
OVOV = Operator(Counter({"o": 1, "v": 1}), Counter({"o": 1, "v": 1}))
OOOV = Operator(Counter({"o": 2}), Counter({"o": 1, "v": 1}))
OOVV = Operator(Counter({"o": 2}), Counter({"v": 2}))
OVVV = Operator(Counter({"o": 1, "v": 1}), Counter({"v": 2}))
OO = Operator(Counter({"o": 1}), Counter({"o": 1}))
OV = Operator(Counter({"o": 1}), Counter({"v": 1}))
VV = Operator(Counter({"v": 1}), Counter({"v": 1}))

### VARIATIONAL CODE
#d_dict = {"oo": {}, "ov": {}, "vv": {}}
#oovv_result = variational(OOVV)
#d_dict["oo"]["o"] = variational(OOOO)
#d_dict["oo"]["v"] = d_dict["vv"]["o"] = variational(OVOV)
#d_dict["ov"]["o"] = variational(OOOV)
#d_dict["ov"]["v"] = variational(OVVV)
#d_dict["vv"]["v"] = variational(VVVV)
#for x in [OO, OV, VV]:
#    variational(x)

#main.d_terms(d_dict)

### PROJECTIVE CODE
for x in [OOOO, OVOV, OOOV, OOVV, OVVV, VVVV, OO, OV, VV]:
    projective(x)
