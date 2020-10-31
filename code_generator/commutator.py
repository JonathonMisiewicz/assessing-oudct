from collections import Counter
from copy import deepcopy
import itertools

from .classes import Operator, Diagram

occupied_generic = "o"
virtual_generic = "v"

def commutator_with(diagrams, simplifications, cluster_ranks):
    """Compute the result of taking a commutator of the diagrams with excitation and de-excitation operators
    of the given cluster ranks, accounting for the available simplifications.

    Input
    -----
    diagrams: iterable of Diagrams
        An iterable containing the diagrams.
    simplifications: list of str
        A list of strings representing various simplificaitons/adjustments to the commutator algorithm.    
    cluster_ranks: iterable of int
        The ranks of excitation/de-excitation operators allowed.

    Output
    ------
    list of Diagrams
    """
    diagrams = list(diagrams)
    diagrams = add_new_SQ(diagrams, simplifications, cluster_ranks)
    diagrams = contract_new_SQ(diagrams, simplifications)
    return filter(lambda x: x.operators[-1].contracted(), diagrams)

def add_new_SQ(diagrams, simplifications, cluster_ranks):
    """Determine which addition algorithm to employ and return the result.

    Input
    -----
    diagrams: iterable of list of [Counter, Counter]
        An iterable containing the diagrams.
    simplifications: list of str
        A list of strings representing various simplificaitons/adjustments to the commutator algorithm.    
    cluster_ranks: iterable of int
        The ranks of excitation/de-excitation operators allowed.

    Output
    ------
    list of Diagram"""
    func = fc_only_add if simplifications.get("fc_only", False) else brute_force_add
    return func(diagrams, cluster_ranks)

def brute_force_add(diagrams, cluster_ranks):
    """
    Return a list of diagrams with all possible (de-)excitation operators appended.
    At this time, no contractions are performed.

    Input
    -----
    diagrams: iterable of list of [Counter, Counter]
        An iterable containing the diagrams.
    cluster_ranks: iterable of int
        The ranks of excitation/de-excitation operators allowed.
    
    Output
    ------
    list of Diagram
    """
    operators = [make_excite_rank(rank) for rank in cluster_ranks]
    operators += [make_deexcite_rank(rank) for rank in cluster_ranks]
    # Older code had a copy here. Probably not necessary.
    new_diagrams = [Diagram(diagram.operators + [operator]) for diagram, operator in itertools.product(diagrams, operators)]
    return new_diagrams

def fc_only_add(diagrams, cluster_ranks):
    """
    Return a list of diagrams with a the new diagram appended allowing for a complete contraction.
    If no complete contractions are possible, don't add to the list.
    At this time, no contractions are performed.

    Input
    -----
    diagrams: iterable of list of [Counter, Counter]
        An iterable containing the diagrams.
    cluster_ranks: iterable of int
        The ranks of excitation/de-excitation operators allowed.
    
    Output
    ------
    list of list of Diagram
    """
    new_diagrams = []

    for diagram in diagrams:
        rank = diagram.excitation_rank()
        if rank is None or abs(rank) not in cluster_ranks:
            continue
        elif rank > 0:
            new_term = make_deexcite_rank(rank)
        elif rank < 0:
            new_term = make_excite_rank(-rank)
        new_diagrams.append(Diagram(diagram.operators + [new_term]))
    return new_diagrams

def make_excite_rank(rank):
    """Return an excitation operator of the desired rank.

    Input
    -----
    rank: int
        The rank of the excitation operator.

    Output
    ------
    Operator"""
    return Operator(Counter({"v": rank}), Counter({"o": rank}))

def make_deexcite_rank(rank):
    """Return a de-excitation operator of the desired rank.

    Input
    -----
    rank: int
        The rank of the excitation operator.

    Output
    ------
    [Counter, Counter]"""
    temp = make_excite_rank(rank)
    return Operator(temp.lower, temp.upper)

def contract_new_SQ(diagrams, simplifications):
    """
    Given a list of diagrams where the last is uncontracted, return all possible diagrams where the last
    operator is contracted with one of the previous diagrams.

    Input
    -----
    diagrams: list of Diagrams
        An iterable containing the diagrams. The last operator is uncontracted.
    simplifications: list of str
        A list of strings representing various simplificaitons/adjustments to the commutator algorithm.    

    Output
    ------
    list of list of [Counter, Counter]
    """
    if not diagrams:
        return diagrams
    num_partner_terms = len(diagrams[0]) - 1
    for partner_idx in range(num_partner_terms):
        diagrams = itertools.chain(*[partner_contractions(diagram, partner_idx, simplifications) for diagram in diagrams])
    return diagrams

def partner_contractions(diagram, term_num, simplifications):
    """
    Perform all contractions (including none) of the given diagram involving the term_num operator.
    Apply the simplifications when determining which contractions to perform.

    Input
    -----
    diagram: Diagram
        The diagram to contract.
    term_num: int
        The index of the term that the last diagram will contract with.
    simplifications: list of str
        A list of strings representing various simplificaitons/adjustments to the commutator algorithm.    

    Output
    ------
    list of Diagrams
    """
    operators = diagram.operators
    new_diagrams = []
    contractions1 = row_contractions(operators[term_num].upper, operators[-1].lower, term_num, len(diagram) - 1, simplifications)
    contractions2 = row_contractions(operators[term_num].lower, operators[-1].upper, term_num, len(diagram) - 1, simplifications)
    for (upper_med, lower_last), (lower_med, upper_last) in itertools.product(contractions1, contractions2):
        new_diagram = deepcopy(operators)
        new_diagram[term_num] = Operator(upper_med, lower_med)
        new_diagram[-1] = Operator(upper_last, lower_last)
        new_diagrams.append(Diagram(new_diagram))
    return new_diagrams

def row_contractions(row1, row2, term_num1, term_num2, simplifications):
    """ Given two rows, return the results of all possible contractions between them, subject
    to the given simplifications.

    Input
    -----
    row1, row2: Counter
        The two rows (one upper, one lower) that we're contracting between.
    term_num1, term_num2: int
        The indices of the terms in the diagram we're contracting.
        The contractions are recorded in the counter by the original symbol plus the term number of what
        it contracts with.
    simplifications: list of str
        A list of strings representing various simplificaitons/adjustments to the commutator algorithm.    

    Output
    ------
    list of [Counter, Counter]
    """
    symbol = occupied_generic if occupied_generic in row2 else virtual_generic
    contractions_list = list()
    max_contractions = min(row1[symbol], row2[symbol])
    min_contractions = max_contractions if simplifications.get("fc_only", False) else 0
    for num_contraction in range(min_contractions, max_contractions + 1): 
        new1 = row1.copy()
        new2 = row2.copy()
        new1[symbol] -= num_contraction
        new2[symbol] -= num_contraction
        new1[symbol + str(term_num2)] += num_contraction
        new2[symbol + str(term_num1)] += num_contraction
        # Purge 0 values from the new rows
        new1 += Counter()
        new2 += Counter()
        contractions_list.append([new1, new2])
    return contractions_list

