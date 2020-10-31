from collections import Counter
from copy import deepcopy
from fractions import Fraction
import itertools
from math import factorial
import sympy
from collections import OrderedDict
from .classes import Operator, Diagram
from . import commutator

cluster_ranks = [1, 2]
max_commutator = 4

occupied_contractable = ["i", "j", "k", "l", "m", "n", "o"]
occupied_contractable_distinguished = [i.upper() for i in occupied_contractable]
virtual_contractable = ["a", "b", "c", "d", "e", "f", "v"]
virtual_contractable_distinguished = [i.upper() for i in virtual_contractable]

### Type Documentations
"""
Indices are defined above.
Generic indices are uncontracted.
A row is a counter of indices. This corresponds to the top/bottom row of a second quantized tensor. The sum of values is the number of indices
An SQ is a list of two rows, each with the same number of indices. The first row is upper indices, second is the lower row.
"""

### SQ Utility Code
def simple_to_string(simple):
    return " ".join(["".join(row) for row in simple])

### Counter to Multiplicity Code

def to_multiplicity(diagrams):
    """
    We've created different diagrams for different orderings of the operators. Now let's find out how many there are of each
    equivalence class. We identify all equivalent terms an inefficient but conceptually simple way.
    Every time we see a unique term, generate everything permutationally equivalent to it.
    When we have a new term, check against all those permutations. If it's a match to nothing we've previously seen, it's a new term.
    Either way, update the count accordingly.
    Of course, a more efficient procedure wouldn't have generated all these permutation-related terms in the first place,
    but doing it this way is harder to get wrong.

    Input
    -----
    diagram: list of Diagram

    Output
    ------
    list of Diagram
    """
    unique_diagrams = []
    for diagram in diagrams:
        weight = Fraction(1, factorial(len(diagram) - 1))
        for unique_diagram in unique_diagrams:
            if unique_diagram.equiv(diagram):
                unique_diagram.prefactor += diagram.prefactor
                break
        else:
            unique_diagrams.append(diagram)
    return unique_diagrams

def string_list_antisym(string_list, externals):
    """ Given a list of antisymmetrizers, convert antisymmetrization over indices that are
        already antisymmetric to constants. We can thus remove them from the list.
        Return the weight from these redundant antisymmetrizers, and the nonredundant ones.

    Input
    -----
    stringlist: list of [list(str), list(str)] 
    externals: [(int, set(str))]
        First elt is the row number. Second elt is the antisymmetric occ/vir indices of that row.
        

    Output
    ------
    int
    list of list of str
    """
    weight = 1
    antisymmetrizers = set()
    for row_num, row in externals:
        asym_row = []
        for operator in string_list:
            new_found_chars = frozenset(filter(lambda x: x in row, operator[row_num]))
            if new_found_chars:
                weight *= factorial(len(new_found_chars))
                asym_row.append(new_found_chars)
        if len(asym_row) > 1:
            antisymmetrizers.add(frozenset(asym_row))
    return weight, antisymmetrizers


def get_perm_weight(diagram) -> int:
    """ Get the permutational weight.
    Input
    -----
    possibility: Diagram
    """
    # Get the permutational weight
    # "labeled" lines are not treated specially and are handled elsewhere
    product = 1
    for i, operator in enumerate(diagram.operators):
        for row in (operator.upper, operator.lower):
            for key, value in row.items():
                partner = key[1:]
                # This restriction is to avoid double-counting contracted lines
                if not partner or int(partner) > i:
                    product *= factorial(value)
    return product


def make_string_list(diagram, bare_indices):
    """
    Convert a Diagram into an ordered format (string_list) convenient for counting purposes.
    * A string_list is list of [list(str), list(str)].
        * Level 1: Operators in the Diagram
        * Level 2: Rows in the Operator
        * Level 3: Indices in the Row
    * Each index is a single letter from the lists of occupied and virtual contractables, distinguished
      and undistinguished. "Distinguished" status is based upon what will be external in the final product.
      The uncontracted indices and the indices contracted to something in bare_operators.
    For storing the final tensors, we will find it convenient to modify this string list later.
    To be explicit, the following is done later:
    * Flipping upper and lower rows to canonicalize the form of amplitude tensors.
    * Removing indices of bare operators.

    Input
    -----
    possibility: diagram
    bare_indices: operator indices (in the diagram) that should be capitalized, marking them as belonging to external
        Conceptualy, these have the trivial coefficient tensor.

    Output
    ------
    list of [list(str), list(str)]
    """

    # Copy the lists, so we can pop from them to keep track of which indices we've used.
    gen_occ_list = occupied_contractable.copy()
    gen_vir_list = virtual_contractable.copy()
    spec_occ_list = occupied_contractable_distinguished.copy()
    spec_vir_list = virtual_contractable_distinguished.copy()
    spec_list = lambda x: spec_occ_list if x[0] == "o" else spec_vir_list
    gen_list = lambda x: gen_occ_list if x[0] == "o" else gen_vir_list
    string_list = [ [list(), list()] for i in range(len(diagram))]

    for i, operator in enumerate(diagram.operators):
        for j, row in enumerate([operator.upper, operator.lower]):
            for symbol, count in row.items():
                contracted_to = symbol[1:]
                # Determine the symbol and the partner index.
                if contracted_to:
                    contracted_to = int(contracted_to)
                    if contracted_to < i:
                        # We already added this term. Skip it!
                        continue
                    elif {i, contracted_to} & bare_indices:
                        # This is or on or contracted to a bare operator. Special index!
                        symb_list = spec_list(symbol)
                    else:
                        # This is an internal index.
                        symb_list = gen_list(symbol)
                else:
                    # It's not a contracted index.
                    contracted_to = None
                    symb_list = spec_list(symbol)

                for _ in range(count):
                    symbol_to_add = symb_list.pop(0)
                    string_list[i][j].append(symbol_to_add)
                    if contracted_to is not None:
                        string_list[contracted_to][0 if j else 1].append(symbol_to_add)
            string_list[i][j].sort(key=sort_func)
    return string_list

def determine_sign(diagram) -> int:
    """Given a diagram, return its sign, accounting for holes and loops.
    
    Input
    -----
    diagram: list of list of str
    """

    creators = list(itertools.chain(*[term[0] for term in diagram]))
    annihilators = list(itertools.chain(*[term[1] for term in diagram]))
    sign = 1
    free_creators, free_annihilators = [], []

    def hole_increment(X):
        # Update the sign for occupied lines.
        return -1 if X.lower() in occupied_contractable else 1

    while creators:
        C, A = creators.pop(0), annihilators.pop(0)
        while True:
            try:
                new_idx = creators.index(A)
            except ValueError:
                # We have a free end.
                while True:
                    try:
                        new_idx = annihilators.index(C)
                    except ValueError:
                        # We found the other free end.
                        free_creators.append(C)
                        free_annihilators.append(A)
                        break
                    sign *= hole_increment(C)
                    C, _ = creators.pop(new_idx), annihilators.pop(new_idx)
                break
            sign *= hole_increment(A)
            _, A = creators.pop(new_idx), annihilators.pop(new_idx)
            if C == A:
                # Loop closed!
                sign *= -1 * hole_increment(A)
                break

    def find_sorted_parity(L):
        # The free terms should all be "distinguished"...
        return find_parity(L, sorted(L, key=sort_func))

    sign *= find_sorted_parity(free_creators) * find_sorted_parity(free_annihilators)

    return sign

def find_partner(diagram, permutation, row, symbol) -> int:
    """ Find the index of the operator in the diagram where a given symbol appears in a given row.
        If it is not found in any of the operators, return -1. Interpret the result as the index
        being external.

    Input
    -----
    diagram: Tensor
    permutation: tuple of int
    row: {0, 1}
        Should we look for the partner in the upper (0) or lower (1) row?
    symbol: string
        A character. Which symbol do we want to find the partner of?
    """
    for j, i in enumerate(permutation):
        if symbol in diagram[i][row]:
            return j
    else:
        return -1

def tensor_flip(indices):
    """ Given a string list, canonicalize which is on top and which on bottom.

    Input
    -----
    indices: list of list of str
        Outer list is length 2.

    Output
    ------
    list of list of str
    """
    upper, lower = indices
    lower_count = sum([i.lower() in occupied_contractable for i in lower])
    upper_count = sum([i.lower() in occupied_contractable for i in upper])
    return indices[::-1] if lower_count > upper_count else indices

def external_antisym(stringlist, stationarity):
    """ Determine all external indices and antisymmetrize all indices of the same occupancy.
        Leave it to another function to reduce the antisymmetrizer for indices that are
        already antisymmetric.

    Input
    -----
    stringlist: list of [list(str), list(str)]
        The current operators in the diagram. Outer list stores operators.
        Inner list stores the top and bottom row, each of which is a list of
        the indices of the operators.

    stationarity: string
        What is the stationarity condition?
    """
    # Depending on the stationarity condition, determine the external indices.
    # externalV also modifies the string_list to remove the bare operator. (0)
    external_func = externalV if stationarity == "variational" else externalP
    externals = external_func(stringlist)
    antisym_elts = []
    for i, row in enumerate(externals):
        for l in [occupied_contractable_distinguished, virtual_contractable_distinguished]:
            temp = set(filter(lambda x: x in l, row))
            if len(temp) > 1:
                antisym_elts.append((i, temp))
    return externals, antisym_elts

def externalV(stringlist):
    """ The indices to make external are those of the central operator. Remove it from the string
        and reverse top and bottom.

    Input
    -----
    stringlist: list of [list(str), list(str)]

    Output
    ------
    list of str """
    return stringlist.pop(0)[::-1]

def externalP(possibility_string):
    creators, annihilators = [], []
    for c_row, a_row in possibility_string:
        creators.extend(list(filter(lambda x: x.isupper(), c_row)))
        annihilators.extend(list(filter(lambda x: x.isupper(), a_row)))
    return [creators, annihilators]

def full_contract(tensor, symbol):
    """ Contract the external indices of a Tensor with a tensor with a new symbol.
        Example use case, going from an RDM formula to the energy contribution.

    Input
    -----
    tensor: Tensor
        The Tensor we're contracting.
    symbol: str
        The string of the thing the Tensor is contracted with. Usually, this is an integral.

    Output
    ------
    Tensor
        The new Tensor
    """
    new_string_list = [tensor.external_indices[::-1]] + tensor.string_list
    denominator = factorial(len(new_string_list[0][0])) ** 2 # 1/(n!)^2
    numerator = compute_central_weight(tensor.external_indices) # Prefactor due to choosing a representative ordering: top/bottom/left/right...
    # All antisymmetrized indices are external, and contracted with something else antisymmetric. They simplify to a weight.
    for group in tensor.antisymmetrizers:
        numerator *= multinomial(list(map(len, group)))
    return Tensor(new_string_list, tensor.weight * Fraction(numerator, denominator), [], set(), [symbol] + tensor.integral_tensors)

### Tensor Aux.
def compute_central_weight(cumulant):
    """ Compute the "cumulant weight" associated with the number of ways to produce this cumulant block.
        This accounts for the ways to permute occupied and virtual indices within the top and bottom rows,
        and also the fact that we don't count the hermitian adjoint, if the adjoint is distinct.

    Input
    -----
    cumulant: list of str
        Each lsit represents a row, top or bottom

    Output
    ------
    int
    """
    top_occ_count = sum([i.lower() in occupied_contractable for i in cumulant[0]])
    top_vir_count = sum([i.lower() in virtual_contractable for i in cumulant[0]])
    bot_occ_count = sum([i.lower() in occupied_contractable for i in cumulant[1]])
    bot_vir_count = sum([i.lower() in virtual_contractable for i in cumulant[1]])
    # // isn't for throwing away the (0) remainder, but to force these numbers to be ints.
    top_weight = factorial(top_occ_count + top_vir_count) // factorial(top_occ_count) // factorial(top_vir_count)
    bot_weight = factorial(bot_occ_count + bot_vir_count) // factorial(bot_occ_count) // factorial(bot_vir_count)
    # The * 2 accounts for only
    weight = top_weight * bot_weight * (2 if top_occ_count != bot_occ_count or top_vir_count != bot_vir_count else 1)
    return int(weight)

### General Purpose

def find_parity(list1, list2):
    """
    Input
    -----
    list1, list 2: list

    Output
    ------
    int
        1 or -1, the parity of the permutaiton to move one list into the other.
    """
    if list1 is None or list2 is None:
        return 1
    num_flips = 0
    permutation = [list1.index(i) for i in list2]
    num_elts = len(permutation)
    for i in range(num_elts):
        index_of_i = permutation.index(i)
        num_flips += index_of_i
        permutation.pop(index_of_i)
    return (-1) ** num_flips

def sort_func(x):
    """Function used to sort indices. 

    Input
    -----
    x: str

    Output
    ------
    int
    """
    master_list = occupied_contractable_distinguished + occupied_contractable + virtual_contractable_distinguished + virtual_contractable
    try:
        return master_list.index(x)
    except ValueError:
        raise Exception("Unexpected index {}.".format(x))

def get_space_string(operator):
    """Given an operator, return the string that specifies its orbital spaces.
       Used to get the name of the variable for code generation.

    Input
    -----
    operator: [str, str]

    Output
    ------
    return str"""
    space_string = ""
    search_string = operator[0] + operator[1]
    for char in search_string:
        if char.lower() in occupied_contractable:
            space_string += "o"
        elif char.lower() in virtual_contractable:
            space_string += "v"
        elif char != " ":
            raise Exception("fDetected non-contractable or space character in {search_string}.")
    return space_string

def multinomial(lst):
    """https://stackoverflow.com/a/46378809"""
    res, i = 1, 1
    for a in lst:
        for j in range(1,a+1):
            res *= i
            res //= j
            i += 1
    return res

### TENSOR CLASS

class Tensor:

    def __init__(self, string_list, weight, indices, asym, integral_tensors=[]):
        # The signed prefactor
        self.weight = weight
        # The string list represents the contraction scheme.
        # Each list element is an operator. Each operator is a list - first the upper indices, then the lower.
        self.string_list = string_list
        self.external_indices = indices
        # Let's say you have T3 and want to antisymmetrize ijk, abc where ij and ab are already antisymmetric.
        # Your top level element collects ijk, abc. Your next level represents a group like ij, and your next level are the individual elements.
        # So three nested lists.
        self.antisymmetrizers = asym
        self.integral_tensors = integral_tensors
        self.claimed_symbols = set()
        for amplitude in string_list:
            self.claimed_symbols.update(amplitude[0])
            self.claimed_symbols.update(amplitude[1])

    def rank(self):
        return len(self.external_indices[0])

    def print_code(self, variable):
        ws = " " * 4
        string = ws + "temp = "
        if self.weight != Fraction(1, 1):
            string += "{} * ".format(self.weight)
        string += "einsum(\""
        flipped_string_list = list(map(tensor_flip, self.string_list))
        flipped_external = tensor_flip(self.external_indices)
        string += ", ".join(map(simple_to_string, flipped_string_list)) + " -> " + simple_to_string(flipped_external) + "\", "
        tensor_names = []
        for i, simple_tensor in enumerate(flipped_string_list):
            try:
                tensor_name = "i[\"" + self.integral_tensors[i] + "_" + get_space_string(simple_tensor) + "\"]"
            except IndexError:
                tensor_name = "i[\"t{}\"]".format(len(simple_tensor[0]))
            tensor_names.append(tensor_name)
        string += ", ".join(tensor_names) + ")\n" + ws + variable + " += "
        if self.antisymmetrizers:
            string += "mla.antisymmetrize_axes_plus(temp"
            group_strings = []
            externals = []
            # Create a list that functions as a hash from symbol to index in einsum
            for row in flipped_external:
                externals += row
            for asym_group in self.antisymmetrizers:
                block_list = []
                for block in asym_group:
                    block_list.append("(" + ", ".join([str(externals.index(symb)) for symb in block]) + ",)")
                string += ", (" + ", ".join(block_list) + ")"
            string+= ")"
        else:
            string += "temp"
        print(string)

def product_rule(tensors):
    """Given a list of Tensors, return the tensors that result upon differentiating with respect to the amplitudes.

    Input
    -----
    tensors: list of Tensor

    Output
    ------
    list of Tensor
    """
    differentiated_tensors = []
    for tensor in tensors:
        for i in range(len(tensor.integral_tensors), len(tensor.string_list)):
            string_list = deepcopy(tensor.string_list)
            external = string_list.pop(i)
            weight = tensor.weight
            antisymmetrizers = deepcopy(tensor.antisymmetrizers)
            # Now let's handle the antisymmetrizers.
            for j, row in enumerate(external):
                asym_chars = frozenset(row)
                asym_row = []
                for term in string_list:
                    new_found_chars = frozenset(filter(lambda x: x in term[0] or x in term[1], asym_chars))
                    if new_found_chars:
                        weight *= factorial(len(new_found_chars))
                        asym_row.append(new_found_chars)
                if len(asym_row) > 1:
                    antisymmetrizers.add(frozenset(asym_row))

            differentiated_tensors.append(Tensor(string_list, weight, external, antisymmetrizers, tensor.integral_tensors))
    return differentiated_tensors

def seek_equivalents(tensors):
    """Given a list of tensors, return a list of only the unique tensors. "Duplicate" tensors are added together.
    TODO: Seems to also be responsible for canonicalizing. These are... very mixed concerns.
    Input
    -----
    tensors: iterable of Tensors

    Output
    ------
    list of Tensor"""
    if not tensors:
        return []
    # Assume all tensors are of the same degree. All terms except the first len(integral_tensors) can be permuted.
    fixed_perm = tuple(range(len(tensors[0].integral_tensors)))
    variable_terms = tuple(range(len(tensors[0].integral_tensors), len(tensors[0].string_list)))
    new_tensors = []
    permutations = list(itertools.permutations(variable_terms))
    flips = list(itertools.chain.from_iterable(itertools.combinations(fixed_perm, r) for r in range(len(fixed_perm) + 1)))
    num_amplitudes = len(tensors[0].string_list)
    for tensor in tensors:
        assert len(tensor.string_list) == num_amplitudes
        for permutation, flip in itertools.product(permutations, flips):
            permutation = fixed_perm + permutation
            new_tensor = permute_canonicalize_tensor(permutation, tensor, flip)
            for candidate_tensor in new_tensors:
                if new_tensor.string_list == candidate_tensor.string_list and new_tensor.external_indices == candidate_tensor.external_indices and new_tensor.antisymmetrizers == candidate_tensor.antisymmetrizers:
                    candidate_tensor.weight += new_tensor.weight
                    break
            else:
                # We didn't find a match. Try another permutation.
                continue
            # We found a match. Stop trying permutations.
            break
        else:
            new_tensors.append(new_tensor)
    return list(filter(lambda x: x.weight, new_tensors)) 

def permute_canonicalize_tensor(permutation, tensor, flip=[]):
    """Given a Tensor, permute the amplitudes in it, and transform it to canonical form.

    Input
    -----
    permutation: tuple of int
        A tuple of integers specifying the permutation

    tensor: Tensor

    Output
    ------
    Tensor
    The new tensor after the permutation is applied.
    """
    # Construct substitution dictionary.
    symbol_hash = OrderedDict()
    for future_idx, current_idx in enumerate(permutation):
        for row in tensor.string_list[current_idx]:
            for symbol in row:
                if symbol not in symbol_hash: symbol_hash[symbol] = []
                symbol_hash[symbol].append(future_idx)
    sorted_symbols = sorted(symbol_hash, key = lambda x: (len(symbol_hash[x]), symbol_hash[x]))
    flip_dict = dict()
    for symbol in sorted_symbols:
        flip_dict[symbol] = get_new_symbol(flip_dict.values(), symbol, len(symbol_hash[symbol]) == 1)
    string_list = [(x[::-1] if i in flip else x) for i, x in enumerate(tensor.string_list)]

    # Perform substitution of indices in the stringlist, externals, and antisymmetrizers.
    subbed_stringlist = swap_symbols(string_list, flip_dict)
    new_externals = swap_symbols(tensor.external_indices, flip_dict)
    new_antisymmetrizers = swap_symbols(tensor.antisymmetrizers, flip_dict)

    # Next, parity and sort the symbols.
    weight = tensor.weight
    for amplitude in subbed_stringlist:
        for i, row in enumerate(amplitude):
            sorted_row = list(sorted(row, key = sort_func))
            weight *= find_parity(row, sorted_row)
            amplitude[i] = sorted_row
    for i, row in enumerate(new_externals):
        sorted_row = list(sorted(row, key = sort_func))
        weight *= find_parity(row, sorted_row)
        new_externals[i] = sorted_row

    # Permute the string list.
    new_stringlist = [subbed_stringlist[i] for i in permutation] 
    return Tensor(new_stringlist, weight, new_externals, new_antisymmetrizers, tensor.integral_tensors[:])

def get_new_symbol(claimed_symbols, char, external):
    """ Return a symbol of the same occupied/virtual and external/internal type, not in the list of claimed_symbols.
    Input
    -----
    symbol_updater: iterable of str
        The previously used symbols
    char: str
        What's the character we're updating?
    special: bool
        Is this a symbol for an external index?

    Output
    ------
    str
    The new symbol.
    """
    if char.lower() in occupied_contractable:
        symbol_list = occupied_contractable_distinguished if external else occupied_contractable
    elif char.lower() in virtual_contractable:
        symbol_list = virtual_contractable_distinguished if external else virtual_contractable
    else:
        raise Exception(f"Symbol {char} isn't recofnized as occupied or virtual.")
    for symbol in symbol_list:
        if symbol not in claimed_symbols:
            return symbol
    else:
        raise Exception(f"We're out of symbols.")

### Begin Called Code

def exponential_similarity_transform(SQ, stationarity, compute_class):
    """
    Input
    -----
    SQ: [Counter, Counter]
        Represents a second quantized operator of generic indices. Each Counter should store only generic
        indices. The first represents the top row, and the second the bottom row.
    stationarity: {"variational", "projective"}
        Are we using projective or variational stationarity conditions?
    compute_class: func
        A function to take a completely contracted diagram and return a string classifying it

    Output
    ------
    list of dict of list of Tensor 
        The outer list is the commutator level. Dict is the class.
    """

    # Variable Initialization
    print("Operator Block: " + str(SQ))
    diagram_rank = SQ.rank()
    selfadjoint = SQ.selfadjoint()
    starting_diagrams = [Diagram([SQ])]
    returns = []

    # All diagrams needed for stationarity conditions. The diagrams you need must be either pure excitation/de-excitation
    # or fully contracted. Variational methods only need fully contracted diagrams. Projective methods only need diagrams
    # that aren't fully contracted, but can fully contract with a single excitation/de-excitation more.
    # By including negative rank operators, we account for excluding the hermitian conjugate of the base operator.
    stationarity_ranks = {0}
    if stationarity == "projective":
        stationarity_ranks.update(cluster_ranks)
        if not selfadjoint:
            stationarity_ranks.update(-1 * i for i in cluster_ranks)

    for commutator_number in range(1, max_commutator+1):

        print(f"Commutator: {commutator_number}")

        # The simplifications are commutator number dependent.
        simplifications = dict()
        if stationarity == "variational" and commutator_number == max_commutator:
            simplifications["fc_only"] = True

        open_diagrams = []
        stationarity_diagrams = {}

        # The expensive step is below.
        new_diagrams = commutator.commutator_with(starting_diagrams, simplifications, cluster_ranks)
        for diagram in new_diagrams:
            rank = diagram.excitation_rank()
            # Have we found a diagram that contributes to stationarity conditions?
            if rank in stationarity_ranks:
                diagram_class = compute_class(diagram)
                if diagram_class not in stationarity_diagrams: stationarity_diagrams[diagram_class] = []
                stationarity_diagrams[diagram_class].append(diagram)
            # Is the diagram open to further contractions? (Rank not defined or a nonzero integer.)
            if rank is not 0:
                open_diagrams.append(diagram)

        unique_counter = {diagram_class: to_multiplicity(diagrams) for diagram_class, diagrams in stationarity_diagrams.items()}
        returns.append({diagram_class: [] for diagram_class in unique_counter})

        for diagram_class, diagrams in unique_counter.items():
            tensors = [tensor_from_counter(diagram, stationarity, diagram_rank, selfadjoint, diagram_class) for diagram in diagrams]
            tensors_to_differentiate = []
            for new_tensor in tensors:
                if stationarity == "projective":
                    varname = "energy" if diagram_class == "0" else "i[\"r{}\"]".format(diagram_class)
                else:
                    prefix = "c" if diagram_class == "Connected" and diagram_rank == 2 else "rdm"
                    new_ext = tensor_flip(new_tensor.external_indices)
                    varname = "i[\"{}_{}\"]".format(prefix, get_space_string(new_ext))
                    tensors_to_differentiate.append(new_tensor)
                new_tensor.print_code(varname)
                returns[-1][diagram_class].append(new_tensor)
            # Differentiate.
            if tensors_to_differentiate:
                rdm_to_en_deriv(tensors_to_differentiate, "g" if diagram_rank == 2 else "f")

        # Use diagrams from n commutators to get those for n+1 commutators
        starting_diagrams = open_diagrams
    print("")
    return returns

def tensor_from_counter(diagram, stationarity, diagram_rank, selfadjoint, diagram_class):
    perm_weight = get_perm_weight(diagram)
    expanded_operators = set([0]) if stationarity == "variational" else set()
    possibility_string = make_string_list(diagram, expanded_operators)
    sign = determine_sign(possibility_string)
    externals, antisym_elts = external_antisym(possibility_string, stationarity)
    asym_weight, antisymmetrizers = string_list_antisym(possibility_string, antisym_elts)
    d_weight = Fraction(diagram.prefactor * asym_weight * sign, perm_weight)
    if stationarity == "projective":
        if diagram_rank == 1:
            integrals = ["f"]
        else:
            integrals = ["g"]
        # In energy diagrams, account for not including the hermitian conjugate.
        if diagram_class == "0" and not selfadjoint:
            d_weight *= Fraction(2, 1)
    else:
        integrals = []
    # Insist on occupied up. If i = 0 and stationarity is projective, the first term is an integral. You can't canonicalize that.
    possibility_string = [(operator[::-1] if (i or stationarity == "variational") and operator[1][0].lower() in occupied_contractable else operator) for i, operator in enumerate(possibility_string)]
    return Tensor(possibility_string, d_weight, externals, antisymmetrizers, integrals)

def substitution(parents, children):
    new_tensors = []
    for parent in parents:
        for child, idx in itertools.product(children, range(len(parent.string_list))):
            # Step 1. Change the indices of the child to those it will have upon insertion
            amplitude = deepcopy(parent.string_list[idx])
            reserved_symbols = parent.claimed_symbols.copy()
            flip_dict = dict()
            for i, row in enumerate(child.external_indices[::-1]):
                for j, symbol in enumerate(row):
                    flip_dict[symbol] = amplitude[i][j]

            for symbol in child.claimed_symbols:
                if symbol in flip_dict: continue
                new_symbol = get_new_symbol(reserved_symbols, symbol, False)
                flip_dict[symbol] = new_symbol
                reserved_symbols.add(new_symbol)
            new3 = swap_symbols(child.string_list, flip_dict)

            # Step 2. The antisymmetrizers.
            to_expand = []
            child_asym = swap_symbols(child.antisymmetrizers, flip_dict)
            new_asym = set()
            for permutation_group in child_asym:
                all_external = all(map(lambda subgroup: ''.join(subgroup).isupper(), permutation_group))
                if not all_external:
                    # There is an internal index in here. Better expand this...
                    to_expand.append(permutation_group)
                else:
                    new_asym.add(permutation_group)

            temp_tensors = [Tensor(new3, child.weight, amplitude, new_asym)]
            for row in to_expand:
                expanded_tensors = []
                for tensor in temp_tensors:
                    expanded_tensors += expand_antisymmetrizer_row(tensor, row)
                temp_tensors = expanded_tensors

            for tensor in temp_tensors:
                weight = tensor.weight * parent.weight * -1
                new_stringlist = parent.string_list[:idx] + tensor.string_list + parent.string_list[idx+1:]
                antisymmetrizers = deepcopy(parent.antisymmetrizers)
                new_tensors.append(Tensor(new_stringlist, weight, parent.external_indices, antisymmetrizers.union(new_asym)))

    return new_tensors

def rdm_to_en_deriv(tensors, symbol):
    """ Given RDM tensors, print out the tensors for the energy derivatives, assuming a simple product rule.

    Input
    -----
    tensors: list of Tensor
        The Tensor objects to differentiate.
    symbol:
        The symbol of the non-amplitude coefficient. Usually an integral.

    Output
    ------
    None
    """
    energy_tensors = [full_contract(tensor, symbol) for tensor in tensors]
    derivative_tensors = product_rule(energy_tensors)
    derivative_tensors = seek_equivalents(derivative_tensors)
    for tensor in derivative_tensors:
        tensor.print_code("i[\"r{}\"]".format(tensor.rank()))

def d_terms(data):
    """ Compute the d (2-RDM cumulant partial trace) terms of DCT.

    Input
    -----
    data: dict
        A structured dictionary. Key one (e.g., "oo") specifies which block
        is left after partial trace. Key two (e.g., "v") specifies which block
        to partial trace over. The value of that is a list of tensors.

    Output
    ------
    None
    """
    for d_block, value in data.items():
        print("D block: {}".format(d_block))
        for i, (o_data, v_data) in enumerate(zip(value["o"], value["v"]), start=1):
            print("{} commutator terms".format(i))
            o_data = compute_d(o_data, occupied_contractable)
            v_data = compute_d(v_data, virtual_contractable)
            d_tensor = seek_equivalents(o_data + v_data)
            d_tensor = [tensor for tensor in d_tensor if tensor.weight]
            print("D matrix")
            for tensor in d_tensor:
                tensor.print_code(f"i[\"d_{d_block}\"]")
            print("1RDM energy derivative")
            rdm_to_en_deriv(d_tensor, "ft")

def compute_d(tensors, contractable):
    """ Partial trace the input tensors over the specified indices. Used to compute the d terms of OUDCT.

    Input
    -----
    tensors: list of Tensor
        The tensors to partial trace.
    contractable: list of str
        The one-character symbols which can be partial-traced.
        Used to control whether occupied or virtual symbols are partial traced over.
        Depending on the block, you can tell which.

    Returns
    -------
    list of Tensor
        The d tensors"""
    tensors = expand_antisymmetrizers(tensors[:])
    tensors = [partial_trace(tensor, contractable) for tensor in tensors]
    return tensors

def partial_trace(tensor, contractable_symbols):
    """ Partial trace the input tensors over the specified indices, assuming no antisymmetrizers.

    Input
    -----
    tensors: list of Tensor
        The tensors to partial trace. Assumiing they have no antisymmetrizers.
    contractable: list of str
        The one-character symbols which can be partial-traced.
        Used to control whether occupied or virtual symbols are partial traced over.
        Depending on the block, you can tell which.

    Returns
    -------
    Tensor
    """
    string_list = deepcopy(tensor.string_list) # Don't assign to actual string_list
    external_indices = deepcopy(tensor.external_indices) # Don't pop from actual row...
    weight = tensor.weight
    traced_indices = []
    new_external = []
    if tensor.antisymmetrizers:
        raise Exception("You must expand antisymmetrizers before you partial trace.")
    for row in external_indices:
        for i, char in enumerate(row):
            if char.lower() in contractable_symbols:
                traced_indices.append(row.pop(i))
                new_external.append(row)
                weight *= (-1) ** i
                break
        else:
            raise Exception("A row of external indices doesn't contain a proper symbol.")
    string_list = swap_symbols(string_list, {traced_indices[1]: traced_indices[0]})
    return Tensor(string_list, weight, new_external, set(), tensor.integral_tensors[:])

def expand_antisymmetrizers(unexpanded_tensors):
    """ Given a list of Tensor, return a list of Tensor where all antisymmetrizers have
        explicitly been expanded to produce other tensors.

    Input
    -----
    unexpanded_tensors: list of Tensor
        The function modified this.

    Output
    ------
    list of Tensor
    """
    fully_expanded_tensors = []
    while unexpanded_tensors:
        tensor = unexpanded_tensors.pop(0)
        expanded_tensors = expand_antisymmetrizer_row(tensor)
        for tensor in expanded_tensors:
            tensor_list = unexpanded_tensors if tensor.antisymmetrizers else fully_expanded_tensors
            tensor_list.append(tensor)
    return fully_expanded_tensors

def expand_antisymmetrizer_row(tensor, expanding_asym = False):
    """
    Given a tensor, expand a single row of antiysmmetrizers into multiple tensors.

    Input
    ----
    tensor: Tensor

    Output
    ------
    list of Tensor
    """
    # Strategy: Once we have a valid permutation, applying it is just a symbol_swap to the stringlist
    # followed by a parity factor. The real challenge is generating the permutations.
    # We convert from an original position to a string SymPy can work with. SymPy gives us the
    # permutations in terms of those strings, and then we have to convert back to our letters.

    if not tensor.antisymmetrizers and not expanding_asym:
        return [tensor]
    new_tensors = []
    to_expand_asym = deepcopy(tensor.antisymmetrizers)
    if not expanding_asym: expanding_asym = to_expand_asym.pop()

    integer_mask = []
    permutation_blocks = []
    old_ordering = []
    for i, quotient in enumerate(expanding_asym):
        ordered_quotient = list(quotient)
        integer_mask += [i] * len(quotient) # Given to Sympy for perm. generation
        permutation_blocks.append(ordered_quotient)
        old_ordering += "".join(ordered_quotient)

    for permutation in sympy.utilities.iterables.multiset_permutations(integer_mask):
        next_index_per_block = [0] * len(permutation_blocks)
        flip_dict = {}
        new_ordering = []
        for block, old_symbol in zip(permutation, old_ordering):
            next_index = next_index_per_block[block]
            new_symbol = permutation_blocks[block][next_index]
            flip_dict[old_symbol] = new_symbol
            new_ordering.append(new_symbol)
            next_index_per_block[block] += 1
        new_string_list = swap_symbols(tensor.string_list, flip_dict)
        weight = tensor.weight * find_parity(old_ordering, new_ordering)
        # Okay, next I'll need the asym
        new_tensors.append(Tensor(new_string_list, weight, deepcopy(tensor.external_indices), to_expand_asym, deepcopy(tensor.integral_tensors)))
    return new_tensors

def swap_symbols(groups, flip_dict):
    if isinstance(groups, frozenset):
        return frozenset(swap_symbols(x, flip_dict) for x in groups)
    if isinstance(groups, set):
        return set(swap_symbols(x, flip_dict) for x in groups)
    elif isinstance(groups, list):
        return list(swap_symbols(x, flip_dict) for x in groups)
    elif isinstance(groups, str):
        return flip_dict.get(groups, groups)
    else:
        raise Exception
