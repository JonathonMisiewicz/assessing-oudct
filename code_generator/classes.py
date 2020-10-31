from collections import Counter
from fractions import Fraction
from math import factorial

import itertools

class Operator:
    """ Represents a single operator. """

    def __init__(self, upper: Counter, lower: Counter):
        self.upper = upper
        self.lower = lower

    def __str__(self):
        return " ".join(row_stringify(row) for row in [self.upper, self.lower])

    def __eq__(self, other):
        return self.upper == other.upper and self.lower == other.lower

    def rank(self) -> int:
        return sum(self.upper.values())

    def selfadjoint(self) -> bool:
        return self.upper == self.lower

    def contracted(self) -> bool:
        return any(not set(["o", "v"]).issuperset(set(x)) for x in (self.upper, self.lower))

def row_stringify(row):
    """ Given a Counter representation of a row of a second quantized operator,
        convert it to a string."""
    return "".join([letter * mult for (letter, mult) in row.items()])

class Diagram:
    """ Represents a diagram. """

    def __init__(self, operators: [Operator]):
        self.operators = operators
        self.prefactor = Fraction(1, factorial(len(operators) - 1))
        self.permutations = None
    
    def __len__(self):
        return len(self.operators)

    def __eq__(self, other):
        return self.operators == other.operators

    def rank(self):
        """ Input a diagram, output a tuple of the number of occ and vir creation and
        annihilation operators. """
        u_o, u_v, l_o, l_v = 0, 0, 0, 0
        for operator in self.operators:
            upper = operator.upper
            lower = operator.lower
            u_o += upper["o"]
            u_v += upper["v"]
            l_o += lower["o"]
            l_v += lower["v"]
        return (u_o, u_v, l_o, l_v)

    def excitation_rank(self):
        """ Input a diagram, output its excitation rank. Negative for deexcitation. None if not pure excite/de-excite"""
        u_o, u_v, l_o, l_v = self.rank()
        if not u_o and not l_v and u_v == l_o:
            return u_v
        if not u_v and not l_o and u_o == l_v:
            return -u_o
        return None

    def equiv(self, other) -> bool:
        if self.permutations is None:
            self.permutations = [self.permute((0,) + tuple(perm)) for perm in itertools.permutations(range(1, len(self)))]

        return other in self.permutations

    def permute(self, perm):
        """ Return the diagram obtained by the specified operator permutation. """
        operators = []
        for i in perm:
            operator = self.operators[i]
            new_rows = []
            for row in (operator.upper, operator.lower):
                new_row = Counter()
                for symbol, count in row.items():
                    letter, old_partner = symbol[0], symbol[1:]
                    new_partner = str(perm.index(int(old_partner))) if old_partner else ""
                    new_row[letter + new_partner] = count
                new_rows.append(new_row)
            operators.append(Operator(*new_rows))

        return Diagram(operators)

