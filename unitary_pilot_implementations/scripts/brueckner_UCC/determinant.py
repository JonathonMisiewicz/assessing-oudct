import numpy as np

class Determinant:
    def __init__(self, a, b): 
        """ Initializes a new Determinant.
        
        Input parameters a and b are the string representation of 
        the occupation vector:
        
        a = '11100'
        b = '11010'
        
        These entries are converted to bits.
        """
        self.alpha = int(a[::-1], 2)
        self.beta = int(b[::-1], 2)
        self.order = len(a)
        if len(a) != len(b):
            raise Exception(f"You have {len(a)} alpha orbitals and {len(b)} beta orbitals? Those aren't the same number.")
    
    def __str__(self):
        out = '\u03B1: ' + np.binary_repr(self.alpha, width=abs(self.order))[::-1]
        out += ' \u03B2: ' + np.binary_repr(self.beta, width=abs(self.order))[::-1]
        return out 

    def __eq__(self, other):
        """ Tests equality of the alpha and beta strings with those of other. 
        
        Return True or False accordingly.
        """
        return self.alpha == self.beta

    def __sub__(self, other):
        """ Subtracting two determinants yields the number of different orbitals.

        Single excitation -> 2 different orbitals.

        Returns the number different orbitals.

        Try involving bitwise operators to determine this.

        Example:

            Determinant('11100', '11100') - Determinant('11010', '11100') = 2
        """
        different_alpha = self.alpha ^ other.alpha
        different_beta = self.beta ^ other.beta
        return sum(sum(test_bit(y, x) for x in range(self.order)) for y in (different_alpha, different_beta))

    def alpha_list(self):
        """Returns a list of the orbital occupations.

        Example:

            Determinant('11100', '11010').alpha_list() -> [1, 1, 1, 0, 0]
        """
        return [int(self.alpha & 1 << bit != 0) for bit in range(self.order)]

    def beta_list(self):
        """Returns a list of the orbital occupations.

        Example:

            Determinant('11100', '11010').alpha_list() -> [1, 1, 0, 1, 0]
        """
        return [int(self.beta & 1 << bit != 0) for bit in range(self.order)]

    def phase(self, other, twobody=True):
        """Returns the phase between self and the other determinant.
        
        When twobody is set, operators that differ by more than two orbitals are zero.
        This is useful for full configuration interaction.

        Examples:
        
            Reference             Other                    Phase
            α: 1111100 β: 1111100 α: 1111100 β: 1111100 -> 0
            α: 1111100 β: 1111100 α: 1111100 β: 0011111 -> 1.0
            α: 1111100 β: 1111100 α: 1111100 β: 1011011 -> 1.0
            α: 1111100 β: 1111100 α: 1111100 β: 1001111 -> 1.0
            α: 1111100 β: 1111100 α: 1111100 β: 1110110 -> -1.0
            α: 1111100 β: 1111100 α: 1111100 β: 1110011 -> 1.0
            α: 1111100 β: 1111100 α: 1111100 β: 0111110 -> 1.0
            α: 1111100 β: 1111100 α: 1111100 β: 0111011 -> -1.0
            α: 1111100 β: 1111100 α: 1111100 β: 0101111 -> -1.0
            α: 1111100 β: 1111100 α: 1111100 β: 1101101 -> 1.0
            α: 1111100 β: 1111100 α: 0011111 β: 0111101 -> 0
        """
        return self.phase_alpha_beta(other.alpha, other.beta, twobody)

    def phase_alpha_beta(self, alpha, beta, twobody=True):
        # Suppose there was some excitation operator that would take you from one to the other.
        # In KM notation, orbitals appear in left-to-right order. What addl. phase factor do you need?
        alpha_quasi = self.alpha ^ alpha
        beta_quasi = self.beta ^ beta
        alpha_vac = self.alpha & alpha
        beta_vac = self.beta & beta
        quasi = 0
        phase = 1
        for quasi_str, vac_string in [(alpha_quasi, alpha_vac), (beta_quasi, beta_vac)]:
            for i in range(self.order):
                if test_bit(quasi_str, i):
                    quasi += 1
                elif test_bit(vac_string, i):
                    # To determine the phase factor, all you need is the PARITY of the number of particles you're moving!
                    phase *= (-1) ** quasi
        return phase if (not twobody or quasi in [2, 4]) else 0

    def set_subtraction(self, other):
        """Returns the bits that are occupied in self but not in other.
        
        Again, try to use bitwise operators to perform this.
        
        Example:
        
            d1 = Determinant('1111100', '1111100')
            d2 = Determinant('1110011', '1110011')
            
            d1.set_subtraction(d2) -> (0b11000, 0b11000)
        """
        return (self.alpha & ~other.alpha, self.beta & ~other.beta)

    def set_subtraction_list(self, other):
        """Returns the bits that are occupied in self but not in other.
        
        Example:
        
            d1 = Determinant('1111100', '1111100')
            d2 = Determinant('1011011', '1011011')
            
            d1.set_subtraction(d2) -> ([1, 4], [1, 4])
        """
        alpha, beta = self.set_subtraction(other)
        return tuple(tuple(i for i in range(self.order) if test_bit(string, i)) for string in (alpha, beta))

def test_bit(string, pos):
    """ Is position pos occupied in bitstring string?"""
    return (string & 1 << pos) != 0
