import torch
from torch.autograd import Variable
from nltk import CFG, Nonterminal

# grammar = """
# S -> S '+' T
# S -> S '*' T
# S -> S '/' T
# S -> S '-' T
# S -> T
# T -> T '^' T
# T -> '-' T    
# T -> '(' S ')'
# T -> 'sin(' S ')'
# T -> 'cos(' S ')'
# T -> 'u'
# T -> 'u^2'
# T -> 'u^3'
# T -> 'x'
# T -> 't'
# T -> 'x^2'
# T -> 't^2'
# T -> 'x^3'
# T -> 't^3'
# T -> '(-x)'
# T -> '(-t)'
# T -> 'd/dx(u)'
# T -> 'd/dt(u)'
# T -> 'd2/dt2(u)'
# T -> 'd2/dx2(u)'
# T -> 'd3/dx3(u)'
# T -> 'u*d/dx(u)'
# T -> 'd/dx(u^2)'
# T -> DIGITS '.' DIGITS
# T -> DIGITS
# DIGITS -> DIGIT DIGITS | DIGIT
# DIGIT -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
# Nothing -> None
# """


grammar = """
S -> S '+' T
S -> S '*' T
S -> S '/' T
S -> S '-' T
S -> S '^' T
S -> T
T -> '-' T    
T -> '(' S ')'
T -> 'sin(' S ')'
T -> 'cos(' S ')'
T -> 'u'
T -> 'u^2'
T -> 'u^3'
T -> 'x'
T -> 't'
T -> 'x^2'
T -> 't^2'
T -> 'x^3'
T -> 't^3'
T -> 'd/dx(u)'
T -> 'd/dt(u)'
T -> 'd2/dt2(u)'
T -> 'd2/dx2(u)'
T -> 'd3/dx3(u)'
T -> 'u*d/dx(u)'
T -> 'd/dx(u^2)'
T -> T '.' T
T -> T T 
T -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
Nothing -> None
"""

GCFG = CFG.fromstring(grammar)

# S, T, N, D, DS = Nonterminal('S'), Nonterminal('T'), Nonterminal('NUMBER'), Nonterminal('DIGIT'), Nonterminal('DIGITS')
S, T = Nonterminal('S'), Nonterminal('T')

def get_mask(nonterminal, grammar, as_variable=False):
    if isinstance(nonterminal, Nonterminal):
        mask = [rule.lhs() == nonterminal for rule in grammar.productions()]
        mask = Variable(torch.FloatTensor(mask)) if as_variable else mask
        return mask
    else:
        raise ValueError('Input must be instance of nltk.Nonterminal')

if __name__ == '__main__':
    # Usage:
    GCFG = nltk.CFG.fromstring(grammar)
    print(get_mask(T))
