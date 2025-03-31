import torch
from torch.autograd import Variable
from nltk import CFG, Nonterminal

grammar = """
# Top-level expression structure
S -> S '+' T
S -> S '-' T
S -> S '*' T
S -> S '/' T
S -> '-' T
S -> S '**' DIGIT
S -> S '**' DIGIT DIGIT
S -> T

# Variables and digits directly as terminals
T -> 'x' | 'y' | 't' | 'pi' | 'u'
T -> DIGIT | DIGIT DIGIT | DIGIT DIGIT DIGIT
T -> DIGIT '.' DIGIT | DIGIT '.' DIGIT DIGIT | DIGIT '.' DIGIT DIGIT DIGIT
T -> DIGIT DIGIT '.' DIGIT | DIGIT DIGIT '.' DIGIT DIGIT | DIGIT DIGIT '.' DIGIT DIGIT DIGIT
T -> DIGIT DIGIT DIGIT '.' DIGIT | DIGIT DIGIT DIGIT '.' DIGIT DIGIT | DIGIT DIGIT DIGIT '.' DIGIT DIGIT DIGIT
T -> '(' S ')'

# Common full patterns from the dataset
T -> 'dt(u)'
T -> 'dx(u)'
T -> 'dy(u)'
T -> 'dxx(u)'
T -> 'dyy(u)'
T -> 'dtt(u)'
T -> 'dxxxx(u)'
T -> 'dxxyy(u)'
T -> 'dyyyy(u)'
T -> 'sin(pi*x)'
T -> 'sin(pi*y)'
T -> 'cos(pi*x)'
T -> 'cos(pi*y)'
T -> 'sin(pi*(t-x))'
T -> 'sin(pi*(t-y))'
T -> 'sin(pi*(t+x))'
T -> 'cos(pi*t)'
T -> 'exp(-pi**2*t)'
T -> 'exp(-t)'
T -> 'tanh(t-x)'
T -> 'pi**2'
T -> 'pi**4'
T -> 'u**2'
T -> 'u**3'
T -> 'u*dx(u)'
T -> 'u-u**2'
T -> 'u-u**3'
T -> '(dxx(u)+dyy(u))'
T -> 'u*(1-u)'
T -> 'sqrt((x-0.5)**2+(y-0.5)**2)'

# Function application
T -> 'sin' '(' S ')'
T -> 'cos' '(' S ')'
T -> 'exp' '(' S ')'
T -> 'tanh' '(' S ')'
T -> 'sinh' '(' S ')'
T -> 'cosh' '(' S ')'
T -> 'sqrt' '(' S ')'

# Basic digit
DIGIT -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
Nothing -> 'None'
"""

GCFG = CFG.fromstring(grammar)
S, T, DIGIT = Nonterminal('S'), Nonterminal('T') , Nonterminal('DIGIT')

def get_mask(nonterminal, grammar, as_variable=False):
    if isinstance(nonterminal, Nonterminal):
        mask = [rule.lhs() == nonterminal for rule in grammar.productions()]
        mask = Variable(torch.FloatTensor(mask)) if as_variable else mask
        return mask
    else:
        raise ValueError('Input must be instance of nltk.Nonterminal')

if __name__ == '__main__':
    # Usage:
    GCFG = CFG.fromstring(grammar)
    print(get_mask(T))