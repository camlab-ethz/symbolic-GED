from __future__ import print_function
import nltk
import grammar_pde as grammar_module  # Assuming grammar is another module
import numpy as np
import h5py
import re



# Read expressions from file
with open('/cluster/home/ooikonomou/LoDE/src/Discovery/test/pde2_simpler_new.txt', 'r') as f:
    L = [line.strip() for line in f]

MAX_LEN = 125
NCHARS = len(grammar_module.GCFG.productions())
print('Nchars', NCHARS)

def tokenize(s):
    funcs = ['sin', 'cos']  # Reflecting only sin and cos as per your grammar

    # Add spaces around functions to isolate them before splitting
    for fn in funcs:
        s = s.replace(fn + '(', f' {fn}( ')
    
    # Adjust the regular expression to match various components
    s = re.sub(r'(\d+\.\d+|\d+|[-+*/()^]|sin\(|cos\(|u\^2|u\^3|u|x\^2|t\^2|x\^3|t\^3|x|t|d/dx\(u\)|d/dt\(u\)|d2/dt2\(u\)|d2/dx2\(u\)|d3/dx3\(u\)|u\*d/dx\(u\)|d/dx\(u\^2\))', r' \1 ', s)

    # Tokenize and split numbers into individual characters
    tokens = []
    for token in re.findall(r'\d+\.\d+|\d+|[-+*/()^]|sin\(|cos\(|u\^2|u\^3|u|x\^2|t\^2|x\^3|t\^3|x|t|d/dx\(u\)|d/dt\(u\)|d2/dt2\(u\)|d2/dx2\(u\)|d3/dx3\(u\)|u\*d/dx\(u\)|d/dx\(u\^2\)', s):
        if re.match(r'\d+\.\d+|\d+', token):  # Match full floating-point numbers and integers
            tokens.extend(list(token))  # Split into individual characters
        else:
            tokens.append(token)

    return tokens


def to_one_hot(expressions):
    """ Encode a list of expressions strings to one-hot vectors """
    assert type(expressions) == list
    prod_map = {prod: ix for ix, prod in enumerate(grammar_module.GCFG.productions())}
    tokens = list(map(tokenize, expressions))
    # print(f"Tokens: {tokens}")
    parser = nltk.ChartParser(grammar_module.GCFG)
    parse_trees = []
    
    for t in tokens:
        try:
            parse_tree = next(parser.parse(t))
            parse_trees.append(parse_tree)
        except StopIteration:
            print(f"Parsing failed for token: {t}")
    
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions), indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN), -1] = 1.
    
    return one_hot

OH = np.zeros((len(L), MAX_LEN, NCHARS))
for i in range(0, len(L), 100):
    print('Processing: i=[' + str(i) + ':' + str(i + 100) + ']')
    onehot = to_one_hot(L[i:i + 100])
    # Handle cases where the batch size is smaller than 100 due to end of list
    batch_size = onehot.shape[0]
    OH[i:i + batch_size, :, :] = onehot

h5f = h5py.File('expressions_LUF-125.h5', 'w')
h5f.create_dataset('data', data=OH)
h5f.close()
