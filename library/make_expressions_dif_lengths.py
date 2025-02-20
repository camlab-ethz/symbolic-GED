from __future__ import print_function
import nltk
import grammar_pde as grammar_module  # Assuming grammar is another module
import numpy as np
import h5py
import re
import csv
import random
import re
import os
import logging
import psutil
import gc
logging.basicConfig(filename='encoded-125-idx.log', level=logging.INFO, format='%(asctime)s %(message)s')
# Read expressions from the first column of a CSV file
# with open('/cluster/scratch/ooikonomou/filtered_100k_file.csv', 'r') as f:
#     reader = csv.reader(f)
#     L = [row[0].strip() for row in reader]  # Extract the first column (PDEs)
# with open('/cluster/scratch/ooikonomou/filtered_100k_file_pde_triplets_no_const_over_dt_newer_no0_in_const20k_smaller.csv', 'r') as f:
with open('/cluster/scratch/ooikonomou/latest/filtered_cases_no-e-no-3decimals.csv', 'r') as f:
    reader = csv.reader(f)
    L = [row[0].strip() for row in reader]  # Extract the first column (PDEs)

# # Read expressions from file
# with open('/cluster/home/ooikonomou/LoDE/src/Discovery/PDEs/accelration_disc.txt', 'r') as f:
# # with open('/cluster/home/ooikonomou/LoDE/src/Discovery/test/pde2_simpler_new.txt', 'r') as f:
#     L = [line.strip() for line in f]

MAX_LEN = 80
NCHARS = len(grammar_module.GCFG.productions())
logging.info('Nchars', NCHARS)

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

def to_one_hot(expressions, start_index=0):
    """ Encode a list of expressions strings to one-hot vectors """
    assert type(expressions) == list
    prod_map = {prod: ix for ix, prod in enumerate(grammar_module.GCFG.productions())}
    tokens = list(map(tokenize, expressions))
    
    # logging.info(f"here are Tokens")
    # logging.info(f"Tokens: {tokens}")
    parser = nltk.ChartParser(grammar_module.GCFG)
    parse_trees = []
    skipped_indices = []
    skipped_expressions = []
    
    for idx, t in enumerate(tokens):
        try:
            # logging.info(f"checkpoint 1")
            # logging.info(f"step: {idx}")
            parse_tree = next(parser.parse(t))
            num_productions = len(parse_tree.productions())
            # logging.info(f"checkpoint 2")
            if num_productions <= MAX_LEN:
                # logging.info(f"checkpoint 3")
                parse_trees.append(parse_tree)
            else:
                # logging.info(f"checkpoint 4")
                # If the number of productions exceeds MAX_LEN, skip it
                skipped_indices.append(start_index + idx)
                skipped_expressions.append(expressions[idx])
        except StopIteration:
            # logging.info(f"checkpoint 5")
            logging.info(f"Parsing failed for token: {t}")
            skipped_indices.append(start_index + idx)
            skipped_expressions.append(expressions[idx])
    
    if not parse_trees:
        # logging.info(f"checkpoint 6")
        return None, skipped_indices, skipped_expressions
    # logging.info(f"checkpoint 7")
    productions_seq = [tree.productions() for tree in parse_trees]
    # logging.info(f"checkpoint 8")
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    # logging.info(f"checkpoint 9")
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    # logging.info(f"checkpoint 10")
    for i in range(len(indices)):

        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions), indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN), -1] = 1.
    
    return one_hot, skipped_indices, skipped_expressions


all_skipped_indices = []
all_skipped_expressions = []
OH = np.zeros((len(L) - len(all_skipped_indices), MAX_LEN, NCHARS))
valid_index = 0

for i in range(0, len(L), 100):
    logging.info('Processing: i=[' + str(i) + ':' + str(i + 100) + ']')
    onehot, skipped_indices, skipped_expressions = to_one_hot(L[i:i + 100], start_index=i)
    
    if onehot is not None:
        batch_size = onehot.shape[0]
        OH[valid_index:valid_index + batch_size, :, :] = onehot
        valid_index += batch_size

    # Store skipped indices and expressions
    all_skipped_indices.extend(skipped_indices)
    all_skipped_expressions.extend(skipped_expressions)
    # Clear variables
    del onehot, skipped_indices, skipped_expressions
    gc.collect()  # Force garbage collection to free memory

h5f = h5py.File('filtered_100_80_euler_sorted_1terminal-new-20-no0_in_const20k-2terminals-latest more-correct-120-no-e-no-3decimals.h5', 'w')
h5f.create_dataset('data', data=OH[:valid_index, :, :])  # Save only the valid part of OH
h5f.close()

# Save skipped indices and expressions for review
with open('skipped_expressions_new_160_correct_order_clustered_filtered_45characters_80_euler_sorted_1terminal-2-latest-more correct-120-noe-e-no-3decimals-80.txt', 'w') as f:
    for idx, expr in zip(all_skipped_indices, all_skipped_expressions):
        f.write(f"Index {idx}: {expr}\n")

print(f"Skipped {len(all_skipped_indices)} expressions. See skipped_expressions.txt for details.")
