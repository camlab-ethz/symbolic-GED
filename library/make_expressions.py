from __future__ import print_function
import nltk
import numpy as np
import h5py
import re
import csv
import os
import sys
import grammar_discovery as grammar_module

NCHARS = len(grammar_module.GCFG.productions())
def tokenize(s):
    """
    Tokenize an expression string, splitting numbers digit-by-digit and preserving '**'.
    
    Args:
        s (str): The expression string to tokenize
        
    Returns:
        list: A list of tokens that match our grammar's structure
    """
    # Replace special middle dot with standard multiplication
    s = s.replace('·', '*')
    
    # Define patterns to keep as single tokens
    derivative_patterns = [
        'dt(u)', 'dx(u)', 'dy(u)', 
        'dxx(u)', 'dyy(u)', 'dtt(u)', 
        'dxxxx(u)', 'dxxyy(u)', 'dyyyy(u)'
    ]
    
    common_patterns = [
        'sin(pi*x)', 'sin(pi*y)', 'cos(pi*x)', 'cos(pi*y)',
        'sin(pi*(t-x))', 'sin(pi*(t-y))', 'sin(pi*(t+x))', 'cos(pi*t)',
        'exp(-pi**2*t)', 'exp(-t)', 'tanh(t-x)',
        'pi**2', 'pi**4', 'u**2', 'u**3',
        'u*dx(u)', 'u-u**2', 'u-u**3',
        '(dxx(u)+dyy(u))', 'u*(1-u)',
        'sqrt((x-0.5)**2+(y-0.5)**2)'
    ]
    
    all_patterns = derivative_patterns + common_patterns
    temp_s = s
    
    # Step 1: Replace '**' with '__POWER__' and ensure separation
    temp_s = temp_s.replace('**', ' __POWER__ ')
    
    # Step 2: Replace patterns with placeholders
    for i, pattern in enumerate(all_patterns):
        pattern_placeholder = pattern.replace('**', '__POWER__')
        if pattern_placeholder in temp_s:
            temp_s = temp_s.replace(pattern_placeholder, f" PATTERN_{i} ")
    
    # Step 3: Add spaces around operators and parentheses
    temp_s = re.sub(r'([-+*/()])', r' \1 ', temp_s)
    
    # Step 4: Split into raw tokens
    raw_tokens = [token for token in temp_s.split() if token]
    
    # Step 5: Process tokens, restoring '**' and splitting numbers
    tokens = []
    for token in raw_tokens:
        if token.startswith('PATTERN_'):
            idx = int(token.split('_')[1])
            tokens.append(all_patterns[idx])
        elif token == '__POWER__':
            tokens.append('**')
        elif re.match(r'^-?[0-9]*\.?[0-9]+$', token):  # Stricter regex for numbers only
            if token.startswith('-'):
                tokens.append('-')
                number_part = token[1:]
            else:
                number_part = token
            tokens.extend(list(number_part))
        else:
            tokens.append(token)
    
    return tokens

def to_one_hot(expressions, max_len):
    assert type(expressions) == list
    prod_map = {prod: ix for ix, prod in enumerate(grammar_module.GCFG.productions())}
    tokenized = []
    for i, expr in enumerate(expressions):
        tokens = tokenize(expr)
        tokenized.append((i, expr, tokens))
        if i < 5:
            print(f"\nTokenized [{i+1}]: {expr}")
            print(f"Tokens: {tokens}")
    parser = nltk.ChartParser(grammar_module.GCFG)
    parse_trees = []
    failed_indices = []
    failed_expressions = []
    for i, expr, tokens in tokenized:
        try:
            parse_tree = next(parser.parse(tokens))
            parse_trees.append(parse_tree)
        except StopIteration:
            print(f"Parsing failed for expression #{i+1}: {expr}")
            print(f"Tokens: {tokens}")
            parse_trees.append(None)
            failed_indices.append(i)
            failed_expressions.append(expr)
        except Exception as e:
            print(f"Error parsing expression #{i+1}: {expr}")
            print(f"Error: {str(e)}")
            parse_trees.append(None)
            failed_indices.append(i)
            failed_expressions.append(expr)
    valid_trees = [tree for tree in parse_trees if tree is not None]
    print(f"\nSuccessfully parsed {len(valid_trees)} out of {len(expressions)} expressions")
    if not valid_trees:
        print("No valid parse trees found.")
        return np.zeros((0, max_len, NCHARS), dtype=np.float32), failed_indices, failed_expressions
    productions_seq = [tree.productions() for tree in valid_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    actual_max_len = max([len(idx) for idx in indices])
    print(f"Longest production sequence: {actual_max_len}")
    if actual_max_len > max_len:
        print(f"Warning: Some sequences exceed max_len ({max_len}). Consider increasing max_len to {actual_max_len}")
    one_hot = np.zeros((len(indices), max_len, NCHARS), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = min(len(indices[i]), max_len)
        one_hot[i][np.arange(num_productions), indices[i][:num_productions]] = 1.
        one_hot[i][np.arange(num_productions, max_len), -1] = 1.
    return one_hot, failed_indices, failed_expressions

def process_csv_file(csv_path, column='operator_L', output_file='expressions_LUF-125.h5', max_len=125):
    print(f"Processing CSV file: {csv_path}")
    print(f"Looking for column: {column}")
    print(f"Using max sequence length: {max_len}")
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} does not exist")
        return
    expressions = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if column in row and row[column]:
                    expressions.append(row[column])
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return
    print(f"Found {len(expressions)} expressions to process")
    print("\nSample of expressions:")
    for i, expr in enumerate(expressions[:5]):
        print(f"[{i+1}] {expr}")
    batch_size = 100
    OH = np.zeros((len(expressions), max_len, NCHARS), dtype=np.float32)
    all_failed_indices = []
    all_failed_expressions = []
    for i in range(0, len(expressions), batch_size):
        end_idx = min(i + batch_size, len(expressions))
        print(f'Processing batch: [{i}:{end_idx}]')
        batch = expressions[i:end_idx]
        onehot, failed_indices, failed_expressions = to_one_hot(batch, max_len)
        failed_indices = [i + idx for idx in failed_indices]
        all_failed_indices.extend(failed_indices)
        all_failed_expressions.extend(failed_expressions)
        batch_size_actual = onehot.shape[0]
        if batch_size_actual > 0:
            OH[i:i + batch_size_actual, :, :] = onehot
    print(f"\nSaving results to {output_file}")
    h5f = h5py.File(output_file, 'w')
    h5f.create_dataset('data', data=OH)
    h5f.attrs['max_len'] = max_len
    h5f.attrs['num_expressions'] = len(expressions)
    h5f.attrs['num_parsed'] = len(expressions) - len(all_failed_indices)
    h5f.attrs['source_column'] = column
    h5f.attrs['source_file'] = os.path.basename(csv_path)
    if all_failed_indices:
        h5f.create_dataset('failed_indices', data=np.array(all_failed_indices))
        dt = h5py.special_dtype(vlen=str)
        failed_dset = h5f.create_dataset('failed_expressions', (len(all_failed_expressions),), dtype=dt)
        for i, expr in enumerate(all_failed_expressions):
            failed_dset[i] = expr
    h5f.close()
    print(f"\nProcessing complete:")
    print(f"Total expressions: {len(expressions)}")
    print(f"Successfully parsed: {len(expressions) - len(all_failed_indices)} ({(len(expressions) - len(all_failed_indices))/len(expressions)*100:.1f}%)")
    print(f"Failed to parse: {len(all_failed_indices)} ({len(all_failed_indices)/len(expressions)*100:.1f}%)")
    print(f"One-hot encoded data saved to {output_file}")
    if all_failed_indices:
        error_file = output_file.replace('.h5', '_errors.txt')
        with open(error_file, 'w') as f:
            f.write(f"Failed to parse {len(all_failed_indices)} expressions:\n\n")
            for i, expr in enumerate(all_failed_expressions):
                f.write(f"[{all_failed_indices[i]+1}] {expr}\n")
        print(f"Error details saved to {error_file}")
    return len(expressions) - len(all_failed_indices), len(all_failed_indices)

if __name__ == "__main__":
    default_csv_path = "/cluster/work/math/camlab-data/symbolic-GED/datasets/test-fix-Derivative-4.csv"
    default_column = "operator_L"
    default_output = "expressions_LUF-125.h5"
    default_max_len = 125
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv_path
    column = sys.argv[2] if len(sys.argv) > 2 else default_column
    output_file = sys.argv[3] if len(sys.argv) > 3 else default_output
    max_len = default_max_len
    if len(sys.argv) > 4:
        try:
            max_len = int(sys.argv[4])
            if max_len <= 0:
                print(f"Invalid max_len: {max_len}. Using default value: {default_max_len}")
                max_len = default_max_len
        except ValueError:
            print(f"Invalid max_len: {sys.argv[4]}. Using default value: {default_max_len}")
    print(f"Nchars: {NCHARS}")
    process_csv_file(csv_path, column=column, output_file=output_file, max_len=max_len)