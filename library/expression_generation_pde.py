from expressions_data_pde import save_expressions_to_file

def main():
    """Main function to generate and save algebraic expressions."""
    filename = 'expressions_test_new30k.txt'
    num_expressions = 30000 # Change this to the desired number of expressions
    case = "default"  # Change this to "default" if you don't want to include '0'
    save_expressions_to_file(filename, num_expressions, case)
    print(f"Generated {num_expressions} expressions and saved to {filename}")
    

if __name__ == "__main__":
    main()
