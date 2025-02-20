from cma import CMAEvolutionStrategy as cmaES
import numpy as np
import torch
import psutil
import cupy as cp
from torch.autograd import Variable
from torch.distributions import Normal, Categorical
import symengine as sm
import sympy as sp
import cProfile
import math
import sys
import os
import signal
import re


from model_pde import GrammarVAE
from util_pde import load_data, make_nltk_tree
from stack import Stack
import gc
from memory_profiler import profile
import tracemalloc
from contextlib import redirect_stdout
from st_pde import SystemProcessor, MeshGenerator
import logging
import random
# Configure logging
logging.basicConfig(filename='my_python_job_1.log', level=logging.INFO, format='%(asctime)s %(message)s')
# Set seeds for reproducibility
SEED = 2435
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Constants
# MODEL_PATH = '/cluster/scratch/ooikonomou/model_latest_10k-epoch=216-val_elbo=0.24-pde.ckpt'
# MODEL_PATH = '/cluster/scratch/ooikonomou/new/model_latest_10k-epoch=165-val_elbo=0.17-pde copy.ckpt'
MODEL_PATH = '/cluster/scratch/ooikonomou/new/model_latest_10k-epoch=248-val_elbo=0.17-pde.ckpt'
# MODEL_PATH = '/cluster/scratch/ooikonomou/model_latest_10k-epoch=171-val_elbo=0.26-pde.ckpt'
# DATA_PATH = '/cluster/scratch/ooikonomou/thesis/filtered_100_80_euler_39 rules-2terminal.h5'
DATA_PATH = '/cluster/scratch/ooikonomou/thesis/filtered_100_60_euler_sorted_1terminal-new-20-no0_in_const20k-2terminals.h5'
EPS = 1e-8
LATENT_DIM = 39
NUM_GENERATIONS = 1000
POPULATION_SIZE = 100
MAX_LENGTH = 80


# Set device to GPU if available, otherwise default to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# If using GPU, set the seed for all CUDA devices
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
   
# Initialize the model
model = GrammarVAE(435, LATENT_DIM, 501, 39, 'gru', MAX_LENGTH).to(device)

# Load the checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
state_dict = checkpoint['state_dict']
new_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# Load the data
data = load_data(DATA_PATH)
# data=None


# Function to log memory usage
def log_memory_usage(step_description):
    process = psutil.Process(os.getpid())
    memory_used = process.memory_info().rss / 1024 ** 2
    # logging.info(f"{step_description} - Memory usage: {memory_used:.2f} MB")

# Function to convert data to input tensor
def data2input(x):
    return Variable(torch.from_numpy(x).float().unsqueeze(0).transpose(-2, -1)).to(device)


# Function to construct an expression from rules
def construct_expression(rules):
    expression = 'S'
    for rule in rules:
        lhs, rhs = rule.lhs().symbol(), " ".join([str(r) for r in rule.rhs()])
        expression = expression.replace(lhs, rhs, 1)
    return expression

# Function to clean the generated expression
def clean_expression(expression):
    # expression = expression.replace('negx', '(-x)')
    # expression = expression.replace('negy', '(-t)')
    expression = expression.replace('y', 't')
    return expression

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, timeout_handler)


def decode_latent_vector(z):
                # torch.tensor(z,dtype=torch.float32, device=device).unsqueeze(0)
    z_tensor = torch.tensor(z,dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():

        decoded_expression = clean_expression(construct_expression(model.generate(z_tensor, sample=False, max_length=150)))
    return decoded_expression


def contains_variable(expr, var):
    """Check if a variable exists in the expression as a standalone symbol."""
    pattern = re.compile(rf'\b{var}\b')
    return bool(pattern.search(expr))
# @profile
def objective_function(z,u_expr):
    
    try:
        # logging.info('mesa')
        X, T, dx, dt = MeshGenerator.generate(0, 1, 100)
        
        decoded_expression = decode_latent_vector(z)
        
    
        
        # simplified_exp = sp.simplify(sm.sympify(decoded_expression))
        # print('simplified_exp',simplified_exp)

        # if (contains_variable(str(decoded_expression), 'x') and contains_variable(str(decoded_expression), 't') and('d3' not in str(decoded_expression)) and('d/dx(u^2)' in str(decoded_expression))
        #      and('d2/dx2(u)' not in str(decoded_expression))and('d2/dt2' not in str(decoded_expression))and('d/dx(u)' not in str(decoded_expression))):
        if (contains_variable(str(decoded_expression), 'x') and
            contains_variable(str(decoded_expression), 't') and
            ('dx'  in str(decoded_expression))              and
            ('dt'  in str(decoded_expression))              and
            ('(u) * d/dx(u^2)' not in str(decoded_expression))
             ):
                logging.info('inside objective fuction if')
                logging.info(f'decoded_expressio:{decoded_expression}')
                processor = SystemProcessor(decoded_expression,u_expr)
                logging.info(f'processor{processor}')
                simplified_expressions = processor.substitute_and_simplify()
                logging.info(f'simplified_expressions{simplified_expressions}')
                # Evaluate the simplified expressions on the grid
                evaluated_results = processor.evaluate_on_grid(simplified_expressions, X, T)
                # logging.info(f'evaluated_results{evaluated_results}')
                # Calculate RMSE for each evaluated result
                loss = processor.calculate_rmse(evaluated_results)
                logging.info(f'loss:{loss}')
                gc.collect()
                return loss
            
        else:
            return 100
    except Exception as e:
        logging.info('Exception: {e}')
        return 100
      

def cmaes_optimization(u_expr,num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,mu=[0]*LATENT_DIM,sigma=1):
    
    best_solution = None
    best_loss = 1000
    # logging.info('miaou')
    current_population_size = population_size
    restart_limit = 10
    generations_no_improvement = 10
    while best_loss > 1e-3:
        no_improvement_counter = 0
        es = cmaES(mu,sigma, {'popsize': population_size, 'verbose': 1, 'maxiter': num_generations, 'ftarget': 1e-3, 'seed': SEED})
        for generation in range(num_generations):
            logging.info(f"\nGeneration {generation + 1}/{num_generations}")
            sys.stdout.flush()
            
            solutions = es.ask()
            losses = [objective_function(np.array(sol),u_expr) for sol in solutions]
            es.tell(solutions, losses)
            
            # logging.info('losses',losses)
            current_best_loss = min(losses)
            current_best_solution = solutions[np.argmin(losses)]

            if current_best_loss < best_loss:
                best_loss = current_best_loss
                best_solution = current_best_solution
                no_improvement_counter = 0  # Reset counter if improvement is found
            else:
                no_improvement_counter += 1
            if no_improvement_counter >= generations_no_improvement:
                logging.info(f"Restarting due to no improvement for {generations_no_improvement} generations.")
                break
            

            logging.info(f"Generation {generation}, Current Best Loss: {current_best_loss}")
            sys.stdout.flush()
            logging.info(f"Current Best Solution (latent vector): {current_best_solution}")
            sys.stdout.flush()

            if best_loss <= 1e-3:  # early stopping condition
                break
            # Clear memory after each generation
            # gc.collect()
            # logging.info('best loss',best_loss)
            log_memory_usage(f"After generation {generation}")
        if best_loss <= 1e-3:  # early stopping condition
                break  
        # Increase population size for the next restart
        current_population_size *= 1
        
    return best_solution, best_loss

def existing_operator_checking(dataset, u_expr, model):
        """
        For each data point in the dataset, compute mu and sigma, sample z,
        decode the expression, and evaluate using the objective function logic.
        """
        X, T, dx, dt = MeshGenerator.generate(0, 1, 100)
        
        total_loss = 0.0
        count = 0
    
        for i in range(len(dataset)):
            try:
        # Get the latent vector z for the data point
        # i = 5929

                # i = 334
                i=201
                # i=104
                x = data[i]  # Assuming data[index] retrieves the input features for the specified data point
                # print(x)
                x_tensor = data2input(x)
                print(x_tensor)
                # Compute mu and sigma
                with torch.no_grad():
                    mu, sigma = model.encoder(x_tensor)
                    # print('check1')
                    mu, sigma = mu.squeeze().cpu(), sigma.squeeze().cpu()
                    # print('check2')
                normal = Normal(torch.zeros(mu.shape).to(device), torch.ones(sigma.shape).to(device))
                eps = Variable(normal.sample()).to(device)
                z = mu + eps * torch.sqrt(sigma)*0.001
                
                # Decode the latent vector to get the expression
                decoded_expression = decode_latent_vector(z)
                logging.info(f'decoded_expression {i}: {decoded_expression}')



        # # Process the expression
        # processor = SystemProcessor(decoded_expression, u_expr)
        # # logging.info(f'processor {i}: {processor}')
        
        # simplified_expressions = processor.substitute_and_simplify()
        # logging.info(f'simplified_expressions {i}: {simplified_expressions}')
        
        # # Evaluate the simplified expressions on the grid
        # evaluated_results = processor.evaluate_on_grid(simplified_expressions, X, T)
        # # logging.info(f'evaluated_results {i}: {evaluated_results}')
        
        # # Calculate RMSE for the evaluated result
        # loss = processor.calculate_rmse(evaluated_results)
        # logging.info(f'loss {i}: {loss}')
        
            
        # if loss == 0:
        #     break
        
                return mu,sigma
            except Exception as e:
                logging.info(f'Exception with expression {i}: {e}')
def find_mu_sigma(index, data, model):
    # Retrieve the input x for data point at the specified index
    x = data[index]  # Assuming data[index] retrieves the input features for the specified data point
    print('x',x)
    x_tensor = data2input(x)

    # Compute mu and sigma
    with torch.no_grad():
        mu, sigma = model.encoder(x_tensor)
        mu, sigma = mu.squeeze().cpu().numpy(), sigma.squeeze().cpu().numpy()
        print('type of',type(mu))
    return mu.tolist(), sigma.tolist()


# Main logic
if __name__ == '__main__':
    #PDE3
   
    # u_expr = "1.5*t + x^2 - 2.0"
    # u_expr = "1.5*t^2 + x - 1.0"
    # u_expr = "1.5*sin(t) + cos(x)^2 - 2.0"
    # u_expr = "2.53*x + sin(t+1.3) - 2.2" #5580 - heat - (6.0*d2/dx2(u)+1.0*d/dt(u)-(4.0*t^3))
    # u_expr = "1.5*x + sin(t) - 3.0" #5580 - heat - (6.0*d2/dx2(u)+1.0*d/dt(u)-(4.0*t^3))
    # u_expr ="2.0*t + 2.0x
    #
    # + 1.0" #17
    # u_expr ="1.0+t^3*cos(x - 1.0)" #wave
    # u_expr="1.0*x*sin(t - 3.0)" # wave 3*x*cos(t - 2.2) - 6.0*d2/dx2(u)+2.0*d2/dt2(u)+(6.0*x*cos(t-2.2)) #new 4.0*d2/dx2(u)+3.0*d2/dt2(u)+(3.0*x*sin(t-3.0))
    # u_expr="sin(x + 1.0) + 2.0*t" # advection or burger x + 5*t - 
    # u_expr="x + sin(t)" # advection cos(x) + t -
    # u_expr="3.1*x + 2.0*t + 1.1" # advection cos(x) + t -
    # u_expr="5.0*x + cos(t)" # advection cos(x) + t -
    # u_expr="sin(x)*cos(x)+2.0*t*cos(x)+2.0"
    # u_expr="2.00*t+1.08*(x)^3"
    # u_expr="(t)^2*cos(x*2.44)"
    # u_expr="5.12*cos((t)^2)+2.03*sin((x)^3)"
    # u_expr="t^2+1.08*(x)"
    # u_expr="sin(t+1.02)+1.232*x+2.34"
    # u_expr="sin(t+2.02)+2.1*x+2.74"
    # u_expr="cos(2.94*x-1.33)-cos(t+1.18)"
    # u_expr="t^2+1.18*(x)^2"
    # u_expr="cos((x)^2)*sin(t-1.99)"
    # u_expr="sin(x+2.21)-cos(t*1.48)"
    # u_expr="t+ cos(x) +1.74"
    # u_expr="sin(t-1.78)-cos(2.25*1.75*x-2.56)"
    # u_expr="sin((x)^3)*t+3.87"#advection
    # u_expr="sin((t)^2)-cos(x+0.23)" #wave
    # u_expr="sin(x*0.75)*sin(t+1.49)"
    # u_expr="sin(t*1.80)-x+2.04"#burger
    # u_expr="sin(t-2.24)+cos(1.80*(x)^3)"#burger
    # u_expr="sin(x*1.80)*t*1.80"#advection
    # u_expr="cos(t*1.32)+sin((x)^3)"
    # u_expr="cos(t*1.80)+cos(x-0.45)"
    # u_expr="sin(0.45*x+0.04)-t*1.10-0.46"#h5
    # u_expr="t-cos(cos((x)^2))"#advection (t-0.64*cos(x+1.10))
    # u_expr="t-cos(x+2.10)" #adv
    # u_expr="sin(1.36*t)+cos(1.24*x +0.12)" #heat
    # u_expr="sin(1.36*t)+cos((x)^1)" #heat
    # u_expr="0.35 + 0.5*t + 0.6*x"#adv
    # u_expr="0.35 + 0.5*t + 0.6*x"#adv
    # u_expr="sin(t*1.57)-x+1.11"#burger
    # # u_expr="t*1.50-x-0.45" #25530 heat
    # # u_expr=" sin((t^2))-cos(x-2.64)"
    # u_expr="4.92*(x+t)-4.32"#heat 5185
    # u_expr="t*0.02-x*0.97"#wave 6702
    # u_expr="t*0.02-x+0.05"#burger 44
    # u_expr="(t)^3*sin(x+0.05)"#longer burger7
    # u_expr="(t)^3*sin(x+0.02)"#longer burger
    # u_expr="x-1.44+sin(0.1*t*7.69+0.01)"#longer burger35
    # u_expr="x-1.44-sin(0.10*t*7.70)"#longer burger35
    # u_expr="0.5*x-0.16+(t)^3"#reaction-diffusion 5481
    # u_expr="cos(t+2.22)+x+0.33"#5512
    # u_expr="(sin(t-1.37)-x*0.36)"#5536
    # u_expr="1.09*x+cos(t+0.57)"#5930
    # u_expr="1.09*x+cos(t+0.66)"#5930
    # u_expr="log(1+x^2+t^2)"#lagaris
    # u_expr="x+1.09*sin((t)^3)"# 144k 2920
    # u_expr="(t)^3*(x)^3"
    # u_expr="2.2(t)*(x)^2"
    # u_expr="cos(x)*sin(t+0.1)"
    # # u_expr="-1.52+1.08+x-t"
    # u_expr="t-x-0.52"
    u_expr="x+2.56+cos(t+1.97-0.20)"
    
    # index = 0

    # mu, sigma = find_mu_sigma(index, data, model)
    # print(f"mu for data point {index}: {mu}")
    # print(f"sigma for data point {index}: {sigma}")
   
    mu,sigma=existing_operator_checking(data, u_expr, model)
    logging.info('sigma',sigma)
    # mu=[0]*LATENT_DIM
    # # logging.info('optimizer starting')
    h_opt, _ = cmaes_optimization(u_expr,mu=mu,sigma=0.1)
    
    optimized_expression = decode_latent_vector(torch.tensor(h_opt))
    logging.info(f"Optimized Expression that we get: {optimized_expression}") 





