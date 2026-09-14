
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Compute stock price tree for a BAPM
def compute_stock_price_tree(S0, u, d, N):
    '''
    S0: Initial stock price
    u: Up factor
    d: Down factor
    N: Number of time steps
    '''
    S = np.zeros(N+1)
    # For a path-independent derivative, we only need the distribution of SN
    # There are N+1 possible values for SN
    S = np.zeros(N+1)
    for i in range(N+1):
        # i is the number of up moves in N steps
        S[i] = S0 * (u ** i) * (d **(N-i))
        
    return S

def risk_neutral_prob(r, u, d):
    # r is the risk free rate
    return ((1+r) - d) / (u - d)

def compute_expectation(payoff, p, r, N):
    '''
    Compute E_p[Ṽ_n] for given probability p
    payoff: Payoff values corresponding to stock price outcomes
    p: Probability of 'up'-step
    '''
    expectation = 0
    discount_factor = 1 / ((1+r) ** N)
    for i in range(N+1):
        # Probability of exactly i up moves within N steps under the given 
        # probability measure p
        prob = binom.pmf(i, N, p)
        expectation += prob * discount_factor

    return expectation * discount_factor

def replicating_portfolio_step(V_up, V_down, S_up, S_down, r):
    '''
    Compute one step of the replicating portfolio
    
    V_up: Derivative value if the stock goes up
    V_down: Derivative value if stock goes down
    S_up: Stock price if it geos up
    S_down: stock price if it goes down
    '''
    # Solve for delta from the wealth equations:
    # V_up = Delta * S_up + (1+r * (V - Delta * S)
    # V_down -> similar
    Delta = (V_up - V_down) / (S_up - S_down)
    # Substitute:
    V = (V_up - Delta * S_up) / (1+r) + Delta * S_up / (1+r)
    # Return number of shares and value of derivative
    return Delta, V

def compute_replicating_portfolio(S0, u, d, r, N, payoff_func):
    '''
    S0: Initial stock price
    u: Up factor
    d: Down factor
    r: Risk free rate
    N: num. time steps
    payoff_func: Function that takes stock price and returns payoff
    '''
    V = {}
    Delta = {}
    # Compute all possible terminal stock prices:
    terminal_stocks = np.zeros(N+1)
    for i in range(N+1):
        terminal_stocks[i] = S0 * (u ** i) * (d ** (N-i))

    # Now compute the terminal payoffs at time step N
    V[N] = np.array([payoff_func(S) for S in terminal_stocks])

    # Extend one-step routine by running backwards to derive X0=V0
    for n in range(N-1, -1, -1):
        V[n] = np.zeros(n+1)
        Delta[n] = np.zeros(n+1)

        for j in range(n+1):
            # The current stock price after 'j' up-moves in n steps
            S_j = S0 * (u ** j) * (d **(n-j))
            # Stock price at n+1 for up and down cases:
            S_up = S_j * u
            S_down = S_j * d
            # Value at n+1 for up and down cases:
            V_up = V[n+1][j+1]  # j+1 up moves in n+1 steps
            V_down = V[n+1][j]  # j up moves in n+1 steps
            # Compute delta and the value at this node
            Delta[n][j], V[n][j] = replicating_portfolio_step(V_up, V_down, S_up, S_down, r)

    # Initial value V0 is the only element in V[0]:
    V0 = V[0][0]
    # Return the inital value, dictionary of deltas, and value at each node
    return V0, Delta, V



