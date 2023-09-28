import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100  # Initial Stock Price (only for initial representation)
mu = 0.06  # Expected Return
sigma = 0.15  # Volatility
T = 1  # Time Period in Years
dt = 1 / 252  # Time Interval (one trading day)
n_steps = int(T * 252)  # Number of Time Steps
n_simulations = 1000  # Number of Simulations
monthly_savings = 0  # Monthly Savings in Euro
initial_investment = 100000  # Initial Investment in Euro

# Monte Carlo Simulation
prices = np.zeros((n_steps + 1, n_simulations))
prices[0] = S0

shares = np.zeros((n_steps + 1, n_simulations))
savings = np.zeros((n_steps + 1, n_simulations))

# Convert initial investment to shares
shares[0] = initial_investment / S0

for t in range(1, n_steps + 1):
    brownian = np.random.normal(0, np.sqrt(dt), n_simulations)
    prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * brownian)

    # Buy shares when monthly deposit is made
    if t % 21 == 0:
        shares_to_buy = monthly_savings / prices[t]
        shares[t] = shares[t - 1] + shares_to_buy
    else:
        shares[t] = shares[t - 1]

    # Update savings based on the number of shares and the current price
    savings[t] = shares[t] * prices[t]

# Total deposited amount (including the initial investment)
total_deposits = initial_investment + (monthly_savings * 12 * T)

# Plot results and calculate probabilities
average_savings = np.mean(savings[-1])
min_savings = np.min(savings[-1])
max_savings = np.max(savings[-1])
below_initial_investment = np.sum(savings[-1] < initial_investment)
prob_below_initial = below_initial_investment / n_simulations * 100
prob_above_initial = 100 - prob_below_initial

# Plot
plt.figure(figsize=(10, 6))
plt.plot(savings, lw=1, alpha=0.5)
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.title('Monte Carlo Simulation of Savings')
plt.xlabel('Time Steps')
plt.ylabel('Savings in Euro')

textstr = '\n'.join((
    f'Average Savings: €{average_savings:,.2f}',
    f'Minimum Savings: €{min_savings:,.2f}',
    f'Maximum Savings: €{max_savings:,.2f}',
    f'Total Deposits: €{total_deposits:,.2f}',
    f'Probability above Initial Investment: {prob_above_initial:.2f}%',
    f'Probability below Initial Investment: {prob_below_initial:.2f}%'
))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

plt.show()
