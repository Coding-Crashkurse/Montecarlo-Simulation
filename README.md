# Monte Carlo Simulation for Investment

This project simulates the potential evolution of an investment portfolio using the Monte Carlo method. It allows users to visualize different scenarios of savings and investment returns over a period, taking into consideration factors like expected return, volatility, and regular monthly savings.

## Features

- **Tkinter GUI:** Allows users to input parameters easily and visualize results interactively.
- **Flexible Parameters:** Users can change initial stock price, expected return, volatility, time period, number of simulations, monthly savings, and initial investment to observe different outcomes.
- **Probabilistic Analysis:** The simulation provides probabilities for ending up above or below the initial investment.

## Getting Started

### Prerequisites

- Python 3
- Matplotlib
- Numpy
- Tkinter

### Running the Application

1. Clone the repository
2. Run the Tkinter GUI Python script.

```shell
python monte_carlo_simulation_gui.py
```


### Input Parameters

- **Initial Stock Price (S0):** The initial price of the stock (Used for the initial representation).
- **Expected Return (mu):** The expected return of the stock.
- **Volatility (sigma):** The volatility (standard deviation of return) of the stock.
- **Time Period in Years (T):** The total time period of the simulation in years.
- **Number of Simulations:** The number of simulations to run.
- **Monthly Savings in Euro:** The amount to be saved every month.
- **Initial Investment in Euro:** The initial amount invested.

## Output

The application will render a plot representing the evolution of savings over time for each simulation. It will also display average, minimum, and maximum savings, total deposits, and probabilities of ending up above or below the initial investment.

## Code Snippet

### Monte Carlo Simulation

```python
# Monte Carlo Simulation
prices = np.zeros((n_steps + 1, n_simulations))
prices[0] = S0
shares = np.zeros((n_steps + 1, n_simulations))
savings = np.zeros((n_steps + 1, n_simulations))
shares[0] = initial_investment / S0  # Convert initial investment to shares

for t in range(1, n_steps + 1):
    brownian = np.random.normal(0, np.sqrt(dt), n_simulations)
    prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * brownian)
    if t % 21 == 0:  # Buy shares when monthly deposit is made
        shares_to_buy = monthly_savings / prices[t]
        shares[t] = shares[t - 1] + shares_to_buy
    else:
        shares[t] = shares[t - 1]
    savings[t] = shares[t] * prices[t]  # Update savings based on the number of shares and the current price
```

Have fun!