import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MonteCarloSimulation:
    def __init__(self, master):
        self.master = master
        master.title("Monte Carlo Simulation")

        # Initialize Variables
        self.S0 = tk.DoubleVar(value=100)
        self.mu = tk.DoubleVar(value=0.03)
        self.sigma = tk.DoubleVar(value=0.15)
        self.T = tk.IntVar(value=20)
        self.n_simulations = tk.IntVar(value=1000)
        self.monthly_savings = tk.DoubleVar(value=100)
        self.initial_investment = tk.DoubleVar(value=100000)

        # Layout
        ttk.Label(master, text="Initial Stock Price (S0):").grid(column=0, row=0, sticky='w')
        ttk.Entry(master, textvariable=self.S0).grid(column=1, row=0)

        ttk.Label(master, text="Expected Return (mu):").grid(column=0, row=1, sticky='w')
        ttk.Entry(master, textvariable=self.mu).grid(column=1, row=1)

        ttk.Label(master, text="Volatility (sigma):").grid(column=0, row=2, sticky='w')
        ttk.Entry(master, textvariable=self.sigma).grid(column=1, row=2)

        ttk.Label(master, text="Time Period in Years (T):").grid(column=0, row=3, sticky='w')
        ttk.Entry(master, textvariable=self.T).grid(column=1, row=3)

        ttk.Label(master, text="Number of Simulations:").grid(column=0, row=4, sticky='w')
        ttk.Entry(master, textvariable=self.n_simulations).grid(column=1, row=4)

        ttk.Label(master, text="Monthly Savings in Euro:").grid(column=0, row=5, sticky='w')
        ttk.Entry(master, textvariable=self.monthly_savings).grid(column=1, row=5)

        ttk.Label(master, text="Initial Investment in Euro:").grid(column=0, row=6, sticky='w')
        ttk.Entry(master, textvariable=self.initial_investment).grid(column=1, row=6)

        ttk.Button(master, text="Start Simulation", command=self.run_simulation).grid(column=0, row=7, columnspan=2)

        self.canvas_frame = ttk.Frame(master)
        self.canvas_frame.grid(column=0, row=8, columnspan=2)

    def run_simulation(self):
        # Retrieve Variables
        S0 = self.S0.get()
        mu = self.mu.get()
        sigma = self.sigma.get()
        T = self.T.get()
        n_simulations = self.n_simulations.get()
        monthly_savings = self.monthly_savings.get()
        initial_investment = self.initial_investment.get()

        # Parameters
        dt = 1 / 252
        n_steps = int(T * 252)

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
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(savings, lw=1, alpha=0.5)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

        ax.set_title('Monte Carlo Simulation of Savings')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Savings in Euro')

        textstr = '\n'.join((
            f'Average Savings: €{average_savings:,.2f}',
            f'Minimum Savings: €{min_savings:,.2f}',
            f'Maximum Savings: €{max_savings:,.2f}',
            f'Total Deposits: €{total_deposits:,.2f}',
            f'Probability above Initial Investment: {prob_above_initial:.2f}%',
            f'Probability below Initial Investment: {prob_below_initial:.2f}%'
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

        # Clear previous canvas
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # Embedding Plot in Tkinter Window
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()


root = tk.Tk()
simulation_app = MonteCarloSimulation(root)
root.mainloop()
