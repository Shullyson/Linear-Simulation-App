import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LinearRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Regression Simulator")

        # Linear Regression Equation Label
        self.equation_label = tk.Label(root, text="Linear System Equation:", font=("Helvetica", 12))
        self.equation_label.grid(row=0, column=0, columnspan=4, pady=10)

        # Slope Slider
        self.slope_label = ttk.Label(root, text="Slope (a):")
        self.slope_slider = tk.Scale(root, from_=-5, to=5, resolution=0.1, orient=tk.HORIZONTAL)
        self.slope_label.grid(row=1, column=0, padx=10)
        self.slope_slider.grid(row=1, column=1, padx=10)

        # Intercept Slider
        self.intercept_label = ttk.Label(root, text="Intercept (b):")
        self.intercept_slider = tk.Scale(root, from_=-50, to=50, resolution=1, orient=tk.HORIZONTAL)
        self.intercept_label.grid(row=1, column=2, padx=10)
        self.intercept_slider.grid(row=1, column=3, padx=10)

        # Number of Points Slider
        self.points_label = ttk.Label(root, text="Number of Points (N):")
        self.points_slider = tk.Scale(root, from_=10, to=100, resolution=10, orient=tk.HORIZONTAL)
        self.points_label.grid(row=2, column=0, padx=10)
        self.points_slider.grid(row=2, column=1, padx=10)

        # Noise Settings
        self.noise_label = ttk.Label(root, text="Noise:")
        self.noise_label.grid(row=3, column=0, padx=10, pady=10)

        self.noise_var = tk.StringVar()  # Variable to store selected noise type
        self.uniform_noise_button = ttk.Radiobutton(root, text="Uniform", variable=self.noise_var, value="uniform")
        self.gaussian_noise_button = ttk.Radiobutton(root, text="Gaussian", variable=self.noise_var, value="gaussian")

        self.uniform_noise_button.grid(row=3, column=1)
        self.gaussian_noise_button.grid(row=3, column=2)

        # Gaussian Noise Parameters
        self.mu_label = ttk.Label(root, text="Mean (μ):")
        self.mu_entry = ttk.Entry(root)
        self.sigma_label = ttk.Label(root, text="Standard Deviation (σ):")
        self.sigma_entry = ttk.Entry(root)

        self.mu_label.grid(row=4, column=0, padx=10)
        self.mu_entry.grid(row=4, column=1, padx=10)
        self.sigma_label.grid(row=4, column=2, padx=10)
        self.sigma_entry.grid(row=4, column=3, padx=10)

        # Buttons
        self.run_button = ttk.Button(root, text="Run Simulation", command=self.run_simulation)
        self.reset_button = ttk.Button(root, text="Reset Parameters", command=self.reset_parameters)
        self.debug_mode_var = tk.BooleanVar()
        self.debug_mode_checkbox = ttk.Checkbutton(root, text="Debug/Export Mode", variable=self.debug_mode_var)

        self.run_button.grid(row=5, column=0, pady=10)
        self.reset_button.grid(row=5, column=1, pady=10)
        self.debug_mode_checkbox.grid(row=5, column=2, pady=10)

        # Matplotlib Figure for Visualization
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=6, column=0, columnspan=4, pady=10)

        # Header/Footer
        self.header_label = ttk.Label(root, text="Your Name | Course Information", font=("Helvetica", 10), foreground="gray")
        self.header_label.grid(row=7, column=0, columnspan=4, pady=10)

    def run_simulation(self):
        # Fetching user inputs from sliders
        slope = self.slope_slider.get()
        intercept = self.intercept_slider.get()
        num_points = int(self.points_slider.get())
        noise_type = self.noise_var.get()
        mu = float(self.mu_entry.get()) if noise_type == "gaussian" and self.mu_entry.get() else None
        sigma = float(self.sigma_entry.get()) if noise_type == "gaussian" and self.sigma_entry.get() else None

        # Update Linear Equation Label
        equation_text = f"Linear System Equation: y = {slope}x + {intercept}"
        self.equation_label.config(text=equation_text)

        # Generate data points
        x = np.linspace(0, 10, num_points)
        y = slope * x + intercept + self.add_noise(noise_type, mu, sigma, num_points)

        # Perform Linear Regression
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        y_pred = model.predict(x.reshape(-1, 1))

        # Plotting
        self.ax.clear()
        self.ax.scatter(x, y, label="Given Points")
        self.ax.plot(x, y_pred.flatten(), color='red', label="Linear Regression")
        self.ax.set_title("Linear Regression Simulation")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()

        # Displaying R² value
        r_squared = r2_score(y, y_pred.flatten())
        self.header_label.config(text=f"Your Name | Course Information | R² Value: {r_squared:.4f}", foreground="gray")

        # Draw the canvas
        self.canvas.draw()

    def add_noise(self, noise_type, mu, sigma, size):
        if noise_type == "uniform":
            return np.random.uniform(-1, 1, size)
        elif noise_type == "gaussian":
            # Ensure that mu and sigma are not None
            if mu is None:
                mu = 0
            if sigma is None:
                sigma = 1
            return np.random.normal(mu, sigma, size)
        else:
            return np.zeros(size)

    def reset_parameters(self):
        # Reset all input fields and checkboxes
        self.slope_slider.set(0)
        self.intercept_slider.set(0)
        self.points_slider.set(50)
        self.mu_entry.delete(0, tk.END)
        self.sigma_entry.delete

if __name__ == "__main__":
    root = tk.Tk()
    app = LinearRegressionApp(root)
    root.mainloop()
