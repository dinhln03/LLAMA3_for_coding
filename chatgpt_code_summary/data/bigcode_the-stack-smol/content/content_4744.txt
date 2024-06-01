import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class BrownianStockSimulator:
    plot_title = "Simulated White noise, Brownian Motion and Stock Price"
    plotly_template = "plotly_white"
    plot_width = 1500
    plot_height = 1000

    def __init__(self, time_horizon, steps_count, sigma):
        self.time_horizon = time_horizon
        self.steps_count = steps_count
        self.sigma = sigma
        self.sampling_points = self.time_horizon * self.steps_count
        self.dt = self.time_horizon / self.sampling_points
        self.time_grid = self._get_time_grid()

    def _get_time_grid(self):
        time_grid = np.arange(0, self.time_horizon + self.dt, self.dt)
        return time_grid

    def _get_white_noise(self):
        white_noise = np.sqrt(self.dt) * np.random.normal(
            loc=0, scale=1.0, size=self.sampling_points
        )
        return white_noise

    def _get_brownian_motion(self, white_noise):
        brownian_motion = np.cumsum(white_noise)
        brownian_motion = np.append(0, brownian_motion)
        return brownian_motion

    def _get_stock_price(self, init_stock_price):
        output = (
            self.sigma * self.brownian_motion - 0.5 * self.sigma ** 2 * self.time_grid
        )
        return init_stock_price * np.exp(output)

    def simulate(self, init_stock_price, random_seed=42):
        np.random.seed(random_seed)
        self.white_noise = self._get_white_noise()
        self.brownian_motion = self._get_brownian_motion(self.white_noise)
        self.price = self._get_stock_price(init_stock_price)

    def plot(self):
        fig = make_subplots(rows=3, cols=1)
        fig.append_trace(
            go.Scatter(x=self.time_grid, y=self.white_noise, name="White Noise"),
            row=1,
            col=1,
        ),
        fig.append_trace(
            go.Scatter(
                x=self.time_grid, y=self.brownian_motion, name="Brownian Motion"
            ),
            row=2,
            col=1,
        ),
        fig.append_trace(
            go.Scatter(x=self.time_grid, y=self.price, name="Stock Price"), row=3, col=1
        )
        fig.update_layout(
            height=self.plot_height,
            width=self.plot_width,
            title_text=self.plot_title,
            template=self.plotly_template,
        )
        fig.show()
