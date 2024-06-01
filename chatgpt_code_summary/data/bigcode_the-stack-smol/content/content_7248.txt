from .history.pyplot_history import pyplot_history
from .history.plotly_history import plotly_history


def plot_history(history, engine="pyplot", **kwargs):
    if engine == "pyplot":
        return pyplot_history(history, **kwargs)
    elif engine == "plotly":
        return plotly_history(history, **kwargs)
    else:
        raise Exception("Unknown plotting engine")
