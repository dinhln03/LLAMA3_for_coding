import time

try:
    import wandb
    wandb_log=True
except ImportError:
    wandb_log=False

import numpy as np
from advbench.lib.plotting import plot_perturbed_wandb
from einops import rearrange

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, avg_mom=0.5):
        self.avg_mom = avg_mom
        self.reset()
        self.print = True

    def reset(self):
        self.val = 0
        self.avg = 0 # running average of whole epoch
        self.smooth_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.smooth_avg = val if self.count == 0 else self.avg*self.avg_mom + val*(1-self.avg_mom)
        self.avg = self.sum / self.count

class TimeMeter:
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.start = time.time()

    def batch_start(self):
        self.data_time.update(time.time() - self.start)

    def batch_end(self):
        self.batch_time.update(time.time() - self.start)
        self.start = time.time()
if wandb:
    class WBHistogramMeter:
        def __init__(self, name):
            self.print = False
            self.name = name

        def reset(self):
            pass

        def update(self, val):
            wandb.log({self.name: wandb.Histogram(val)})
    
    class WBDeltaMeter(WBHistogramMeter):
        def __init__(self, names = [], dims = 0, max_points = 100):
            self.max_points = max_points
            self.print = False
            self.dims = dims
            if isinstance(names, str):
                names = [f"{names} {i}" for i in range(dims)]
            self.meters = [WBHistogramMeter(name) for name in names]

        def reset(self):
            pass

        def update(self, vals):
            if self.dims>3:
                pass
            elif len(vals.shape)==3:
                for i in range(len(self.meters)):
                    self.meters[i].update(vals[:,i,:self.max_points].flatten())
            else:
                for i in range(len(vals[0])):
                    self.meters[i].update(vals[:,i])

    
    class WBLinePlotMeter():
        def __init__(self, name):
            self.print = False
            self.name = name
        def reset(self):
            pass
        def update(self, grid, vals):
            plot_perturbed_wandb(grid, vals, name=self.name)
    
    
    
    class WBDualMeter(WBHistogramMeter):
        def __init__(self, grid, translations, names = "dual vs angle", locs = [(0, 0), (-1,-1)], log_every=500):
            self.print = False
            self.locs = []
            tx, ty = translations
            for loc in locs:
                self.locs.append((grid[:,1]==tx[loc[0]])&(grid[:,2]==ty[loc[1]]))
            if isinstance(names, str):
                names = [f"{names} {grid[i[0], 1].detach().cpu().item(), grid[i[0], 2].detach().cpu().item()}" for i in locs]
            self.grid = grid
            self.meters = [WBLinePlotMeter(name) for name in names]
            self.log_every = log_every
            self.counter = 0

        def reset(self):
            self.counter=0

        def update(self, vals):
            if self.counter%self.log_every == 0:
                print("*"*10)
                print("log")
                for i in range(len(self.locs)):
                    self.meters[i].update(self.grid[self.locs[i], 0].detach().cpu().numpy(), vals[self.locs[i]].detach().cpu().numpy())
            self.counter+=1

else:
    class WBHistogramMeter:
        def __init__(self, name):
            self.print = False

        def reset(self):
            pass

        def update(self, val):
            pass

    class WBDeltaMeter(WBHistogramMeter):
        def __init__(self,names = [], dims = 0):
            self.print = False
