"""
@brief      test log(time=20s)
"""

import sys
import os
import unittest
from pyquickhelper.loghelper import fLOG, run_cmd
from pyquickhelper.pycode import get_temp_folder, fix_tkinter_issues_virtualenv, skipif_appveyor, skipif_travis
from pyquickhelper.pycode import add_missing_development_version


class TestPyData2016Animation(unittest.TestCase):

    @skipif_appveyor("no ffmpeg installed")
    @skipif_travis("issue with datashader.bokeh_ext, skipping")
    @skipif_appveyor("issue with pyproj")
    def test_matplotlib_example(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        progs = ["ffmpeg"]
        if not sys.platform.startswith("win"):
            progs.append("avconv")
        errs = []
        prog = None
        for prog in progs:
            out, err = run_cmd(prog, wait=True, fLOG=fLOG)
            exps = "usage:"
            if (exps not in out and exps not in err) or err is None or len(err) == 0:
                errs.append((prog, err))
            else:
                break

        if len(errs) >= len(progs):
            if sys.platform.startswith("win"):
                fLOG("download ffmpeg")
                add_missing_development_version(
                    ["pyensae"], __file__, hide=True)
                from pyensae.datasource import download_data
                download_data("ffmpeg.zip", website="xd")
            else:
                raise FileNotFoundError(
                    "Unable to find '{1}'.\nPATH='{0}'\n--------\n[OUT]\n{2}\n[ERR]\n{3}".format(
                        os.environ["PATH"], prog, out,
                        "\n----\n".join("{0}:\n{1}".format(*_) for _ in errs)))

        temp = get_temp_folder(__file__, "temp_example_example")
        fix_tkinter_issues_virtualenv()

        # update a distribution based on new data.
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.stats as ss
        from matplotlib.animation import FuncAnimation, writers

        # To get the list of available writers
        if not writers.is_available(prog):
            writers.register(prog)
        fLOG(writers.list())

        class UpdateDist:

            def __init__(self, ax, prob=0.5):
                self.success = 0
                self.prob = prob
                self.line, = ax.plot([], [], 'k-')
                self.x = np.linspace(0, 1, 200)
                self.ax = ax

                # Set up plot parameters
                self.ax.set_xlim(0, 1)
                self.ax.set_ylim(0, 15)
                self.ax.grid(True)

                # This vertical line represents the theoretical value, to
                # which the plotted distribution should converge.
                self.ax.axvline(prob, linestyle='--', color='black')

            def init(self):
                self.success = 0
                self.line.set_data([], [])
                return self.line,

            def __call__(self, i):
                # This way the plot can continuously run and we just keep
                # watching new realizations of the process
                if i == 0:
                    return self.init()

                # Choose success based on exceed a threshold with a uniform
                # pick
                if np.random.rand(1,) < self.prob:  # pylint: disable=W0143
                    self.success += 1
                y = ss.beta.pdf(self.x, self.success + 1,
                                (i - self.success) + 1)
                self.line.set_data(self.x, y)
                return self.line,

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ud = UpdateDist(ax, prob=0.7)
        anim = FuncAnimation(fig, ud, frames=np.arange(100), init_func=ud.init,
                             interval=100, blit=True)

        try:
            Writer = writers[prog]
        except KeyError as e:
            if prog == "avconv":
                from matplotlib.animation import AVConvWriter
                Writer = AVConvWriter
            else:
                raise e
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(os.path.join(temp, 'lines2.mp4'), writer=writer)

        plt.close('all')
        fLOG("end")


if __name__ == "__main__":
    unittest.main()
