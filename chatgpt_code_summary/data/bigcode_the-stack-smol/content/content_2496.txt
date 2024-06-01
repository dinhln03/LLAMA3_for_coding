#!/usr/bin/env python -u
"""
All commands that can be run in this project are available through this unified interface.
This should be run with the ./plaster.sh helper to get into the correct context.
"""
import tempfile
import numpy as np
import time
import os
import sys
import pandas as pd
import json
from pathlib import Path
from munch import Munch
from plumbum import colors
from plumbum import FG, TF, cli, local
from plaster.tools.zlog.zlog import important
from plaster.run.sigproc_v2 import synth
from plaster.tools.zlog.profile import prof, profile_from_file, profile_dump
from plaster.tools.utils.tmp import tmp_file
from plaster.tools.assets import assets
from plaster.tools.test_tools.test_tools import run_p
from plaster.run.run import RunResult
from plaster.tools.zlog import zlog
from plaster.tools.zlog.zlog import tell, h_line, spy
from plaster.tools.utils import tmp
from plaster.tools.utils import utils


import logging

log = logging.getLogger(__name__)


class CommandError(Exception):
    def __init__(self, retcode=None):
        self.retcode = retcode


def assert_env():
    must_exist = ("ERISYON_ROOT", "JOBS_FOLDER")
    found = 0
    for e in must_exist:
        if e in local.env:
            found += 1
        else:
            print(f'Environment variable "{e}" not found.')

    if found != len(must_exist):
        raise CommandError(f"Environment variable(s) not found.")


class DoFuncs:
    def is_dev(self):
        return local.env.get("ERISYON_DEV") == "1"

    def folder_user(self):
        return local.env["FOLDER_USER"]

    def run_user(self):
        return local.env["RUN_USER"]

    def clear(self):
        local["clear"] & FG

    def _print_job_folders(self, file_list, show_plaster_json=True):
        """
        file_list is a list of munches [Munch(folder="folder", name="foo.txt", size=123, mtime=123456789)]
        """

        if len(file_list) == 0:
            print("No files found")
            return

        folders = {
            file.folder: Munch(folder=file.folder, size_gb=0, file_count=0,)
            for file in file_list
        }

        gb = 1024 ** 3
        total_gb = 0
        for file in file_list:
            folder = file.folder
            total_gb += file.size / gb
            folders[folder].size_gb += file.size / gb
            folders[folder].file_count += 1

        df = pd.DataFrame.from_dict(folders, orient="index")
        formatters = dict(
            size_gb="{:10.2f}".format,
            folder="{:<40.40s}".format,
            file_count="{:.0f}".format,
        )
        columns = ["folder", "size_gb", "file_count"]

        df = df.append(dict(folder="TOTAL", size_gb=total_gb), ignore_index=True)

        print(df.to_string(columns=columns, formatters=formatters))

    def print_local_job_folders(self):
        important("Local job folders:")

        root = local.path("./jobs_folder")
        self._print_job_folders(
            [
                Munch(
                    folder=(p - root)[0],
                    name=p.name,
                    size=int(p.stat().st_size),
                    mtime=int(p.stat().st_mtime),
                )
                for p in root.walk()
            ]
        )

    def validate_job_folder(self, job_folder, allow_run_folders=False):
        return assets.validate_job_folder(
            job_folder, allow_run_folders=allow_run_folders
        )

    def run_zests_v2(self, cli_args, debug_mode):
        tell(f"Running zests v2...")

        # as os.environ is evaluated when it is first imported
        # we can't use any of the more graceful ways to set the environment
        with local.env(RUN_ENV="test", ZAP_DEBUG_MODE=debug_mode):
            zest_version = None
            try:
                from zest.version import __version__ as zest_version
            except ImportError:
                pass

            assert zlog.config_dict is not None
            assert zest_version.startswith("1.1.")
            with tmp.tmp_file() as tmp_path:
                with open(tmp_path, "w") as f:
                    f.write(json.dumps(zlog.config_dict))

                # cli_args += ["--logger_config_json", tmp_path]
                local["python"]["-u", "-m", "zest.zest_cli"].bound_command(
                    *cli_args
                ) & FG(retcode=None)

    def run_nbstripout(self):
        """Strip all notebooks of output to save space in commits"""
        important("Stripping Notebooks...")
        result = (
            local["find"][
                ".",
                "-type",
                "f",
                "-not",
                "-path",
                "*/\.*",
                "-name",
                "*.ipynb",
                "-print",
            ]
            | local["xargs"]["nbstripout"]
        ) & TF(FG=True)

        if not result:
            raise CommandError

    def run_docker_build(self, docker_tag, quiet=False):
        important(f"Building docker tag {docker_tag}")
        with local.env(LANG="en_US.UTF-8"):
            args = [
                "build",
                "-t",
                f"erisyon:{docker_tag}",
                "-f",
                "./scripts/main_env.docker",
            ]
            if quiet:
                args += ["--quiet"]
            args += "."
            local["docker"][args] & FG


class DoCommand(cli.Application, DoFuncs):
    def main(self):
        return


@DoCommand.subcommand("run_notebook")
class RunNotebookCommand(cli.Application, DoFuncs):
    """
    Run a notebook rendered to HTML
    """

    def main(self, notebook_path, output_path: Path = None):
        args = [
            "nbconvert",
            "--to",
            "html",
            "--execute",
            notebook_path,
            "--ExecutePreprocessor.timeout=1800",
        ]
        if output_path is not None:
            args += ["--output", output_path]
        local["jupyter"].bound_command(*args) & FG


@DoCommand.subcommand("profile")
class ProfileCommand(cli.Application, DoFuncs):
    gb = 1024 ** 3

    skip_hardware = cli.Flag("--skip_hardware", help="Do not include hardware profile")
    skip_sigproc = cli.Flag("--skip_sigproc", help="Do not include sigproc profile")

    def fileio_test(self, jobs_folder):
        job_name = f"_profile/_{int(time.time()):08x}"
        large_random = np.random.uniform(
            size=1024 ** 3 // 8
        )  # 8 because floats are 8 bytes

        def write_to(write_path):
            # import shutil
            # total, used, free = shutil.disk_usage(write_path.dirname)
            # print(f"Free disk at {write_path}: {free / gb:2.2f}GB ({free / total:2.1f}%)")

            write_path.dirname.mkdir()
            with open(write_path, "wb") as f:
                f.write(large_random)

        # PROFILE write to jobs_folder
        job_folder_write_path = jobs_folder / job_name
        try:
            with prof(
                "fileio_to_jobs_folder", gbs=large_random.nbytes / self.gb, _tell=True,
            ):
                write_to(job_folder_write_path)
        finally:
            job_folder_write_path.delete()

        # PROFILE write to plaster_tmp
        with tmp_file() as plaster_tmp_folder_write_path:
            with prof(
                "fileio_to_plaster_tmp", gbs=large_random.nbytes / self.gb, _tell=True,
            ):
                write_to(plaster_tmp_folder_write_path)

        # PROFILE write to /tmp
        tmp_folder_write_path = local.path(tempfile.mkstemp())
        try:
            with prof("fileio_to_tmp", gbs=large_random.nbytes / self.gb, _tell=True):
                write_to(tmp_folder_write_path)
        finally:
            tmp_folder_write_path.delete()

    def cpu_test(self):
        mat = np.random.uniform(size=(5000, 5000))
        with prof(
            "cpu_tests_matrix_invert",
            mega_elems=(mat.shape[0] * mat.shape[1]) / 1e6,
            _tell=True,
        ):
            np.linalg.inv(mat)

    def mem_test(self):
        gb = 1024 ** 3
        rnd = np.random.uniform(size=(1_000, 500_000))

        with prof("mem_tests_copy", gbs=rnd.nbytes / gb, _tell=True):
            rnd.copy()

    def sigproc_test(self, jobs_folder):
        """
        This is adapted from zest_sigproc_v2_integration
        """
        profile_folder = jobs_folder / "_profile"
        profile_folder.delete()
        job_folder = profile_folder / "sigproc_test"
        source_folder = profile_folder / "_synth_field"
        job_folder.mkdir()
        source_folder.mkdir()

        # GENERATE some fake data

        dim = (1024, 1024)
        n_channels = 1
        n_cycles = 10
        n_peaks = 500
        psf_width = 1.5
        bg_mean = 100.0
        bg_std = 30.0
        gain = 5000.0

        def _synth_field(fl_i):
            with synth.Synth(n_channels=n_channels, n_cycles=n_cycles, dim=dim) as s:
                peaks = (
                    synth.PeaksModelGaussianCircular(n_peaks=n_peaks)
                    .locs_randomize()
                    .widths_uniform(psf_width)
                    .amps_constant(gain)
                )
                synth.CameraModel(bg_mean=bg_mean, bg_std=bg_std)
                synth.HaloModel()
                synth.IlluminationQuadraticFalloffModel()

            chcy_ims = s.render_chcy(0)

            for ch_i in range(chcy_ims.shape[0]):
                for cy_i in range(chcy_ims.shape[1]):
                    np.save(
                        str(
                            source_folder
                            / f"area_{fl_i:03d}_cell_000_{ch_i:03d}nm_{cy_i:03d}.npy"
                        ),
                        chcy_ims[ch_i, cy_i],
                    )

        n_fields = 2
        for fl_i in range(n_fields):
            _synth_field(fl_i)

        run_p(
            [
                f"gen",
                f"sigproc_v2",
                f"--job={job_folder}",
                f"--sigproc_source={source_folder}",
                f"--force",
                f"--self_calib",
            ]
        )

        log_file = local.path(local.env["PLASTER_ROOT"]) / "plaster.log"
        log_file.delete()

        run_p(["run", job_folder, "--no_progress", "--skip_reports"])

        profile_lines = profile_from_file(log_file)

        with colors.fg.DeepSkyBlue3:
            print()
            print(h_line("--"))
            print("PROFILE RESULTS")
            print(h_line("--"))
        profile_dump(profile_lines)

    def main(self, jobs_folder):
        assert_env()

        jobs_folder = local.path(jobs_folder)

        if not self.skip_hardware:
            tell(colors.cyan | "Profiling file_io")
            self.fileio_test(jobs_folder)

            tell(colors.cyan | "Profiling cpu")
            self.cpu_test()

            tell(colors.cyan | "Profiling mem")
            self.mem_test()

        if not self.skip_sigproc:
            tell(colors.cyan | "Profiling sigproc")
            self.sigproc_test(jobs_folder)


@DoCommand.subcommand("profile_dump")
class ProfileDumpCommand(cli.Application, DoFuncs):
    def main(self, log_path):
        assert_env()

        log_file = local.path(log_path)
        profile_lines = profile_from_file(log_file)
        profile_dump(profile_lines)


@DoCommand.subcommand("test")
class TestCommand(cli.Application, DoFuncs):
    """
    Run tests
    """

    no_clear = cli.Flag("--no_clear", help="Do not clear screen")
    integration = cli.Flag("--integration", help="Run integration tests")
    debug_mode = cli.Flag("--debug_mode", help="Put zap into debug_mode")
    cli_mode = cli.Flag("--cli_mode", help="Run without ui")

    def main(self, *args):
        if not self.no_clear:
            self.clear()

        cli_args = list(args)

        root = local.env["PLASTER_ROOT"]
        cli_args += [f"--root={root}"]

        folders = (
            "./plaster",
            "./plaster/scripts",
        )
        include_dirs = ":".join(folders)
        cli_args += [f"--include_dirs={include_dirs}"]
        with local.cwd(root):
            cli_args += [f"--hook_start=./scripts/testing_start.py:test_setup_logs"]

            if not self.debug_mode:
                if not self.cli_mode:
                    cli_args += [f"--ui"]
                cli_args += [f"--n_workers", "8"]

            if self.integration:
                cli_args += [f"--groups=integration"]
            else:
                cli_args += [f"--exclude_groups=integration"]

            return self.run_zests_v2(cli_args, self.debug_mode)


@DoCommand.subcommand("jupyter")
class JupyterCommand(cli.Application, DoFuncs):
    ip = cli.SwitchAttr("--ip", str, default="0.0.0.0", help="ip to bind to")
    port = cli.SwitchAttr("--port", int, default="8080", help="port to bind to")

    def main(self, *args):
        assert_env()
        os.execlp(
            "jupyter",
            "jupyter",
            "notebook",
            f"--ip={self.ip}",
            f"--port={self.port}",
            "--allow-root",
            *args,
        )


@DoCommand.subcommand("pluck")
class PluckCommand(cli.Application, DoFuncs):
    """
    Pluck a field from a result pickle
    """

    save_npy = cli.SwitchAttr("--save_npy", str, default=None, help="save as npy file")
    save_csv = cli.SwitchAttr(
        "--save_csv", str, default=None, help="save as csv file (dataframe only)"
    )
    save_pkl = cli.SwitchAttr(
        "--save_pkl", str, default=None, help="save as pkl file (dataframe only)"
    )

    def main(self, run_path, symbol):
        """
        run_path: path to the run folder
        symbol: Eg: "sigproc_v2.sig"
        """
        run = RunResult(run_path)
        parts = symbol.split(".")
        result = run[parts[0]]
        sym = getattr(result, parts[1])
        if callable(sym):
            val = sym()
        else:
            val = sym

        if self.save_npy is not None:
            assert isinstance(val, np.ndarray)
            np.save(self.save_npy, val)
        if self.save_csv is not None:
            assert isinstance(val, pd.DataFrame)
            val.to_csv(self.save_csv)
        if self.save_pkl is not None:
            assert isinstance(val, pd.DataFrame)
            val.to_pickle(self.save_pkl)


@DoCommand.subcommand("export_sigproc_v2")
class ExportSigprocV2Command(cli.Application, DoFuncs):
    """
    Export sigproc_v2 and raw data in easy to use formats.
    """

    def main(self, run_path):
        """
        run_path: path to the run folder (don't forget this is a subfolder of job)
        """
        run = RunResult(run_path)
        name = run.run_folder.parent.name

        prefix = f"{name}__"
        tell(f"Prefixing saved files with {prefix}")

        tell("Saving sig.npy")
        np.save(f"{prefix}sig.npy", run.sigproc_v2.sig())

        tell("Saving noi.npy")
        np.save(f"{prefix}noi.npy", run.sigproc_v2.noi())

        tell("Saving df.csv")
        run.sigproc_v2.fields__n_peaks__peaks__radmat().to_csv(f"{prefix}df.csv")

        ims = []
        for fl_i in range(run.sigproc_v2.n_fields):
            tell(f"Loading align field {fl_i} of {run.sigproc_v2.n_fields}")
            ims += [run.sigproc_v2.aln_unfilt_chcy_ims(fl_i)]

        tell("Saving aln_ims.npy")
        np.save(f"{prefix}aln_ims.npy", np.stack(ims))

        tell("Saving example.py")
        utils.save(
            f"{prefix}example.py",
            f"import numpy as np\n"
            + f"import pandas as pd\n\n"
            + f'prefix = "{prefix}"'
            + utils.smart_wrap(
                """
                sig = np.load(f"{prefix}sig.npy")
                noi = np.load(f"{prefix}noi.npy")
                df = pd.read_csv(f"{prefix}df.csv")
                ims = np.load(f"{prefix}aln_ims.npy", mmap_mode="r")
                n_peaks = sig.shape[0]
                n_fields, n_channels, n_cycles, im_mea, _ = ims.shape

                # Examine some peak
                peak_i = 123  # 0 <= peak_i < n_peaks
                ch_i = 0  # 0 <= ch_i < n_channels
                cy_i = 0  # 0 <= cy_i < n_cycles
                y, x, fl_i = df[df.peak_i == peak_i][["aln_y", "aln_x", "field_i"]].drop_duplicates().values.flatten().astype(int)
                peak_radius = 10
                peak_im = ims[fl_i, ch_i, cy_i, y-peak_radius:y+peak_radius, x-peak_radius:x+peak_radius]
                # Now peak_im is a centered sub-image of that peak with shape=(peak_radius, peak_radius)
            """,
                width=200,
                assert_if_exceeds_width=True,
            ),
        )

        tell("\n\nThe following commands may be useful:")
        # tell(f"  tar czf {prefix}data.tar.gz {prefix}sig.npy {prefix}noi.npy {prefix}df.csv")
        # tell(f"  tar czf {prefix}ims.tar.gz {prefix}aln_ims.npy")
        # tell("")
        # tell(f"  aws s3 cp {prefix}data.tar.gz s3://erisyon-public")
        # tell(f"  aws s3 cp {prefix}ims.tar.gz s3://erisyon-public")
        tell(f"  aws s3 cp {prefix}sig.npy s3://erisyon-public")
        tell(f"  aws s3 cp {prefix}noi.npy s3://erisyon-public")
        tell(f"  aws s3 cp {prefix}df.csv s3://erisyon-public")
        tell(f"  aws s3 cp {prefix}aln_ims.npy s3://erisyon-public")
        tell(f"  aws s3 cp {prefix}example.py s3://erisyon-public")


if __name__ == "__main__":
    try:
        DoCommand.subcommand("gen", "plaster.gen.gen_main.GenApp")
        DoCommand.subcommand("run", "plaster.run.run_main.RunApp")
        DoCommand.run()
    except (KeyboardInterrupt):
        print()  # Add an extra line because various thing terminate with \r
        sys.exit(1)
    except Exception as e:
        log.exception(e)
        sys.exit(1)
