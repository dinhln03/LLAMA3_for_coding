"""Analysis of a repository for needed Python updates."""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

from git import Repo

from neophile.analysis.base import BaseAnalyzer
from neophile.exceptions import UncommittedChangesError
from neophile.update.python import PythonFrozenUpdate

if TYPE_CHECKING:
    from pathlib import Path
    from typing import List, Optional

    from neophile.update.base import Update
    from neophile.virtualenv import VirtualEnv

__all__ = ["PythonAnalyzer"]


class PythonAnalyzer(BaseAnalyzer):
    """Analyze a tree for needed Python frozen dependency updates.

    Parameters
    ----------
    root : `pathlib.Path`
        Root of the directory tree to analyze.
    virtualenv : `neophile.virtualenv.VirtualEnv`, optional
        Virtual environment manager.
    """

    def __init__(
        self, root: Path, virtualenv: Optional[VirtualEnv] = None
    ) -> None:
        self._root = root
        self._virtualenv = virtualenv

    async def analyze(self, update: bool = False) -> List[Update]:
        """Analyze a tree and return needed Python frozen dependency updates.

        Parameters
        ----------
        update : `bool`, optional
            If set to `True`, leave the update applied.  This avoids having
            to run ``make update-deps`` twice, once to see if an update is
            needed and again to apply it properly.

        Returns
        -------
        results : List[`neophile.update.base.Update`]
            Will contain either no elements (no updates needed) or a single
            element (an update needed).

        Raises
        ------
        neophile.exceptions.UncommittedChangesError
            The repository being analyzed has uncommitted changes and
            therefore cannot be checked for updates.
        subprocess.CalledProcessError
            Running ``make update-deps`` failed.
        """
        for name in ("Makefile", "requirements/main.in"):
            if not (self._root / name).exists():
                return []
        repo = Repo(str(self._root))

        if repo.is_dirty():
            msg = "Working tree contains uncommitted changes"
            raise UncommittedChangesError(msg)

        try:
            if self._virtualenv:
                self._virtualenv.run(
                    ["make", "update-deps"],
                    cwd=str(self._root),
                    check=True,
                    capture_output=True,
                )
            else:
                subprocess.run(
                    ["make", "update-deps"],
                    cwd=str(self._root),
                    check=True,
                    capture_output=True,
                )
        except subprocess.CalledProcessError as e:
            logging.error("make update-deps failed: %s%s", e.stdout, e.stderr)
            return []

        if not repo.is_dirty():
            return []

        if not update:
            repo.git.restore(".")
        return [
            PythonFrozenUpdate(
                path=self._root / "requirements",
                applied=update,
                virtualenv=self._virtualenv,
            )
        ]

    def name(self) -> str:
        return "python"
