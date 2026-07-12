"""Trimmed, dependency-light port of cpoker.iparser (from
~/Library/CloudStorage/OneDrive-UvA/scriptarium/pylib/pylib/cpoker/iparser.py),
used to correctly read CP2K's `&CELL ABC ...` value out of a real production
input deck.

Why this exists instead of a flat regex over cp2k.inp: production CP2K
inputs (see the original mlp_path/test/dft/cp2k.inp + force_eval.inp) split
the deck across multiple files via `@include`, and substitute values through
`@SET` variables - the &CELL section commonly lives in an @include'd file,
not the top-level input. A regex that only scans the top-level file (the
first version of legacy_cp2k.py::_parse_outputs had exactly this) silently
misses the cell entirely whenever that's the case. This follows `@include`
recursively and resolves `@SET` substitutions, same as the original.

Deliberately NOT vendored from cpoker: `genfun.py` (pulled in only for
`unconv()`'s grep/awk convergence check, which legacy_cp2k.py's
`_parse_outputs` no longer uses - see its docstring) and everything else in
that package (plotting, cube-file reading, block averaging, etc. - unrelated
to CP2K input parsing and would drag in `numba`/`uncertainties` as
dependencies of this package for no reason).
"""

from __future__ import annotations

import os
import re
from pathlib import Path


class CP2KInputParser:
    """Reads @SET/@include-aware key=value pairs out of a CP2K input deck.

    Only tracks the keys legacy_cp2k.py's _parse_outputs actually needs
    (currently just ABC, the cell dimensions) - add more to `alias()` /
    `set_dict()` if another value needs the same @SET/@include-aware lookup.
    """

    def __init__(self, loc, input_file="cp2k.inp"):
        self.loc = str(loc)
        self.input_file = input_file

        if not os.path.exists(self.loc):
            raise FileNotFoundError(f"Path {self.loc} does not exist")
        if not os.path.exists(os.path.join(self.loc, self.input_file)):
            raise FileNotFoundError(f"{os.path.join(self.loc, self.input_file)} does not exist")

        self.def_var = {}
        self._set_dict()
        self._read(os.path.join(self.loc, self.input_file))

    def _set_dict(self):
        self.val_dict = {"ABC": "0 0 0"}
        self.alias_dict = {"ABC": "ABC"}

    def _read(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            try:
                line = line.split("!")[0]
                line = line.split("#")[0]
                if "@SET" in line:
                    var, val = self._get_var(line)
                    self.def_var[var] = val
                elif "@include" in line:
                    include_file = line.split()[-1].strip("\n")
                    self._read(os.path.join(self.loc, include_file))
                elif line.split()[0].strip("\n") in self.alias_dict:
                    var = line.split()[0].strip("\n")
                    val_num = len(str(self.val_dict[self.alias_dict[var]]).split(" "))
                    val = " ".join(line.strip("\n").split()[-1 * val_num:])
                    if "!" in var or "#" in var:
                        continue
                    if "$" in val:
                        parts = val.split(" ")
                        for i, word in enumerate(parts):
                            if "{" not in word:
                                parts[i] = f"{{{re.sub('[^A-Za-z0-9]+', '', word)}}}"
                        val = " ".join(parts).format(**self.def_var)
                        val = re.sub("[^A-Za-z0-9 .-]+", "", val)
                    self.val_dict[self.alias_dict[var]] = val
            except IndexError:
                pass

    @staticmethod
    def _get_var(line):
        parts = line.split()
        return parts[-2].strip("\n"), parts[-1].strip("\n")


def read_cell_abc(struct_dir: Path, input_file: str = "cp2k.inp", default=(20.0, 20.0, 20.0)) -> list:
    """Return [A, B, C] cell lengths for the CP2K deck in struct_dir, or
    `default` if the deck can't be found/parsed (e.g. hand-written test
    inputs that don't define ABC at all).
    """
    try:
        parsed = CP2KInputParser(str(struct_dir), input_file=input_file)
        return [float(x) for x in parsed.val_dict["ABC"].split()]
    except Exception:
        return list(default)
