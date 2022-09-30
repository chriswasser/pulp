# PuLP : Python LP Modeler
# Version 2.4

# Copyright (c) 2002-2005, Jean-Sebastien Roy (js@jeannot.org)
# Modifications Copyright (c) 2007- Stuart Anthony Mitchell (s.mitchell@auckland.ac.nz)
# $Id:solvers.py 1791 2008-04-23 22:54:34Z smit023 $

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""

# Modified by Sam Mathew (@samiit on Github)
# Users would need to install HiGHS on their machine and provide the path to the executable. Please look at this thread: https://github.com/ERGO-Code/HiGHS/issues/527#issuecomment-894852288
# More instructions on: https://www.highs.dev

from typing import List

from .core import LpSolver_CMD, subprocess, PulpSolverError
import os, sys
from .. import constants
from .. import pulp as pl


class HiGHS_CMD(LpSolver_CMD):
    """The HiGHS_CMD solver"""

    name: str = "HiGHS_CMD"

    SOLUTION_STYLE: int = 2

    def __init__(
        self,
        path=None,
        keepFiles=False,
        mip=True,
        msg=True,
        options=None,
        timeLimit=None,
        threads=None,
    ):
        """
        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param float timeLimit: maximum time for solver (in seconds)
        :param list[str] options: list of additional options to pass to solver
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary (you can get binaries for your platform from https://github.com/JuliaBinaryWrappers/HiGHS_jll.jl/releases, or else compile from source - https://highs.dev)
        :param int threads: sets the maximum number of threads
        """
        LpSolver_CMD.__init__(
            self,
            mip=mip,
            msg=msg,
            timeLimit=timeLimit,
            options=options,
            path=path,
            keepFiles=keepFiles,
            threads=threads,
        )

    def defaultPath(self):
        return self.executableExtension("highs")

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp: pl.LpProblem):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError("PuLP: cannot execute " + self.path)
        lp.checkDuplicateVars()

        tmpMps, tmpSol, tmpOptions, tmpLog = self.create_tmp_files(
            lp.name, "mps", "sol", "HiGHS", "HiGHS_log"
        )
        lp.writeMPS(tmpMps)

        file_options: List[str] = []
        file_options.append(f"solution_file = {tmpSol}")
        file_options.append("write_solution_to_file = true")
        file_options.append(f"write_solution_style = {HiGHS_CMD.SOLUTION_STYLE}")
        if "threads" in self.optionsDict:
            file_options.append(f"threads = {self.optionsDict['threads']}")
        with open(tmpOptions, "w") as options_file:
            options_file.write("\n".join(file_options))

        command: List[str] = []
        command.append(self.path)
        command.append(tmpMps)
        command.append(f"--options_file={tmpOptions}")
        if self.timeLimit is not None:
            command.append(f"--time_limit={self.timeLimit}")
        if not self.mip:
            command.append("--solver=simplex")
        if "threads" in self.optionsDict:
            command.append("--parallel=on")
        command.extend(self.options)

        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ) as proc, open(tmpLog, "w") as log_file:
            for line in proc.stdout:
                if self.msg:
                    sys.__stdout__.write(line)
                log_file.write(line)

        # The return code for HiGHS on command line follows: 0:program ran successfully, 1: warning, -1: error - https://github.com/ERGO-Code/HiGHS/issues/527#issuecomment-946575028
        return_code = proc.wait()
        if return_code in [0, 1]:
            with open(tmpLog, "r") as log_file:
                content = log_file.readlines()
            content = [l.strip().split() for l in content]
            # LP
            model_line = [l for l in content if l[:2] == ["Model", "status"]]
            if len(model_line) > 0:
                model_status = " ".join(model_line[0][3:])  # Model status: ...
            else:
                # ILP
                model_line = [l for l in content if "Status" in l][0]
                model_status = " ".join(model_line[1:])
            sol_line = [l for l in content if l[:2] == ["Solution", "status"]]
            sol_line = sol_line[0] if len(sol_line) > 0 else ["Not solved"]
            sol_status = sol_line[-1]
            if model_status.lower() == "optimal":  # optimal
                status, status_sol = (
                    constants.LpStatusOptimal,
                    constants.LpSolutionOptimal,
                )
            elif sol_status.lower() == "feasible":  # feasible
                # Following the PuLP convention
                status, status_sol = (
                    constants.LpStatusOptimal,
                    constants.LpSolutionIntegerFeasible,
                )
            elif model_status.lower() == "infeasible":  # infeasible
                status, status_sol = (
                    constants.LpStatusInfeasible,
                    constants.LpSolutionNoSolutionFound,
                )
            elif model_status.lower() == "unbounded":  # unbounded
                status, status_sol = (
                    constants.LpStatusUnbounded,
                    constants.LpSolutionNoSolutionFound,
                )
        else:
            status = constants.LpStatusUndefined
            status_sol = constants.LpSolutionNoSolutionFound
            raise PulpSolverError("Pulp: Error while executing", self.path)

        if status == constants.LpStatusUndefined:
            raise PulpSolverError("Pulp: Error while executing", self.path)

        if not os.path.exists(tmpSol) or os.stat(tmpSol).st_size == 0:
            status_sol = constants.LpSolutionNoSolutionFound
            values = None
        else:
            values = self.readsol(lp.variables(), tmpSol)

        self.delete_tmp_files(tmpMps, tmpSol, tmpOptions, tmpLog)
        lp.assignStatus(status, status_sol)

        if status == constants.LpStatusOptimal:
            lp.assignVarsVals(values)

        return status

    @staticmethod
    def readsol(variables, filename):
        """Read a HiGHS solution file"""
        with open(filename) as f:
            content = f.readlines()
        content = [l.strip() for l in content]
        values = {}
        if not len(content):  # if file is empty, update the status_sol
            return None
        # extract everything between the line Columns and Rows
        col_id = content.index("Columns")
        row_id = content.index("Rows")
        solution = content[col_id + 1 : row_id]
        # check whether it is an LP or an ILP
        if "T Basis" in content:  # LP
            for var, line in zip(variables, solution):
                value = line.split()[0]
                values[var.name] = float(value)
        else:  # ILP
            for var, value in zip(variables, solution):
                values[var.name] = float(value)
        return values
