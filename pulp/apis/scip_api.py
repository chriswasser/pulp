# PuLP : Python LP Modeler
# Version 1.4.2

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

from typing import Dict, List, Optional, Tuple, Union
from .core import LpSolver_CMD, subprocess, PulpSolverError
from .core import scip_path, fscip_path
import os
from .. import constants
import sys


class SCIP_CMD(LpSolver_CMD):
    """The SCIP optimization solver"""

    name = "SCIP_CMD"

    def __init__(
        self,
        path=None,
        keepFiles=False,
        msg=True,
        options=None,
        timeLimit=None,
        gapRel=None,
        gapAbs=None,
        maxNodes=None,
        logPath=None,
    ):
        """
        :param bool msg: if False, no log is shown
        :param list options: list of additional options to pass to solver
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary
        :param float timeLimit: maximum time for solver (in seconds)
        :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
        :param float gapAbs: absolute gap tolerance for the solver to stop
        :param int maxNodes: max number of nodes during branching. Stops the solving when reached.
        :param str logPath: path to the log file
        """
        LpSolver_CMD.__init__(
            self,
            msg=msg,
            options=options,
            path=path,
            keepFiles=keepFiles,
            timeLimit=timeLimit,
            gapRel=gapRel,
            gapAbs=gapAbs,
            maxNodes=maxNodes,
            logPath=logPath,
        )

    SCIP_STATUSES = {
        "unknown": constants.LpStatusUndefined,
        "user interrupt": constants.LpStatusNotSolved,
        "node limit reached": constants.LpStatusNotSolved,
        "total node limit reached": constants.LpStatusNotSolved,
        "stall node limit reached": constants.LpStatusNotSolved,
        "time limit reached": constants.LpStatusNotSolved,
        "memory limit reached": constants.LpStatusNotSolved,
        "gap limit reached": constants.LpStatusOptimal,
        "solution limit reached": constants.LpStatusNotSolved,
        "solution improvement limit reached": constants.LpStatusNotSolved,
        "restart limit reached": constants.LpStatusNotSolved,
        "optimal solution found": constants.LpStatusOptimal,
        "infeasible": constants.LpStatusInfeasible,
        "unbounded": constants.LpStatusUnbounded,
        "infeasible or unbounded": constants.LpStatusNotSolved,
    }
    NO_SOLUTION_STATUSES = {
        constants.LpStatusInfeasible,
        constants.LpStatusUnbounded,
        constants.LpStatusNotSolved,
    }

    def defaultPath(self):
        return self.executableExtension(scip_path)

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError("PuLP: cannot execute " + self.path)

        tmpLp, tmpSol, tmpOptions = self.create_tmp_files(lp.name, "lp", "sol", "set")
        lp.writeLP(tmpLp)

        file_options: List[str] = []
        if self.timeLimit is not None:
            file_options.append(f"limits/time={self.timeLimit}")
        if "gapRel" in self.optionsDict:
            file_options.append(f"limits/gap={self.optionsDict['gapRel']}")
        if "gapAbs" in self.optionsDict:
            file_options.append(f"limits/absgap={self.optionsDict['gapAbs']}")
        if "maxNodes" in self.optionsDict:
            file_options.append(f"limits/nodes={self.optionsDict['maxNodes']}")

        command: List[str] = []
        command.append(self.path)
        command.extend(["-s", tmpOptions])
        if not self.msg:
            command.append("-q")
        if "logPath" in self.optionsDict:
            command.extend(["-l", self.optionsDict["logPath"]])

        options = iter(self.options)
        for option in options:
            # identify cli options by a leading dash (-) and treat other options as file options
            if option.starts_with("-"):
                # assumption: all cli options require an argument which is provided as a separate parameter
                argument = next(options)
                command.extend([option, argument])
            else:
                # assumption: all file options require an argument which is provided after the equal sign (=)
                if "=" not in option:
                    argument = next(options)
                    option += f"={argument}"
                file_options.append(option)

        # append scip commands after parsing self.options to allow the user to specify additional -c arguments
        command.extend(["-c", f'read "{tmpLp}"'])
        command.extend(["-c", "optimize"])
        command.extend(["-c", f'write solution "{tmpSol}"'])
        command.extend(["-c", "quit"])

        with open(tmpOptions, "w") as options_file:
            options_file.write("\n".join(file_options))
        subprocess.check_call(command, stdout=sys.stdout, stderr=sys.stderr)

        if not os.path.exists(tmpSol):
            raise PulpSolverError("PuLP: Error while executing " + self.path)
        status, values = self.readsol(tmpSol)
        # Make sure to add back in any 0-valued variables SCIP leaves out.
        finalVals = {}
        for v in lp.variables():
            finalVals[v.name] = values.get(v.name, 0.0)

        lp.assignVarsVals(finalVals)
        lp.assignStatus(status)
        self.delete_tmp_files(tmpLp, tmpSol, tmpOptions)
        return status

    @staticmethod
    def readsol(filename):
        """Read a SCIP solution file"""
        with open(filename) as f:

            # First line must contain 'solution status: <something>'
            try:
                line = f.readline()
                comps = line.split(": ")
                assert comps[0] == "solution status"
                assert len(comps) == 2
            except Exception:
                raise PulpSolverError("Can't get SCIP solver status: %r" % line)

            status = SCIP_CMD.SCIP_STATUSES.get(
                comps[1].strip(), constants.LpStatusUndefined
            )
            values = {}

            if status in SCIP_CMD.NO_SOLUTION_STATUSES:
                return status, values

            # Look for an objective value. If we can't find one, stop.
            try:
                line = f.readline()
                comps = line.split(": ")
                assert comps[0] == "objective value"
                assert len(comps) == 2
                float(comps[1].strip())
            except Exception:
                raise PulpSolverError("Can't get SCIP solver objective: %r" % line)

            # Parse the variable values.
            for line in f:
                try:
                    comps = line.split()
                    values[comps[0]] = float(comps[1])
                except:
                    raise PulpSolverError("Can't read SCIP solver output: %r" % line)

            return status, values


SCIP = SCIP_CMD


class FSCIP_CMD(LpSolver_CMD):
    """The multi-threaded FiberSCIP version of the SCIP optimization solver"""

    name = "FSCIP_CMD"

    def __init__(
        self,
        path=None,
        keepFiles=False,
        msg=True,
        options=None,
        timeLimit=None,
        gapRel=None,
        gapAbs=None,
        maxNodes=None,
        threads=None,
        logPath=None,
    ):
        """
        :param bool msg: if False, no log is shown
        :param list options: list of additional options to pass to solver
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary
        :param float timeLimit: maximum time for solver (in seconds)
        :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
        :param float gapAbs: absolute gap tolerance for the solver to stop
        :param int maxNodes: max number of nodes during branching. Stops the solving when reached.
        :param int threads: sets the maximum number of threads
        :param str logPath: path to the log file
        """
        LpSolver_CMD.__init__(
            self,
            msg=msg,
            options=options,
            path=path,
            keepFiles=keepFiles,
            timeLimit=timeLimit,
            gapRel=gapRel,
            gapAbs=gapAbs,
            maxNodes=maxNodes,
            threads=threads,
            logPath=logPath,
        )

    FSCIP_STATUSES = {
        "No Solution": constants.LpStatusNotSolved,
        "Final Solution": constants.LpStatusOptimal,
    }
    NO_SOLUTION_STATUSES = {
        constants.LpStatusInfeasible,
        constants.LpStatusUnbounded,
        constants.LpStatusNotSolved,
    }

    def defaultPath(self):
        return self.executableExtension(fscip_path)

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError("PuLP: cannot execute " + self.path)

        tmpLp, tmpSol, tmpOptions, tmpParams = self.create_tmp_files(
            lp.name, "lp", "sol", "set", "prm"
        )
        lp.writeLP(tmpLp)

        file_options: List[str] = []
        if self.timeLimit is not None:
            file_options.append(f"limits/time={self.timeLimit}")
        if "gapRel" in self.optionsDict:
            file_options.append(f"limits/gap={self.optionsDict['gapRel']}")
        if "gapAbs" in self.optionsDict:
            file_options.append(f"limits/absgap={self.optionsDict['gapAbs']}")
        if "maxNodes" in self.optionsDict:
            file_options.append(f"limits/nodes={self.optionsDict['maxNodes']}")

        file_parameters: List[str] = []

        command: List[str] = []
        command.append(self.path)
        command.append(tmpParams)
        command.append(tmpLp)
        command.extend(["-s", tmpOptions])
        command.extend(["-fsol", tmpSol])
        if not self.msg:
            command.append("-q")
        if "logPath" in self.optionsDict:
            command.extend(["-l", self.optionsDict["logPath"]])
        if "threads" in self.optionsDict:
            command.extend(["-sth", f"{self.optionsDict['threads']}"])

        options = iter(self.options)
        for option in options:
            # identify cli options by a leading dash (-) and treat other options as file options
            if option.starts_with("-"):
                # assumption: all cli options require an argument which is provided as a separate parameter
                argument = next(options)
                command.extend([option, argument])
            else:
                # assumption: all file options contain a slash (/)
                is_file_options = "/" in option

                # assumption: all file options and parameters require an argument which is provided after the equal sign (=)
                if "=" not in option:
                    argument = next(options)
                    option += f"={argument}"

                if is_file_options:
                    file_options.append(option)
                else:
                    file_parameters.append(option)

        print(' '.join(command))
        # wipe the solution file since FSCIP does not overwrite it if no solution was found which causes parsing errors
        self.silent_remove(tmpSol)
        with open(tmpOptions, "w") as options_file:
            options_file.write("\n".join(file_options))
        with open(tmpParams, "w") as parameters_file:
            parameters_file.write("\n".join(file_parameters))
        subprocess.check_call(command, stdout=sys.stdout, stderr=sys.stderr)

        if not os.path.exists(tmpSol):
            raise PulpSolverError("PuLP: Error while executing " + self.path)
        status, values = self.readsol(tmpSol)
        # Make sure to add back in any 0-valued variables SCIP leaves out.
        finalVals = {}
        for v in lp.variables():
            finalVals[v.name] = values.get(v.name, 0.0)

        lp.assignVarsVals(finalVals)
        lp.assignStatus(status)
        self.delete_tmp_files(tmpLp, tmpSol, tmpOptions, tmpParams)
        return status

    @staticmethod
    def parse_status(string: str) -> Optional[int]:
        for fscip_status, pulp_status in FSCIP_CMD.FSCIP_STATUSES.items():
            if fscip_status in string:
                return pulp_status
        return None

    @staticmethod
    def parse_objective(string: str) -> Optional[float]:
        fields = string.split(":")
        if len(fields) != 2:
            return None

        label, objective = fields
        if label != "objective value":
            return None

        objective = objective.strip()
        try:
            objective = float(objective)
        except ValueError:
            return None

        return objective

    @staticmethod
    def parse_variable(string: str) -> Optional[Tuple[str, float]]:
        fields = string.split()
        if len(fields) < 2:
            return None

        name, value = fields[:2]
        try:
            value = float(value)
        except ValueError:
            return None

        return name, value

    @staticmethod
    def readsol(filename):
        """Read a FSCIP solution file"""
        with open(filename) as file:

            # First line must contain a solution status
            status_line = file.readline()
            status = FSCIP_CMD.parse_status(status_line)
            if status is None:
                raise PulpSolverError(f"Can't get FSCIP solver status: {status_line!r}")

            if status in FSCIP_CMD.NO_SOLUTION_STATUSES:
                return status, {}

            # Look for an objective value. If we can't find one, stop.
            objective_line = file.readline()
            objective = FSCIP_CMD.parse_objective(objective_line)
            if objective is None:
                raise PulpSolverError(
                    f"Can't get FSCIP solver objective: {objective_line!r}"
                )

            # Parse the variable values.
            variables: Dict[str, float] = {}
            for variable_line in file:
                variable = FSCIP_CMD.parse_variable(variable_line)
                if variable is None:
                    raise PulpSolverError(
                        f"Can't read FSCIP solver output: {variable_line!r}"
                    )

                name, value = variable
                variables[name] = value

            return status, variables


FSCIP = FSCIP_CMD
