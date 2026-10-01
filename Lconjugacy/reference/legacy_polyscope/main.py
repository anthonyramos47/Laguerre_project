# Code/gui/main.py
"""Entry point for the refactored L-Conjugacy GUI."""

import argparse
import polyscope as ps
import os
#from pathlib import Path

from LconjugacyGUI.app import App
import LconjugacyGUI.computations as comp


def main():
    parser = argparse.ArgumentParser(description="L-Conjugacy GUI")
    parser.add_argument("name", nargs="?", default="helixexample",
                        help="Name of the surface file (without .json extension)")
    parser.add_argument("experiment", type=str, default="experiment0",
                        help="Experiment name")
    args = parser.parse_args()

    # Project path
    project_path = os.path.dirname(os.path.abspath(__file__))

    app = App(project_path, args.name,  experiment_name=args.experiment)
    app.init()
    comp.update_visualization(app.state)
    ps.show()


if __name__ == "__main__":
    main()