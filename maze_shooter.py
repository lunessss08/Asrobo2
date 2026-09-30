"""Compatibility entry point. Implementation lives in robo_app.maze."""

from robo_app.maze import *  # Preserve the original public imports.
from robo_app.maze.cli import main


if __name__ == "__main__":
    main()
