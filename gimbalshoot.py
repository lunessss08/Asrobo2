"""Compatibility entry point. Implementation lives in robo_app.aim."""

from robo_app.aim import *  # Preserve the original public imports.
from robo_app.aim.cli import main


if __name__ == "__main__":
    main()
