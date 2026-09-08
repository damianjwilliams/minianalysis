"""
Unified command line: `python -m minianalysis <command> [args...]`, with the
commands in the order you'd actually use them:

    optimize -> run -> check

Each command forwards its remaining arguments verbatim to that submodule's
own argparse parser (so `python -m minianalysis run --help` shows run's own
options, not a generic summary) -- this dispatcher only picks which module
to run; it doesn't redeclare any of their flags.

The same three steps are also available as plain scripts in the repo root
(optimize_params.py, run_detection.py, check_events.py), for anyone who'd
rather double-click or type a filename than remember a subcommand.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

COMMANDS = {
    "launch": "Pop up a window to pick a recording and a step, then run it",
    "optimize": "Tune detection parameters live: edit a value, click a peak, see it measured",
    "run": "Detect every event in a recording and write the events CSV + params sidecar + plot",
    "check": "Click a detected event to see the windows/thresholds behind it; accept or reject it",
    "preprocess": "Bessel low-pass filter + downsample a trace on its own, checking the ABF "
                  "header's own hardware filter setting first",
}

# Command name -> module name, where they differ. `check` and `run` are named for what they do at
# the command line; their modules keep those same names, so this is only here for clarity when a
# future command needs to differ.
_MODULES = {name: name for name in COMMANDS}


def _print_top_help():
    width = max(len(n) for n in COMMANDS)
    lines = ["usage: python -m minianalysis <command> [args...]", "",
             textwrap.dedent(__doc__).strip(), "", "commands:"]
    lines += [f"    {name:<{width}}  {help_text}" for name, help_text in COMMANDS.items()]
    lines += ["", "Run `python -m minianalysis <command> --help` for that command's own options."]
    print("\n".join(lines))


def main(argv=None):
    # Deliberately NOT argparse subparsers + REMAINDER: that combination intercepts "-h"/"--help"
    # at the top-level parser instead of forwarding it to the chosen subcommand's own parser (a
    # known argparse gotcha), which would break `minianalysis <command> --help`. Plain manual
    # dispatch sidesteps it and is simpler besides.
    argv = sys.argv[1:] if argv is None else list(argv)

    if not argv or argv[0] in ("-h", "--help"):
        _print_top_help()
        sys.exit(0 if argv else 2)

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        _print_top_help()
        print(f"\npython -m minianalysis: error: unknown command {command!r}", file=sys.stderr)
        sys.exit(2)

    module = importlib.import_module(f".{_MODULES[command]}", package="minianalysis")
    module.main(rest)


if __name__ == "__main__":
    main()
