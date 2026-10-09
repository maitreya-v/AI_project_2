"""Open a graphical Pac-Man game without changing the project runner.

Examples:
    python3 play_gui.py
    python3 play_gui.py --agent AlphaBetaAgent --layout minimaxClassic --depth 2
"""

import argparse
import os
from pathlib import Path

import graphicsUtils
import pacman


def main():
    parser = argparse.ArgumentParser(description="Open Pac-Man in a separate window")
    parser.add_argument("--agent", default="ReflexAgent",
                        choices=("ReflexAgent", "MinimaxAgent", "AlphaBetaAgent"))
    parser.add_argument("--layout", default="testClassic")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--frame-time", type=float, default=0.5)
    parser.add_argument("--ghosts", type=int, default=4)
    parser.add_argument("--fixed-seed", action="store_true")
    parser.add_argument("--close-on-finish", action="store_true",
                        help="Close the graphics window when the game ends")
    options = parser.parse_args()

    # The original layout loader searches relative to the current directory.
    os.chdir(Path(__file__).resolve().parent)
    game_args = [
        "-p", options.agent,
        "-l", options.layout,
        "-n", "1",
        "-k", str(options.ghosts),
        "--frameTime", str(options.frame_time),
    ]
    if options.agent != "ReflexAgent":
        game_args.extend(("-a", "depth=%d" % options.depth))
    if options.fixed_seed:
        game_args.append("-f")

    game = pacman.readCommand(game_args)
    if not options.close_on_finish:
        display = game["display"]
        original_finish = display.finish

        def finish_after_click():
            # Ignore any clicks made during play, then keep the final board
            # visible until the user clicks the graphics window again.
            graphicsUtils._leftclick_loc = None
            graphicsUtils._rightclick_loc = None
            graphicsUtils._ctrl_leftclick_loc = None
            print("Game over. Click the Pac-Man window to close it.")
            graphicsUtils.wait_for_click()
            original_finish()

        display.finish = finish_after_click

    pacman.runGames(**game)


if __name__ == "__main__":
    main()
