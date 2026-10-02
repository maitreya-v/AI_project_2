"""Reproduce the performance measurements used in the Project 2 report."""

import argparse
import contextlib
import gc
import io
import json
import random
import statistics
import sys
import time
import tracemalloc

import ghostAgents
import layout
import pacman
import textDisplay
from pacman import GameState
from multiAgents import AlphaBetaAgent, MinimaxAgent, ReflexAgent


def initial_state(layout_name, num_ghosts=None):
    board = layout.getLayout(layout_name)
    state = GameState()
    state.initialize(board, board.getNumGhosts() if num_ghosts is None else num_ghosts)
    return state


def benchmark_decision(agent_class, layout_name, depth=None, repeats=7,
                       memory_repeats=5, num_ghosts=None):
    """Measure time and Python allocation in separate fresh decisions."""
    timings = []
    memory_peaks = []
    decisions = []

    for _ in range(repeats):
        state = initial_state(layout_name, num_ghosts)
        kwargs = {"depth": str(depth)} if depth is not None else {}
        agent = agent_class(**kwargs)
        random.seed("cs188")
        gc.collect()
        started = time.perf_counter()
        action = agent.getAction(state)
        elapsed = time.perf_counter() - started
        timings.append(elapsed * 1000.0)
        decisions.append((action, agent._lastSearchValue,
                          agent._expandedStates, agent._expandedNodes))

    for _ in range(memory_repeats):
        state = initial_state(layout_name, num_ghosts)
        kwargs = {"depth": str(depth)} if depth is not None else {}
        agent = agent_class(**kwargs)
        random.seed("cs188")
        gc.collect()
        tracemalloc.start()
        agent.getAction(state)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_peaks.append(peak_bytes / 1024.0)

    if any(decision != decisions[0] for decision in decisions):
        raise AssertionError("non-reproducible decision for %s" % agent_class.__name__)

    action, value, expanded, generated = decisions[0]

    return {
        "agent": agent_class.__name__,
        "layout": layout_name,
        "ghosts": initial_state(layout_name, num_ghosts).getNumAgents() - 1,
        "depth": depth,
        "timing_repeats": repeats,
        "memory_repeats": memory_repeats,
        "action": action,
        "value": value,
        "expanded": expanded,
        "generated": generated,
        "runtime_ms": statistics.median(timings),
        "peak_kib": statistics.median(memory_peaks),
    }


def benchmark_games(agent_class, layout_name, games, num_ghosts, depth=None):
    """Run deterministic quiet games and summarize outcomes."""
    random.seed("cs188")
    board = layout.getLayout(layout_name)
    kwargs = {"depth": str(depth)} if depth is not None else {}
    player = agent_class(**kwargs)
    ghosts = [ghostAgents.RandomGhost(index + 1) for index in range(num_ghosts)]
    display = textDisplay.NullGraphics()

    with contextlib.redirect_stdout(io.StringIO()):
        results = pacman.runGames(
            board,
            player,
            ghosts,
            display,
            games,
            record=False,
            catchExceptions=False,
        )

    scores = [game.state.getScore() for game in results]
    wins = sum(game.state.isWin() for game in results)
    return {
        "agent": agent_class.__name__,
        "layout": layout_name,
        "depth": depth,
        "ghosts": min(num_ghosts, board.getNumGhosts()),
        "games": games,
        "wins": wins,
        "win_rate": wins / float(games),
        "average_score": statistics.mean(scores),
        "scores": scores,
    }


def collect_results(repeats):
    decision_results = [
        benchmark_decision(ReflexAgent, name, repeats=repeats,
                           num_ghosts=ghosts)
        for name, ghosts in (("testClassic", 2), ("openClassic", 2),
                             ("mediumClassic", 1), ("mediumClassic", 2))
    ]

    for depth in range(1, 5):
        decision_results.append(
            benchmark_decision(
                MinimaxAgent, "minimaxClassic", depth=depth, repeats=repeats
            )
        )
        decision_results.append(
            benchmark_decision(
                AlphaBetaAgent, "minimaxClassic", depth=depth, repeats=repeats
            )
        )

    decision_results.append(
        benchmark_decision(
            MinimaxAgent, "smallClassic", depth=2, repeats=repeats
        )
    )
    decision_results.append(
        benchmark_decision(
            AlphaBetaAgent, "smallClassic", depth=2, repeats=repeats
        )
    )
    decision_results.append(
        benchmark_decision(
            AlphaBetaAgent, "smallClassic", depth=3, repeats=repeats
        )
    )
    decision_results.append(
        benchmark_decision(
            MinimaxAgent, "smallClassic", depth=3, repeats=repeats
        )
    )

    gameplay_results = [
        benchmark_games(ReflexAgent, "testClassic", 10, 2),
        benchmark_games(ReflexAgent, "openClassic", 10, 2),
        benchmark_games(ReflexAgent, "mediumClassic", 10, 1),
        benchmark_games(ReflexAgent, "mediumClassic", 10, 2),
    ]

    return {
        "methodology": {
            "decision_repeats": repeats,
            "memory_repeats": 5,
            "runtime_statistic": "median untraced wall-clock time",
            "memory_statistic": "median separate peak Python allocation via tracemalloc",
            "expanded_definition": "nonterminal state queries to GameState.getLegalActions during one decision",
            "generated_definition": "calls to GameState.generateSuccessor during one decision",
            "gameplay_seed": "cs188",
            "python_version": sys.version.split()[0],
        },
        "decisions": decision_results,
        "gameplay": gameplay_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    results = collect_results(args.repeats)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")

    print("Wrote %s" % args.output)


if __name__ == "__main__":
    main()
