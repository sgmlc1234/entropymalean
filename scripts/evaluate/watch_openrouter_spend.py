"""Stop the OpenRouter-billed cells before the credit limit is reached.

A cell that runs out of credit mid-episode does not fail cleanly: the request
comes back an error, the transport records an empty reply, and the episode is
filed as `generator_empty` -- indistinguishable at a glance from a dead server,
and three in a row abort the cell. Stopping while there is still credit leaves
every episode either measured or absent, which `--resume` can pick up.
"""
import json, os, subprocess, time, urllib.request
from pathlib import Path

#: parents[2], not parents[1]: this file moved from scripts/ into
#: scripts/evaluate/ and the depth moved with it. The failure was a
#: FileNotFoundError at startup, which is the good case -- the same slip in a
#: path that is only read on a branch would have gone unnoticed.
CONFIG = Path(__file__).resolve().parents[2] / "config/exam_cells.json"
LIMIT_RESERVE = float(os.environ.get("SPEND_RESERVE", "8"))
POLL_SECONDS = int(os.environ.get("SPEND_POLL", "120"))
#: Five quiet polls, not one. A single empty poll is a cell being restarted
#: between arms; leaving on it is how an earlier version of this watchdog went
#: away while two cells were still spending.
IDLE_POLLS_BEFORE_EXIT = 5


def patterns():
    """Every cell billed to OpenRouter, read from the config rather than fixed.

    The hardcoded list this replaces named four cells. Cells were added and
    withdrawn around it until it named one that no longer existed and three that
    had finished, so it saw an idle panel and exited -- while two other
    OpenRouter cells kept running.
    """
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    return tuple(
        f"model-label {name}"
        for name, budget in (cfg.get("budgets") or {}).items()
        if "openrouter.ai" in str(budget.get("url") or "")
    )


def usage():
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API") or ""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}
    )
    data = (json.load(urllib.request.urlopen(req, timeout=30)) or {}).get("data") or {}
    return float(data.get("usage") or 0), (float(data["limit"]) if data.get("limit") else None)


def alive(pats):
    total = 0
    for pattern in pats:
        found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        total += len(found.stdout.split())
    return total


def main():
    idle = 0
    while True:
        pats = patterns()          # re-read: cells are added mid-campaign
        try:
            spent, limit = usage()
        except Exception as exc:
            print(f"[watch] usage query failed: {exc}", flush=True)
            time.sleep(POLL_SECONDS)
            continue
        running = alive(pats)
        left = (limit - spent) if limit else float("inf")
        print(
            f"[watch] ${spent:.2f}"
            + (f" / ${limit:.2f} (${left:.2f} left)" if limit else "")
            + f" · {running} of {len(pats)} cells",
            flush=True,
        )
        if running == 0:
            idle += 1
            if idle >= IDLE_POLLS_BEFORE_EXIT:
                print("[watch] no cells for five polls, exiting", flush=True)
                return
        else:
            idle = 0
            if limit and left <= LIMIT_RESERVE:
                print(
                    f"[watch] ${left:.2f} left, at or under the "
                    f"${LIMIT_RESERVE:.2f} reserve — stopping cells",
                    flush=True,
                )
                for pattern in pats:
                    subprocess.run(["pkill", "-TERM", "-f", pattern])
                return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
