import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import List


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _venv_python() -> Path:
    v = _project_root() / ".venv"
    return v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _in_venv() -> bool:
    # Works for both venv and virtualenv
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("VIRTUAL_ENV")
    )


def _maybe_reexec_into_venv():
    vp = _venv_python()
    # If .venv exists and we are not running inside it, re-exec the CLI under .venv/python
    if vp.is_file() and not _in_venv():
        os.execv(str(vp), [str(vp), "-m", "src.cli.main", *sys.argv[1:]])


_maybe_reexec_into_venv()

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.cli.environment import check_environment, run_environment_wizard


STAGE_MODULES = {
    "s1": "src.s1_data.stage",
    "s2": "src.s2_align.stage",
    "s3": "src.s3_degrade.stage",
    "s4_gfpgan": "src.s4_gfpgan.stage",
    "s4_codeformer": "src.s4_codeformer.stage",
    "s5": "src.s5_metrics.stage",
    "s6": "src.s6_figures.stage",
    "s7": "src.s7_logging.stage",
}

STAGE_ORDER: List[str] = [
    "s1",
    "s2",
    "s3",
    "s4_gfpgan",
    "s4_codeformer",
    "s5",
    "s6",
    "s7",
]

STAGE_LABELS = {
    "s1": "S1 — Data ingestion and verification",
    "s2": "S2 — Alignment verification",
    "s3": "S3 — Synthetic degradation",
    "s4_gfpgan": "S4A — GFPGAN inference",
    "s4_codeformer": "S4B — CodeFormer inference",
    "s5": "S5 — Metrics computation",
    "s6": "S6 — Figure generation",
    "s7": "S7 — Run manifest and provenance",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Face restoration pipeline CLI (GFPGAN vs CodeFormer).",
        add_help=True,
    )
    parser.add_argument(
        "--stage",
        choices=["all"] + STAGE_ORDER,
        help=(
            "Optional non-interactive mode. "
            "Use 'all' to run full S1–S7, or choose a single stage like 's3'. "
            "If omitted, an interactive menu is shown."
        ),
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip environment check in non-interactive mode.",
    )
    return parser.parse_args(argv)


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def pause() -> None:
    input("\nPress Enter to return to the menu...")


def run_stage(stage_name: str, config) -> None:
    logger = get_logger("CLI")
    if stage_name not in STAGE_MODULES:
        logger.error("Unknown stage: %s", stage_name)
        raise SystemExit(1)

    mod_path = STAGE_MODULES[stage_name]
    try:
        mod = importlib.import_module(mod_path)
        mod = importlib.reload(mod)
        # Reload sibling utils if already imported to avoid stale helpers
        pkg_name = mod_path.rsplit(".", 1)[0] + ".utils"
        if pkg_name in sys.modules:
            importlib.reload(sys.modules[pkg_name])
    except Exception as e:
        logger.error("Failed to load stage '%s': %s", stage_name, e)
        raise SystemExit(1)

    logger.info("=== START %s ===", stage_name.upper())
    getattr(mod, "run")(config)
    logger.info("=== END   %s ===", stage_name.upper())


def run_full_pipeline(config) -> None:
    logger = get_logger("CLI")
    logger.info("Running full pipeline: %s", ", ".join(STAGE_ORDER))
    for stage_name in STAGE_ORDER:
        run_stage(stage_name, config)
    logger.info("Full pipeline completed.")


def clear_workspace(preserve_logs: bool = True) -> None:
    logger = get_logger("CLI")
    logger.info("Clearing workspace (results/*). preserve_logs=%s", preserve_logs)

    results_root = "results"
    if not os.path.isdir(results_root):
        logger.info("No 'results/' directory found. Nothing to clear.")
        return

    targets = ["outputs", "tables", "figures"]
    if not preserve_logs:
        targets.append("logs")

    for name in targets:
        path = os.path.join(results_root, name)
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info("Removed '%s'.", path)
        os.makedirs(path, exist_ok=True)
        logger.info("Recreated '%s'.", path)

    logger.info("Workspace cleared.")


def show_logs(lines: int = 50) -> None:
    logger = get_logger("CLI")
    log_path = os.path.join("results", "logs", "pipeline.log")

    print("\n--- Last %d log lines ---" % lines)
    if not os.path.isfile(log_path):
        print("(No log file found at %s)" % log_path)
        logger.info("No log file found when attempting to show logs.")
        return

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except OSError as e:
        print("(Failed to read log file: %s)" % e)
        logger.error("Failed to read log file: %s", e)
        return

    for line in all_lines[-lines:]:
        print(line.rstrip())


def show_config_summary(config) -> None:
    print("\n--- Config summary ---")
    project_name = config.get("project_name", "unknown")
    print(f"Project: {project_name}")

    det = config.get("determinism", {})
    if det:
        print("\nDeterminism:")
        print(f"  seed: {det.get('seed', 'n/a')}")
        print(f"  numpy_seed: {det.get('numpy_seed', 'n/a')}")
        print(f"  python_hash_seed: {det.get('python_hash_seed', 'n/a')}")

    experiments = config.get("experiments", {}).get("matrix", {})
    methods = experiments.get("methods", [])
    degradations = experiments.get("degradations", [])
    codeformer_ws = experiments.get("codeformer_fidelity_w", [])

    print("\nExperiment matrix:")
    print(f"  Methods: {', '.join(methods) or 'n/a'}")
    print(f"  Degradations: {', '.join(degradations) or 'n/a'}")
    print(
        "  CodeFormer w values: "
        f"{', '.join(str(w) for w in codeformer_ws) or 'n/a'}"
    )

    print("\nEnvironment:")
    print("  Managed via env.yml (conda) or requirements.txt (pip).")


def interactive_menu() -> int:
    logger = get_logger("CLI")
    logger.info("Starting interactive CLI menu.")
    config = load_config("config.json")

    while True:
        clear_screen()
        print("Face Restoration Pipeline — CLI")
        print("--------------------------------")
        print("[1] Run full pipeline (S1–S7)")
        print("[2] Run single stage")
        print("[3] Show config summary")
        print("[4] Show last 50 log lines")
        print("[5] Clear workspace (keep logs)")
        print("[6] Clear workspace (everything, including logs)")
        print("[7] Environment wizard (check + setup instructions)")
        print("[0] Exit")
        print("--------------------------------")

        choice = input("Select an option: ").strip()

        if choice == "1":
            clear_screen()
            ok, _ = check_environment()
            if not ok:
                print(
                    "Environment check FAILED. Use option [7] to see setup steps "
                    "for venv/requirements or conda/env.yml."
                )
                pause()
                continue
            run_full_pipeline(config)
            pause()
        elif choice == "2":
            clear_screen()
            print("Available stages:\n")
            for idx, stage_name in enumerate(STAGE_ORDER, start=1):
                label = STAGE_LABELS.get(stage_name, stage_name)
                print(f"[{idx}] {label}")
            print("[0] Back\n")
            raw = input("Select a stage: ").strip()

            if raw == "0":
                continue

            try:
                idx = int(raw)
            except ValueError:
                print("\nInvalid selection.")
                pause()
                continue

            if not (1 <= idx <= len(STAGE_ORDER)):
                print("\nInvalid selection.")
                pause()
                continue

            stage_name = STAGE_ORDER[idx - 1]
            clear_screen()
            ok, _ = check_environment()
            if not ok:
                print(
                    "Environment check FAILED. Use option [7] to see setup steps "
                    "for venv/requirements or conda/env.yml."
                )
                pause()
                continue

            print(f"Running {STAGE_LABELS.get(stage_name, stage_name)}\n")
            run_stage(stage_name, config)
            pause()
        elif choice == "3":
            clear_screen()
            show_config_summary(config)
            pause()
        elif choice == "4":
            clear_screen()
            show_logs(lines=50)
            pause()
        elif choice == "5":
            clear_screen()
            confirm = input(
                "Clear workspace (outputs/tables/figures), keep logs? [y/N]: "
            ).strip().lower()
            if confirm == "y":
                clear_workspace(preserve_logs=True)
            else:
                print("Cancelled.")
            pause()
        elif choice == "6":
            clear_screen()
            confirm = input(
                "Clear entire workspace including logs? [y/N]: "
            ).strip().lower()
            if confirm == "y":
                clear_workspace(preserve_logs=False)
            else:
                print("Cancelled.")
            pause()
        elif choice == "7":
            clear_screen()
            run_environment_wizard()
            pause()
        elif choice == "0":
            logger.info("Exiting interactive CLI menu.")
            return 0
        else:
            print("\nInvalid selection.")
            pause()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Non-interactive mode
    if argv:
        args = parse_args(argv)
        logger = get_logger("CLI")

        logger.info("Loading configuration from config.json")
        config = load_config("config.json")

        if not args.skip_env_check:
            ok, _ = check_environment()
            if not ok:
                logger.error(
                    "Environment check failed. "
                    "Use 'python -m src.cli.main' and menu option [7] "
                    "to see setup steps for venv/requirements or conda/env.yml."
                )
                return 1

        if args.stage is None or args.stage == "all":
            run_full_pipeline(config)
        else:
            run_stage(args.stage, config)

        logger.info("Pipeline execution completed.")
        return 0

    # Interactive menu if no args
    return interactive_menu()


if __name__ == "__main__":
    raise SystemExit(main())
