# template.py
from pathlib import Path


PROJECT_STRUCTURE = {
    "src": {
        "sac": ["__init__.py", "sac_agent.py", "models.py"],
        "discrete_sac": ["__init__.py", "discrete_sac_agent.py", "models.py"],
        "pebble": [
            "__init__.py",
            "pebble_agent.py",
            "reward_model.py",
            "preference_buffer.py",
            "teacher.py",
        ],
        "rewards": [
            "__init__.py",
            "pendulum_rewards.py",
            "lunar_lander_rewards.py",
            "reacher_rewards.py",
        ],
        "train": [
            "__init__.py",
            "train_sac.py",
            "train_discrete_sac.py",
            "train_pebble.py",
        ],
        "files": [
            "replay_buffer.py",
            "env_wrappers.py",
            "utils.py",
            "experiments.py",
        ],
    },
    "experiments": [
        "pendulum",
        "lunar_lander_continuous",
        "lunar_lander_discrete",
        "reacher",
        "pebble",
    ],
    "notebooks": ["report_and_analysis.ipynb"],
    "figures": [],
    "root_files": ["README.md", "requirements.txt", ".gitignore"],
}


def create_file(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def create_structure(base_dir: Path) -> None:
    # Root files
    for filename in PROJECT_STRUCTURE["root_files"]:
        create_file(base_dir / filename)

    # src/
    src_dir = base_dir / "src"
    src_dir.mkdir(exist_ok=True)

    for folder_name, files in PROJECT_STRUCTURE["src"].items():
        if folder_name == "files":
            for filename in files:
                create_file(src_dir / filename)
        else:
            folder = src_dir / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            for filename in files:
                create_file(folder / filename)

    # other top-level folders
    for folder_name in ["experiments", "notebooks", "figures"]:
        (base_dir / folder_name).mkdir(exist_ok=True)

    # experiments subfolders
    exp_dir = base_dir / "experiments"
    for subfolder in PROJECT_STRUCTURE["experiments"]:
        (exp_dir / subfolder).mkdir(parents=True, exist_ok=True)

    # notebook file
    for filename in PROJECT_STRUCTURE["notebooks"]:
        create_file(base_dir / "notebooks" / filename)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    create_structure(base)
    print(f"Project structure created in: {base}")