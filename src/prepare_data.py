import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "car_data"
DST = ROOT / "data" / "classification"
VAL_RATIO = 0.15
random.seed(42)


def find_split(name):
    for p in SRC.rglob(name):
        if p.is_dir() and any(c.is_dir() for c in p.iterdir()):
            return p
    raise SystemExit(f"Could not find a '{name}' folder with class subfolders in {SRC}")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    train_src, test_src = find_split("train"), find_split("test")

    for name, src in (("train", train_src), ("test", test_src)):
        target = DST / name
        if target.exists():
            raise SystemExit(f"{target} already exists, aborting to avoid overwriting")
        shutil.move(str(src), str(target))

    for csv in ROOT.glob("*.csv"):
        shutil.move(str(csv), str(DST / csv.name))

    val_dir = DST / "val"
    n_val = 0
    for cls in sorted((DST / "train").iterdir()):
        if not cls.is_dir():
            continue
        files = sorted(f for f in cls.iterdir() if f.is_file())
        random.shuffle(files)
        k = max(1, int(len(files) * VAL_RATIO))
        (val_dir / cls.name).mkdir(parents=True, exist_ok=True)
        for f in files[:k]:
            shutil.move(str(f), str(val_dir / cls.name / f.name))
            n_val += 1

    n_train = sum(1 for _ in (DST / "train").rglob("*.jpg"))
    n_test = sum(1 for _ in (DST / "test").rglob("*.jpg"))
    print(f"train: {n_train} | val: {n_val} | test: {n_test}")
    print(f"classes: {len(list((DST / 'train').iterdir()))}")
    print("You can now delete the empty car_data folder.")


if __name__ == "__main__":
    main()