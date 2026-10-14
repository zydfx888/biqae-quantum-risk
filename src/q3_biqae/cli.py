import argparse
from dataclasses import replace
import json
from .config import Config, ROOT

def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent BIQAE risk and resource experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "optimize", "sweep", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--config", default=str(ROOT/"configs/mvp_4asset_16state.json"))
        if name in {"run", "optimize"}:
            command.add_argument("--with-optimization", action="store_true")
        if name == "sweep":
            command.add_argument("--grid", default=str(ROOT/"configs/tail_grid.json"))
    commands.add_parser("download-data")
    command = commands.add_parser("import-frozen")
    command.add_argument("source")
    command.add_argument("--provenance", required=True)
    commands.add_parser("demo")
    args = parser.parse_args(argv)
    if args.command == "download-data":
        from .data_download import download_french
        print(download_french())
    elif args.command == "import-frozen":
        from .data_adapter import import_frozen
        print(import_frozen(args.source, args.provenance))
    elif args.command == "demo":
        from .experiment import run
        assets = int(input("Assets [4]: ") or "4")
        dataset = input("Distribution (normal/student_t/regime/extreme/t_copula) [student_t]: ") or "student_t"
        alpha = float(input("Alpha [.975]: ") or ".975")
        epsilon = float(input("Epsilon [.01]: ") or ".01")
        method = input("Method [beta_biqae]: ") or "beta_biqae"
        cfg = Config(assets=assets, dataset="synthetic_"+dataset, alpha=alpha, epsilon=epsilon, methods=[method], seeds=[13])
        print(run(cfg, with_optimization=True))
    else:
        config = Config.load(args.config)
        if args.command == "validate":
            print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
            return
        from .experiment import run, sweep
        if args.command == "sweep":
            from pathlib import Path
            print(sweep(config, json.loads(Path(args.grid).read_text(encoding="utf-8-sig"))))
        else:
            print(run(config, with_optimization=args.command == "optimize" or args.with_optimization))

if __name__ == "__main__":
    main()

