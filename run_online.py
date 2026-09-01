"""Single command-line entry point for the online Scene01 pipeline."""
from __future__ import annotations
import sys
import subprocess
from importlib.util import find_spec
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def _runtime_python(argv: list[str]) -> tuple[Path | None, list[str]]:
    """Resolve an optional/bootstrap interpreter without a shell wrapper."""
    forwarded = list(argv)
    selected: Path | None = None
    if '--python' in forwarded:
        index = forwarded.index('--python')
        if index + 1 == len(forwarded):
            raise SystemExit('--python requires an interpreter path')
        selected = Path(forwarded[index + 1]).expanduser()
        del forwarded[index:index + 2]
    elif find_spec('rosbags') is None:
        candidates = [ROOT / '.venv' / 'Scripts' / 'python.exe', Path('D:\\navwareset_scene01_clean\\.venv\\Scripts\\python.exe')]
        selected = next((path for path in candidates if path.is_file()), None)
    return (selected, forwarded)

def main() -> None:
    selected, argv = _runtime_python(sys.argv[1:])
    if selected is not None and selected.resolve() != Path(sys.executable).resolve():
        raise SystemExit(subprocess.call([str(selected), str(Path(__file__).resolve()), *argv]))
    sys.path.insert(0, str(ROOT / 'src'))
    try:
        from online_v4.runtime import parser, run
    except ModuleNotFoundError as exc:
        raise SystemExit(f'Missing dependency {exc.name!r}. Install requirements.txt or pass --python PATH_TO_PYTHON.') from exc
    cli = parser()
    cli.add_argument('--python', metavar='PATH', help='Run with the specified Python interpreter (handled before argument parsing)')
    output = ROOT / 'outputs' / 'online_v4_coco_person'
    cli.set_defaults(output_dir=str(output), output_jsonl=str(output / 'online_frames.jsonl'),
                     record=str(output / 'online_person_cylinders.mp4'), no_record=True,
                     persistent_identity=False, person_only_lightweight=True)
    args = cli.parse_args(argv)
    if args.detector == 'yolo26s-coco':
        branch_output = ROOT / 'outputs' / 'online_v4_yolo26s_coco_person'
        if '--output-dir' not in argv:
            args.output_dir = str(branch_output)
        if '--output-jsonl' not in argv:
            args.output_jsonl = str(branch_output / 'online_frames.jsonl')
        if '--record' not in argv:
            args.record = str(branch_output / 'online_person_cylinders.mp4')
    if '--record' in argv:
        args.no_record = False
    run(args)
if __name__ == '__main__':
    main()
