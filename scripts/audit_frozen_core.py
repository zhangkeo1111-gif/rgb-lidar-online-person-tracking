"""Verify that detector-independent v3/v4 core code stayed frozen."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = Path(r'D:\detection\versions\06_online_v3')
OUT = ROOT / 'outputs' / 'online_v4_coco_person' / 'FROZEN_CORE_AUDIT.json'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node(path: Path, class_name: str, method_name: str) -> str:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(item for item in tree.body
               if isinstance(item, ast.ClassDef) and item.name == class_name)
    method = next(item for item in cls.body
                  if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and item.name == method_name)
    return hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()


def main() -> None:
    exact_files = ('tracking.py', 'support.py', 'suppression.py', 'recovery.py',
                   'registration.py', 'identity.py')
    files = {}
    for name in exact_files:
        left = V3 / 'src' / 'online_v3' / name
        right = ROOT / 'src' / 'online_v4' / name
        files[name] = {'v3_sha256': digest(left), 'v4_sha256': digest(right),
                       'byte_identical': digest(left) == digest(right)}
    runtime_methods = ('__init__', 'correct', 'snapshot', 'mark_reported', 'update_bbox')
    runtime_nodes = {}
    for method in runtime_methods:
        left = node(V3 / 'src' / 'online_v3' / 'runtime.py', 'CausalMotionBank', method)
        right = node(ROOT / 'src' / 'online_v4' / 'runtime.py', 'CausalMotionBank', method)
        runtime_nodes[method] = {'v3_ast_sha256': left, 'v4_ast_sha256': right,
                                 'identical': left == right}
    pipeline_methods = ('project_physical', 'project_inference', 'project_display',
                        '_static_masks', '_static_evidence', '_static_keep',
                        'suppressed_candidate', 'apply_suppressed_recovery',
                        '_cylinder_geometry')
    pipeline_nodes = {}
    for method in pipeline_methods:
        left = node(V3 / 'src' / 'online_v3' / 'pipeline.py',
                    'OnlineRgbFrustumPipeline', method)
        right = node(ROOT / 'src' / 'online_v4' / 'pipeline.py',
                     'OnlineRgbFrustumPipeline', method)
        pipeline_nodes[method] = {'v3_ast_sha256': left, 'v4_ast_sha256': right,
                                  'identical': left == right}
    result = {
        'status': 'PASS' if (all(row['byte_identical'] for row in files.values())
                              and all(row['identical'] for row in runtime_nodes.values())
                              and all(row['identical'] for row in pipeline_nodes.values())) else 'FAIL',
        'exact_files': files,
        'causal_motion_bank_methods': runtime_nodes,
        'geometry_and_render_methods': pipeline_nodes,
        'expected_v4_differences': [
            'official COCO detector checkpoint and metadata-resolved PERSON class',
            'dataset profile safety boundary',
            'read-only detector/core audit fields and output paths',
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

