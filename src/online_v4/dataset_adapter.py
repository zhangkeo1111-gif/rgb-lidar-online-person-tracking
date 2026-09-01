"""Dataset-profile boundary for the shared online_v4 perception core."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    path: Path
    payload: dict

    @property
    def ready(self) -> bool:
        return self.payload.get('adapter_status') == 'ready'


def load_profile(root: Path, name: str) -> DatasetProfile:
    path = root / 'configs' / 'datasets' / f'{name}.json'
    if not path.is_file():
        raise FileNotFoundError(f'Dataset profile does not exist: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('dataset') != name:
        raise ValueError(f'Dataset profile name mismatch: requested {name!r}')
    return DatasetProfile(name, path, payload)


def require_runtime_ready(profile: DatasetProfile) -> None:
    if not profile.ready:
        raise RuntimeError(
            f'Dataset {profile.name!r} is an adapter stub; official calibration/schema '
            'must be audited before runtime use')
    priors = profile.payload.get('scene_specific_priors', {})
    if profile.name != 'scene01' and any(bool(value) for value in priors.values()):
        raise RuntimeError(
            f'Dataset {profile.name!r} may not silently enable Scene01-specific priors')

