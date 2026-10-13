"""Explicit Hugging Face component downloads pinned to a resolved revision."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from native_quantizer.errors import InputError, UsageError
from native_quantizer.files import _load_json_object


def browse_huggingface(repo: str, revision: str = 'main', subfolder: str = '') -> dict:
    from huggingface_hub import HfApi

    folder = subfolder.strip('/')
    if '..' in Path(folder).parts or '\\' in folder:
        raise ValueError('Invalid component subfolder')
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    prefix = folder + '/' if folder else ''
    files = [{'name': sibling.rfilename[len(prefix):], 'bytes': sibling.size}
             for sibling in info.siblings
             if sibling.rfilename.startswith(prefix)
             and '/' not in sibling.rfilename[len(prefix):]
             and (sibling.rfilename.endswith('.safetensors')
                  or sibling.rfilename.endswith('.safetensors.index.json'))]
    return {'repo': repo, 'commit': info.sha, 'subfolder': folder,
            'files': sorted(files, key=lambda item: item['name'])}


def download_hf_component(args: Any) -> str:
    if getattr(args, 'offline', False):
        raise UsageError('model downloads are disabled by --offline; use a local checkpoint')
    if args.model:
        raise UsageError('use a local model path or --hf-repo, not both')
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise UsageError('install huggingface-hub to use --hf-repo') from exc
    from pathlib import PurePosixPath
    folder = args.hf_subfolder.strip('/')
    for value in (folder, args.hf_filename or ''):
        if '..' in PurePosixPath(value).parts or '\\' in value or value.startswith('/'):
            raise UsageError('HF component paths must be relative and contain no parent traversal')
    api = HfApi()
    commit = api.model_info(args.hf_repo, revision=args.hf_revision).sha
    names = set(api.list_repo_files(args.hf_repo, revision=commit))
    prefix = folder + '/' if folder else ''
    local_names = sorted(n[len(prefix):] for n in names
                         if n.startswith(prefix) and '/' not in n[len(prefix):])
    if args.hf_filename:
        selected = args.hf_filename
        if '/' in selected or selected not in local_names:
            raise InputError('--hf-filename must name a file directly inside --hf-subfolder')
    else:
        indices = [n for n in local_names if n.endswith('.safetensors.index.json')]
        weights = [n for n in local_names if n.endswith('.safetensors')]
        candidates = indices or weights
        if len(candidates) != 1:
            raise InputError('choose --hf-subfolder / --hf-filename; component is ambiguous: '
                             + ', '.join(candidates[:20]))
        selected = candidates[0]
    from huggingface_hub.constants import HF_HUB_CACHE
    cache_key = hashlib.sha256((args.hf_repo + '@' + commit).encode()).hexdigest()
    local_dir = str(Path(args.hf_cache or HF_HUB_CACHE) / 'comfy-quantizer' / cache_key)
    def fetch(name: str) -> str:
        if name not in names:
            raise InputError(f'HF repository is missing {name}')
        return hf_hub_download(args.hf_repo, name, revision=commit, cache_dir=args.hf_cache, local_dir=local_dir)
    selected_path = fetch(prefix + selected)
    if selected.endswith('.safetensors.index.json'):
        index = _load_json_object(selected_path, 'HF safetensors index')
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise InputError('HF index needs a nonempty weight_map')
        for shard in sorted(set(weight_map.values())):
            if not isinstance(shard, str) or PurePosixPath(shard).name != shard or not shard.endswith('.safetensors'):
                raise InputError('HF index contains an invalid shard path')
            fetch(prefix + shard)
    elif not selected.endswith('.safetensors'):
        raise InputError('remote input must be safetensors or a safetensors index')
    if prefix + 'config.json' in names:
        fetch(prefix + 'config.json')
    args._hf_provenance = {'repo_id': args.hf_repo, 'commit': commit,
                           'subfolder': folder, 'filename': selected}
    print(f'Downloaded {args.hf_repo}@{commit[:12]} / {prefix}{selected}')
    return selected_path
