#!/usr/bin/env python3
"""Extract one pinned official INT8 recipe using metadata byte ranges only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import struct
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_quantizer.architectures.reference import recipe_from_header


def read_range(url, start, end, label):
    url += '?' + urllib.parse.urlencode({'download': 'true', 'recipe_range': label})
    request = urllib.request.Request(url, headers={
        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=45) as response:
        expected = f'bytes {start}-{end}/'
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
            raise ValueError('server did not honor the exact metadata byte range')
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError('metadata byte-range length mismatch')
    return data


def read_quant_configs(header, data_start, read):
    spans = sorted((v['data_offsets'][0], v['data_offsets'][1], k)
                   for k, v in header.items() if k != '__metadata__'
                   and (k.endswith('.comfy_quant') or k.endswith('.comfy_attention.config')))
    groups = []
    for start, end, name in spans:
        tensor = header[name]
        if (tensor['dtype'] != 'U8' or tensor['shape'] != [end - start]
                or not 0 <= start < end or end - start > 4096):
            raise ValueError('reference metadata must be a bounded JSON byte tensor')
        if groups and groups[-1][-1][1] == start:
            groups[-1].append((start, end, name))
        else:
            groups.append([(start, end, name)])
    allowed = {name for _, _, name in spans}
    for start, end, name in spans:
        for other, tensor in header.items():
            if other == '__metadata__' or other in allowed:
                continue
            a, b = tensor['data_offsets']
            if a < end and b > start:
                raise ValueError('metadata range overlaps a non-metadata tensor')

    def fetch(pair):
        index, group = pair
        start, end = group[0][0], group[-1][1]
        if end - start > 4 * 1024 * 1024:
            raise ValueError('reference metadata span exceeds 4 MiB')
        data = read(data_start + start, data_start + end - 1, f'quant-{index}')
        return {name[:-len('.comfy_quant')]: json.loads(data[a-start:b-start])
                for a, b, name in group if name.endswith('.comfy_quant')}

    configs = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for part in pool.map(fetch, enumerate(groups)):
            configs.update(part)
    if not configs:
        configs = json.loads(header.get('__metadata__', {}).get('_quantization_metadata', '{}')).get('layers', {})
    return configs


def extract(repo, filename, revision='main'):
    if not re.fullmatch(r'Comfy-Org/[A-Za-z0-9._-]+', repo):
        raise ValueError('expected a Comfy-Org model repository')
    if (not filename.endswith('.safetensors') or filename.startswith('/')
            or '..' in filename.split('/') or '\\' in filename):
        raise ValueError('expected a repository-relative safetensors filename')
    url = 'https://huggingface.co/api/models/' + repo + '/revision/' + urllib.parse.quote(revision, safe='')
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError('repository metadata exceeds 16 MiB')
    metadata = json.loads(data)
    revision = metadata['sha']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('expected a pinned repository revision')
    if filename not in {f['rfilename'] for f in metadata['siblings']}:
        raise ValueError('reference artifact is absent from the pinned repository')
    url = f'https://huggingface.co/{repo}/resolve/{revision}/' + urllib.parse.quote(filename, safe='/')
    read = lambda start, end, label: read_range(url, start, end, label)
    length = struct.unpack('<Q', read(0, 7, 'size'))[0]
    if not 2 <= length <= 16 * 1024 * 1024:
        raise ValueError('safetensors header size exceeds the metadata limit')
    header = json.loads(read(8, 7 + length, 'header'))
    configs = read_quant_configs(header, 8 + length, read)
    return recipe_from_header(header, configs, {'repo': repo, 'revision': revision, 'file': filename})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo')
    parser.add_argument('filename')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new recipe path')
    recipe = extract(args.repo, args.filename, args.revision)
    with args.output.open('x') as file:
        json.dump(recipe, file, indent=2)
        file.write('\n')
    print(f"Saved {len(recipe['layers'])} native INT8 layers at {recipe['source']['revision']}")


if __name__ == '__main__':
    main()
