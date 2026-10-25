"""Fetch COCO validation and KITTI 2D data into the project, with resumable archives.

Downloads only the first N images (0 means all). KITTI subset mode uses HTTP
byte ranges to read selected training images from the official ZIP archive.
"""
import argparse
import json
import io
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path


class RemoteZipReader(io.RawIOBase):
    """Seekable read-only HTTP file; refuse servers that ignore byte ranges."""

    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=120) as response:
            self.size = int(response.headers['Content-Length'])
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b''
        start = self.position
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{start + size - 1}'})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                        raise IOError('Server does not honor byte ranges; use --kitti-images 0 for a full download')
                    data = response.read()
                if len(data) != size:
                    raise IOError('Incomplete range response')
                self.position += len(data)
                return data
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        print(f'Skip existing: {target}', flush=True)
        return
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = offset > 0 and response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Server returned an unexpected resume offset')
                total = int(response.headers.get('Content-Length', 0))
                received = 0
                last = time.monotonic()
                with partial.open('ab' if resumed else 'wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last > 15:
                            print(f'{target.name}: {(received + (offset if resumed else 0))/1e9:.2f} GB received', flush=True)
                            last = time.monotonic()
                if total and received != total:
                    raise IOError('Incomplete response; retry to resume')
            partial.replace(target)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


def extract_selected(archive, root, select):
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir() or not select(member.filename):
                continue
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError(f'Unsafe archive member: {member.filename}')
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.extracting')
            with source.open(member) as inp, temporary.open('wb') as out:
                shutil.copyfileobj(inp, out)
            temporary.replace(target)


def write_kitti_ids(root, names):
    ids = [Path(name).stem for name in sorted(names)]
    split = root / 'kitti/ImageSets'
    split.mkdir(parents=True, exist_ok=True)
    (split / 'train.txt').write_text('\n'.join(ids) + '\n')
    (root / 'kitti/downloaded-image-ids.json').write_text(json.dumps(ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--coco-images', type=int, default=50)
    parser.add_argument('--kitti-images', type=int, default=50)
    parser.add_argument('--datasets', nargs='+', choices=['coco', 'kitti'], default=['coco', 'kitti'])
    args = parser.parse_args()
    if args.coco_images < 0 or args.kitti_images < 0:
        parser.error('Image counts must be nonnegative')
    root = args.data_root.resolve()
    cache = root / 'downloads'
    if 'coco' in args.datasets:
        ann = root / 'coco/annotations/instances_val2017.json'
        if not ann.exists():
            archive = cache / 'annotations_trainval2017.zip'
            download('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip', archive)
            extract_selected(archive, root / 'coco', lambda name: name == 'annotations/instances_val2017.json')
        document = json.loads(ann.read_text(encoding='utf-8'))
        images = sorted(document['images'], key=lambda image: image['id'])
        images = images[:args.coco_images] if args.coco_images else images
        for image in images:
            download(f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{image['file_name']}", root / 'coco/val2017' / image['file_name'])
        (root / 'coco/downloaded-image-ids.json').write_text(json.dumps([i['id'] for i in images]))
    if 'kitti' in args.datasets:
        for component in ('label_2', 'calib', 'image_2'):
            if component == 'image_2' and args.kitti_images:
                subset_manifest = root / 'kitti/image_2-subset-manifest.json'
                if subset_manifest.exists():
                    saved = json.loads(subset_manifest.read_text())
                    if saved['requested'] == args.kitti_images and all((root / 'kitti' / n).exists() and (root / 'kitti' / n).stat().st_size == size for n, size in saved['files']):
                        print('Skip complete KITTI image subset', flush=True)
                        write_kitti_ids(root, [name for name, _ in saved['files']])
                        continue
                url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip'
                with RemoteZipReader(url) as remote, zipfile.ZipFile(remote) as source:
                    names = sorted(n for n in source.namelist() if n.startswith('training/image_2/') and n.endswith('.png'))[:args.kitti_images]
                    for index, name in enumerate(names):
                        member = source.getinfo(name)
                        target = root / 'kitti' / name
                        if target.exists() and target.stat().st_size == member.file_size:
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix('.png.extracting')
                        temporary.write_bytes(source.read(name))
                        temporary.replace(target)
                        print(f'KITTI image {index + 1}/{len(names)}: {target.name}', flush=True)
                # Explicit pilot list: use every downloaded image, without the loader's automatic 80/20 split.
                write_kitti_ids(root, names)
                subset_manifest.write_text(json.dumps({'requested': args.kitti_images, 'files': [(name, source.getinfo(name).file_size) for name in names]}))
                continue
            archive = cache / f'data_object_{component}.zip'
            # A manifest enables later runs to verify extracted files without requiring the archive.
            manifest = root / 'kitti' / f'{component}-manifest.json'
            if manifest.exists():
                entries = json.loads(manifest.read_text())
                if entries and all((root / 'kitti' / name).exists() and (root / 'kitti' / name).stat().st_size == size for name, size in entries):
                    print(f'Skip complete KITTI {component}', flush=True)
                    if component == 'image_2':
                        write_kitti_ids(root, [name for name, _ in entries])
                    continue
            download(f'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_{component}.zip', archive)
            extract_selected(archive, root / 'kitti', lambda name: name.startswith(f'training/{component}/'))
            with zipfile.ZipFile(archive) as source:
                entries = [(m.filename, m.file_size) for m in source.infolist() if not m.is_dir() and m.filename.startswith(f'training/{component}/')]
            manifest.write_text(json.dumps(entries))
            if component == 'image_2':
                write_kitti_ids(root, [name for name, _ in entries])
    print('Dataset preparation complete.', flush=True)


if __name__ == '__main__':
    main()
