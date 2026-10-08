from concurrent.futures import ThreadPoolExecutor

import argparse, glob, os, torch, tqdm

# column of m(l+l-b0b1) in the tensor processor's feature layout, used as the reco-level stand-in for m(tt)
MLLBB_COLUMN = 18


def list_files(directories: list):
    """
    List the processor output files to combine, in a fixed order.

    Args:
        directories: directories containing the processor's *.p files (e.g. .../nominal/to_train)
    Returns:
        sorted list of file paths
    """
    files = []
    for directory in directories:
        found = sorted(glob.glob(os.path.join(directory, '*.p')))
        if len(found) == 0:
            raise FileNotFoundError(f'no *.p files found in {directory}')
        files += found
    return files


def load_file(path: str):
    """
    Load one processor output file.

    Args:
        path: path to a saved TensorDataset (or Subset of one) of (features, coefficients, ids)
    Returns:
        features, coefficients, ids
    """
    try:
        return torch.load(path, weights_only=False)[:]
    except Exception as error:
        raise RuntimeError(f'failed to load {path}') from error


def combine(files: list, max_mllbb: float = 2000.0, workers: int = 1):
    """
    Concatenate processor output files into a single dataset.

    Args:
        files: files to combine, in the order their events should appear
        max_mllbb: keep events with m(llbb) <= max_mllbb (inf keeps every event)
        workers: number of threads used to load files; may help on network filesystems where loading is latency-bound
    Returns:
        TensorDataset of (features, coefficients, ids)
    """
    feats, coefs, ids = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for temp_feats, temp_coefs, temp_ids in tqdm.tqdm(pool.map(load_file, files), total=len(files)):
            feats += [temp_feats]
            coefs += [temp_coefs]
            ids += [temp_ids]
    feats = torch.vstack(feats)
    coefs = torch.vstack(coefs)
    ids = torch.vstack(ids)
    print(f'{feats.shape[0]:,} events in {len(files):,} files')

    mask = feats[:, MLLBB_COLUMN] <= max_mllbb
    print(f'{int(mask.sum()):,} events after m(llbb) <= {max_mllbb} filter')
    return torch.utils.data.TensorDataset(feats[mask], coefs[mask], ids[mask])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='combine tensor processor output files into one dataset')
    parser.add_argument('directories', nargs='+', help='directories of processor *.p files, e.g. .../nominal/to_train')
    parser.add_argument('--out', '-o', required=True, help='output file, e.g. .../nominal/train.p')
    parser.add_argument('--max-mllbb', type=float, default=2000.0, help='m(llbb) upper cut; inf disables it')
    parser.add_argument(
        '--workers', '-j', type=int, default=1, help='threads used to load files (try >1 on network filesystems)'
    )
    args = parser.parse_args()

    dataset = combine(list_files(args.directories), args.max_mllbb, args.workers)
    torch.save(dataset, args.out)
    print(f'saved {args.out}')
