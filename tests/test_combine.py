import pytest
import torch

from topsbi.combine import MLLBB_COLUMN, combine, list_files

from conftest import make_dataset


def write_processor_output(directory, n_files, seed=0):
    """Write files the way tensor_processor.py does: random_split Subsets of a TensorDataset."""
    directory.mkdir(parents=True)
    for k in range(n_files):
        to_train, _ = torch.utils.data.random_split(
            make_dataset(20, seed=seed + k), [0.8, 0.2], generator=torch.Generator().manual_seed(42)
        )
        torch.save(to_train, directory / f'{k:03d}.p')


def test_combine_concatenates_in_sorted_order(tmp_path):
    write_processor_output(tmp_path / 'to_train', 5)
    files = list_files([tmp_path / 'to_train'])
    assert files == sorted(files)

    combined = combine(files, max_mllbb=float('inf'), workers=2)
    expected = [torch.load(f, weights_only=False)[:] for f in files]
    for i in range(3):
        assert torch.equal(combined.tensors[i], torch.vstack([e[i] for e in expected]))


def test_combine_applies_mllbb_cut(tmp_path):
    write_processor_output(tmp_path / 'to_train', 3)
    combined = combine(list_files([tmp_path / 'to_train']), max_mllbb=100.0)
    assert len(combined) > 0
    assert (combined.tensors[0][:, MLLBB_COLUMN] <= 100.0).all()


def test_combine_multiple_directories(tmp_path):
    write_processor_output(tmp_path / 'to_train', 2)
    write_processor_output(tmp_path / 'validation', 2, seed=10)
    files = list_files([tmp_path / 'to_train', tmp_path / 'validation'])
    assert len(combine(files, max_mllbb=float('inf'))) == 4 * 16


def test_combine_reports_bad_file(tmp_path):
    write_processor_output(tmp_path / 'to_train', 2)
    (tmp_path / 'to_train' / '999.p').write_text('not a torch file')
    with pytest.raises(RuntimeError, match='999.p'):
        combine(list_files([tmp_path / 'to_train']))


def test_list_files_empty_directory(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError):
        list_files([tmp_path / 'empty'])
