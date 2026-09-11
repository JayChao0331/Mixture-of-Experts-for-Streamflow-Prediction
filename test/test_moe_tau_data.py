"""Keep prepared CDEC inputs consistent with the MoE-tau configuration."""
from pathlib import Path

import pytest
import torch

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config


@pytest.mark.parametrize('use_basin_id', [False, True])
def test_cdec_basin_encoding_follows_configuration(tmp_path, use_basin_id):
    root = Path(__file__).resolve().parents[1]
    config = Config(root / 'runs' / 'moe_tau.yml')
    basin_file = root / 'cdec_nondet_basin.txt'
    basin_ids = basin_file.read_text().split()
    config.update_config({
        'data_dir': root / 'processed_data',
        'train_basin_file': basin_file,
        'train_dir': tmp_path,
        'train_start_date': '01/01/2000',
        'train_end_date': '07/01/2000',
        'validation_start_date': '08/01/2000',
        'validation_end_date': '14/01/2000',
        'seq_length': 7,
        'use_basin_id_encoding': use_basin_id,
        'number_of_basins': len(basin_ids),
        'verbose': 0,
    })

    training = get_dataset(config, is_train=True, period='train')
    sample = training[0]
    assert ('x_one_hot' in sample) == use_basin_id
    batch = {key: value.unsqueeze(0) for key, value in sample.items() if isinstance(value, torch.Tensor)}
    layer = InputLayer(config)
    assert layer(batch).shape[-1] == layer.output_size

    # Evaluation must honor the supplied mapping, including a non-default order.
    mapping = {basin: index for index, basin in enumerate(reversed(basin_ids))}
    validation = get_dataset(config, is_train=False, period='validation', basin=basin_ids[0],
                             scaler=training.scaler, id_to_int=mapping)
    sample = validation[0]
    assert ('x_one_hot' in sample) == use_basin_id
    if use_basin_id:
        assert sample['x_one_hot'].argmax().item() == mapping[basin_ids[0]]
