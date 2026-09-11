"""Checks for the retained models and their shared input layer."""
import pytest
import torch

from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.samplingutils import _SamplingSetup


def _config(**overrides):
    values = {
        'model': 'moe_tau',
        'dynamic_inputs': ['precipitation', 'temperature'],
        'static_attributes': ['elevation'],
        'target_variables': ['discharge'],
        'head': 'regression',
        'output_activation': 'linear',
        'hidden_size': 4,
        'initial_forget_bias': 1,
        'output_dropout': 0.0,
        'predict_last_n': 1,
    }
    values.update(overrides)
    return Config(values)


@pytest.mark.parametrize('model_name', ['cudalstm', 'moe_tau'])
def test_model_forward_and_backward_without_dates(model_name):
    torch.manual_seed(42)
    model = get_model(_config(model=model_name))
    data = {'x_d': torch.randn(2, 5, 2), 'x_s': torch.randn(2, 1)}

    predictions = model(data)['y_hat']
    assert predictions.shape == (2, 5, 1)
    assert torch.isfinite(predictions).all()
    predictions.square().mean().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    if hasattr(model, 'gating_net'):
        gates = model.gating_net(model.embedding_net(data))
        torch.testing.assert_close(gates.sum(dim=-1), torch.ones(2))
        assert model.gating_net.raw_scale.grad.abs().item() > 0


@pytest.mark.parametrize('model_name', ['gru', 'transformer', 'cudalstm_date', 'moe_lstm', 'unknown'])
def test_unavailable_model_reports_supported_names(model_name):
    with pytest.raises(NotImplementedError, match='Supported models:.*moe_tau'):
        get_model(_config(model=model_name))


@pytest.mark.parametrize('settings, message', [
    ({'use_frequencies': ['1D', '1h']}, 'multiple frequencies'),
    ({'autoregressive_inputs': ['discharge_shift1']}, 'autoregression'),
    ({'mass_inputs': ['precipitation']}, 'mass_inputs'),
])
def test_moe_rejects_unsupported_inputs(settings, message):
    with pytest.raises(ValueError, match=message):
        get_model(_config(**settings))


@pytest.mark.parametrize('has_statics, use_basin_id', [(False, False), (True, False), (False, True), (True, True)])
def test_input_layer_optional_statics_and_basin_ids(has_statics, use_basin_id):
    layer = InputLayer(_config(static_attributes=['elevation'] if has_statics else [],
                               use_basin_id_encoding=use_basin_id, number_of_basins=2))
    data = {'x_d': torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)}
    if has_statics:
        data['x_s'] = torch.tensor([[10.0], [20.0]])
    if use_basin_id:
        data['x_one_hot'] = torch.eye(2)

    output = layer(data)
    assert output.shape == (3, 2, layer.output_size)
    torch.testing.assert_close(output[:, :, :2], data['x_d'].transpose(0, 1))
    if has_statics:
        torch.testing.assert_close(output[:, :, 2], torch.tensor([[10.0, 20.0]]).expand(3, -1))
    if use_basin_id:
        torch.testing.assert_close(output[:, :, -2:], torch.eye(2).expand(3, -1, -1))

    dynamic, static = layer(data, concatenate_output=False)
    assert dynamic.shape == (3, 2, 2)
    assert (static is None) == (not has_statics and not use_basin_id)


@pytest.mark.parametrize('embedding_type, data_key', [('hindcast', 'x_h'), ('forecast', 'x_f')])
def test_forecast_input_layers_remain_available(embedding_type, data_key):
    config = _config(static_attributes=[], hindcast_inputs=['precipitation'], forecast_inputs=['precipitation'])
    layer = InputLayer(config, embedding_type=embedding_type)
    values = torch.randn(2, 3, 1)
    torch.testing.assert_close(layer({data_key: values}), values.transpose(0, 1))


@pytest.mark.parametrize('dropout', [0.0, 1.0])
def test_mc_dropout_validation_without_transformer_settings(dropout):
    config = _config(model='cudalstm', mc_dropout=True, output_dropout=dropout)
    model = get_model(config)
    with pytest.raises(RuntimeError, match='between 0 and 1'):
        _SamplingSetup(model, {'y': torch.zeros(2, 5, 1)}, head='regression')
