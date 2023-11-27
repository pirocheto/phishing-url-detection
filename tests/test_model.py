def test_training(sample_data, model):
    X_train, y_train = sample_data
    model.fit(X_train, y_train)
    assert model.__sklearn_is_fitted__()


def test_predict(sample_data, pkl_model):
    X_test, _ = sample_data
    X_pred = pkl_model.predict(X_test)
    assert X_pred.shape == X_test.shape


def test_onnx_model(onnx_model):
    model_inputs = onnx_model.get_inputs()
    assert len(model_inputs) == 1

    model_input = model_inputs[0]

    assert model_input.name == "inputs"
    assert model_input.type == "tensor(string)"
    assert model_input.shape == [None]

    model_outputs = onnx_model.get_outputs()
    assert len(model_outputs) == 2

    output_label = model_outputs[0]
    assert output_label.name == "label"
    assert output_label.type == "tensor(int64)"
    assert output_label.shape == [None]

    output_proba = model_outputs[1]
    assert output_proba.name == "probabilities"
    assert output_proba.type == "tensor(float)"
    assert output_proba.shape == [None, 2]
