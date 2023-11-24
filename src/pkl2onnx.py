import pickle
from pathlib import Path

from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType


def pkl2onnx(model_path: str | Path = "model/model.pkl") -> str:
    model_path = Path(model_path)
    model = pickle.loads(model_path.read_bytes())

    onx = to_onnx(
        model,
        initial_types=[("inputs", StringTensorType((None,)))],
        options={"zipmap": False},
    )

    onnx_path = model_path.parent / "model.onnx"
    onnx_path.write_bytes(onx.SerializeToString())
    return onnx_path


if __name__ == "__main__":
    pkl2onnx()
