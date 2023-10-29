import dvc.api
import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset
from evidently.report import Report

params = dvc.api.params_show()

cls_report = Report(metrics=[ClassificationPreset()])

df_test = pd.read_csv(params["data_test_pred"])
df_train = pd.read_csv(params["data_train_pred"])

column_mapping = ColumnMapping(**params["column_mapping"])

cls_report.run(
    current_data=df_test,
    reference_data=None,
    column_mapping=column_mapping,
)

cls_report.save_html(params["report_cls_html"])
metrics = cls_report.as_dict()["metrics"][0]["result"]["current"]
metrics = {key: value.item() for key, value in metrics.items()}

with open(params["metrics"], "w", encoding="utf8") as fp:
    yaml.dump(metrics, fp)
