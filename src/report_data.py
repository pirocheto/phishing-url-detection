import dvc.api
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataQualityPreset
from evidently.report import Report

params = dvc.api.params_show()

report = Report(metrics=[DataQualityPreset()])

df = pd.read_csv(params["data_all"])
column_mapping = ColumnMapping(**params["column_mapping"])

report.run(current_data=df, reference_data=None, column_mapping=column_mapping)

report.save_html(params["report_data_html"])
