import pandas as pd

df = pd.read_json('/CVPR23-FAS-WILD/benchmark/benchmark.json')
df.to_csv('/CVPR23-FAS-WILD/benchmark/benchmark.csv', index=None)