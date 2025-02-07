from typing import List, Tuple
import os
import sys
import json
import glob
import flwr as fl
from flwr.common import Metrics
import flbenchmark.datasets
from sklearn.metrics import roc_auc_score

config = json.load(open(sys.argv[1], 'r'))

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if config['dataset'] != 'reddit' and config['dataset'] != 'femnist' and config['dataset'] != 'celeba' and config['dataset'] != 'student_horizontal' and config['dataset'] != 'vehicle_scale_horizontal':
        labels = []
        logits = []
        for _, m in metrics:
            metric = json.loads(m["target_metric"])
            labels += metric["labels"]
            logits += metric["logits"]
        return {"target_metric": roc_auc_score(labels, logits)}
    else:
        # Multiply accuracy of each client by number of examples used
        target_metrics = [num_examples * m["target_metric"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"target_metric": sum(target_metrics) / sum(examples)}

# load dataset
flbd = flbenchmark.datasets.FLBDatasets('~/flbenchmark.working/data')
val_dataset = None
if config['dataset'] == 'reddit':
    train_dataset, test_dataset, val_dataset = flbd.leafDatasets(config['dataset'])
elif config['dataset'] == 'femnist':
    train_dataset, test_dataset = flbd.leafDatasets(config['dataset'])
else:
    train_dataset, test_dataset = flbd.fateDatasets(config['dataset'])
train_data_base = '~/flbenchmark.working/data/'+config['dataset']+'_train'
test_data_base = '~/flbenchmark.working/data/'+config['dataset']+'_test'
val_data_base = '~/flbenchmark.working/data/'+config['dataset']+'_val'
flbenchmark.datasets.convert_to_csv(train_dataset, out_dir=train_data_base)
if test_dataset is not None:
    flbenchmark.datasets.convert_to_csv(test_dataset, out_dir=test_data_base)
if val_dataset is not None:
    flbenchmark.datasets.convert_to_csv(val_dataset, out_dir=val_data_base)

client_num = len(glob.glob(os.path.join(os.path.expanduser('~'), f'flbenchmark.working/data/{config["dataset"]}_train/*.csv')))
fit_client_num = int(config['training_param']['client_per_round'])
# Define strategy
strategy = fl.server.strategy.FedAvg(fraction_fit=config['training_param']['client_per_round']/client_num, fraction_evaluate=1.0, min_fit_clients=fit_client_num, min_evaluate_clients=client_num, min_available_clients=client_num, evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
fl.server.start_server(
    server_address=f"{sys.argv[3]}:8080",
    config=fl.server.ServerConfig(num_rounds=config['training_param']['epochs']),
    strategy=strategy,
)
