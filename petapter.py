import gc
import torch
import json
import yaml
import sys
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
import click
from loguru import logger
from transformers import XLMRobertaTokenizer, XLMRobertaModelWithHeads, XLMRobertaConfig
import numpy as np
import time
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from transformers import TrainingArguments, AdapterTrainer, PfeifferConfig, PfeifferInvConfig
import torch.nn as nn
from transformers.adapters import PredictionHead, CausalLMHead


tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', required=True)
def run(path):
    click.echo('running experiments')
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    run_all_experiments(path)


def compute_metrics(a, p):
    return {"accuracy": (p == a).mean(),
            "f1": f1_score(a, p, average='macro'),
            "precision": precision_score(a, p, average='macro'),
            "recall": recall_score(a, p, average='macro')}


def encode_batch(batch, tokenizer=tokenizer):
    return tokenizer(batch["text"], max_length=512, truncation=True, padding="max_length")


class UkrainePEThead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        id2tokenid,
        vocab_size=None,
        **kwargs
    ):
        super().__init__(head_name)
        self.config = {
            "vocab_size": model.config.vocab_size,
            "id2tokenid": {key:id2tokenid[key] for key in sorted(id2tokenid)}, # ensures sorted dict
            "id2tokenid_values": sorted(set([value for sublist in id2tokenid.values() for value in sublist])),
        }
        self.build(model)

    def build(self, model):
        model_config = model.config
        # Additional FC layers
        pred_head = []
        pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size))
        pred_head.append(nn.GELU())
        pred_head.append(nn.LayerNorm(model_config.hidden_size, eps=1e-12))
        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        # Final embedding layer
        self.add_module(
            str(len(pred_head)),
            nn.Linear(model_config.hidden_size, len(self.config["id2tokenid_values"]), bias=True),
        )

        self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent


    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        # First, pass through all layers except the last embedding layer
        seq_outputs = outputs[0]
        for i in range(len(self) - 1):
            seq_outputs = self[i](seq_outputs)

        # Now, pass through an invertible adapter if available
        inv_adapter = kwargs.pop("invertible_adapter", None)
        if inv_adapter is not None:
            seq_outputs = inv_adapter(seq_outputs, rev=True)

        # Finally, pass through the last embedding layer
        lm_logits = self[len(self) - 1](seq_outputs)

        loss = None
        loss_fct = nn.CrossEntropyLoss()
        labels = kwargs.pop("labels", None)
        mask_indices1 = kwargs.get("mask_indices1")
        mask_indices2 = kwargs.get("mask_indices2")
        
        logits_mask1 = lm_logits[range(lm_logits.shape[0]),mask_indices1,:]
        logits_mask2 = lm_logits[range(lm_logits.shape[0]),mask_indices2,:]

        id2newid = {i:z for i,z in zip(self.config["id2tokenid_values"], range(len(self.config["id2tokenid_values"])))}
        id2dim = {k: [id2newid[v1] for v1 in v] for k, v in self.config["id2tokenid"].items()}
        verbalizerid = list(id2dim.values())

        logits_mask1 = logits_mask1[:,[k[0] for k in verbalizerid]]
        logits_mask2 = logits_mask2[:,[k[1] for k in verbalizerid]]

        logits_for_loss = logits_mask1 + logits_mask2
        
        if labels is not None:
            loss = loss_fct(logits_for_loss.view(-1, len(self.config["id2tokenid"])), labels.view(-1))
        outputs = (logits_for_loss,) + outputs[1:]
        if loss is not None:
            outputs = (loss,) + outputs
        return outputs
 

def read_config(path):
    if type(path) == str:
        path = Path(path)
    with open (path / 'config.yml', 'r') as cfg:
        config = yaml.safe_load(cfg)
    logger.debug('loaded config')
    if not config['dataset']['out_path']:
        config['dataset']['out_path'] = path / 'output'
    if not isinstance(config['adapter']['c_rate'], list):
        config['adapter']['c_rate'] = [config['adapter']['c_rate']]
    if not isinstance(config['adapter']['arch'], list):
        config['adapter']['arch'] = [config['adapter']['arch']]
    if 'per_device_train_batch_size' not in config['adapter']:
        config['adapter']['per_device_train_batch_size'] = 2
    return config


def find_experiments(path):
    if type(path) == str:
        path = Path(path)
    for experiment in path.iterdir():
        if experiment.is_dir() and (experiment / 'test.csv').exists() and (experiment / 'train.csv').exists():
            yield experiment
        else:
            yield from find_experiments(experiment)


def create_eperiment_data(experiment, pattern, verbalizer, tokenizer=tokenizer):
    logger.debug(f'creating data for {experiment}')
    data = {}
    text_test = pd.read_csv(experiment /"test.csv", header = None, names = ["text", "label"])
    text_test.text = pattern + text_test.text
    labels = list(text_test.label.unique())
    labels.sort()
    id2tokenid = {idx:tokenizer(verbalizer[label])["input_ids"][1:3] for idx,label in enumerate(labels)}
    # @TODO: hat jedes Label einen verbalizer
    text_test['labels'] = [labels.index(x) for x in text_test['label']]
    dataset_test = DatasetDict({'eval': Dataset.from_pandas(text_test)})
    dataset_test = dataset_test.map(encode_batch, batched=True)
    mask_indices_test = [np.where(np.array(ids) == tokenizer.mask_token_id)[0] for ids in dataset_test["eval"]["input_ids"]]
    dataset_test["eval"] = dataset_test["eval"].add_column("mask_indices1", [x[0] for x in mask_indices_test])
    dataset_test["eval"] = dataset_test["eval"].add_column("mask_indices2", [x[1] for x in mask_indices_test])
    dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "mask_indices1", "mask_indices2"])
    input_ids = dataset_test["eval"]["input_ids"].cuda()
    attention_mask = dataset_test["eval"]["attention_mask"].cuda()
    mask_indices1 = dataset_test["eval"]["mask_indices1"].cuda()
    mask_indices2 = dataset_test["eval"]["mask_indices2"].cuda()
    actual = dataset_test["eval"]["labels"].numpy()
    text_train = pd.read_csv(experiment / "train.csv", header = None, names = ["text", "label"])
    text_train.text = pattern + text_train.text
    text_train['labels'] = [labels.index(x) for x in text_train['label']]
    dataset = DatasetDict({'train': Dataset.from_pandas(text_train)})
    dataset = dataset.map(encode_batch, batched=True)
    mask_indices = [np.where(np.array(ids) == tokenizer.mask_token_id)[0] for ids in dataset["train"]["input_ids"]]
    dataset["train"] = dataset["train"].add_column("mask_indices1", [x[0] for x in mask_indices])
    dataset["train"] = dataset["train"].add_column("mask_indices2", [x[1] for x in mask_indices])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "mask_indices1", "mask_indices2"])
    data['test_dataset'] = dataset_test
    data['train_dataset'] = dataset
    data['actual'] = actual
    data['input_ids'] = input_ids
    data['attention_mask'] = attention_mask
    data['mask_indices1'] = mask_indices1
    data['mask_indices2'] = mask_indices2
    data['labels'] = labels
    data['id2tokenid'] = id2tokenid
    return data


def evaluate_model(model, data, output_dir):
    model.eval()
    preds = [-100] * len(data['input_ids'])
    for i in range(len(data['input_ids'])):
        with torch.no_grad():
            res = model(input_ids = data['input_ids'][i], attention_mask = data['attention_mask'][i],
                        mask_indices1 = data['mask_indices1'][i], mask_indices2 = data['mask_indices2'][i])[0]
            preds[i] = np.argmax(res.cpu().detach().numpy(), axis = 1)[0]
    scores = compute_metrics(data['actual'], preds)
    with open(output_dir / "scores.json", "w") as fp:
        json.dump(scores, fp)
    conf = confusion_matrix(data['actual'], preds)
    conf_pd = pd.DataFrame(conf)
    conf_pd.columns = conf_pd.index = data['labels']
    conf_pd.to_csv(output_dir / "confusion.csv")
    pd.DataFrame([data['labels'][x] for x in preds]).to_csv(output_dir / "predictions.csv",
                                                    header = False, index = False)


#@TODO: logging into file
def run_all_experiments(path):
    config = read_config(path)
    dataset_path = Path(config['dataset']['data_path'])
    experiments = find_experiments(dataset_path)
    for experiment in experiments:
        experiment_path = experiment.relative_to(dataset_path)
        for arch in config['adapter']['arch']:
            for c_rate in config['adapter']['c_rate']:
                logger.debug(f'running {experiment} with architecture {arch} and c_rate {c_rate}')
                data = create_eperiment_data(experiment, config['dataset']['pattern'], config['dataset']['verbalizer'])
                for run in range(1, config['adapter']['number_of_runs'] + 1):
                    adapter_name = f"{config['dataset']['prefix']}_{arch}_{c_rate}_{run}"
                    output_dir = config['dataset']['out_path'] / experiment_path / adapter_name
                    logger.debug(f'running {experiment} with architecture {arch} and c_rate {c_rate} run {run}')
                    training_args = TrainingArguments(
                        seed=int(1000*run),
                        full_determinism=True,
                        learning_rate=config['adapter']['learning_rate'],
                        num_train_epochs=config['adapter']['max_epochs'],
                        logging_strategy="no",
                        evaluation_strategy="no",
                        save_strategy="no",
                        output_dir=output_dir,
                        overwrite_output_dir=True,
                        remove_unused_columns=False,
                        per_device_train_batch_size = config['adapter']['per_device_train_batch_size'],
                    )
                    model_config = XLMRobertaConfig.from_pretrained("xlm-roberta-large", num_labels=5)
                    model = XLMRobertaModelWithHeads.from_pretrained("xlm-roberta-large", config=model_config)
                    if arch == "pfeiffer":
                        config_adapter = PfeifferConfig(reduction_factor=c_rate)
                    if arch == "pfeifferinv":
                        config_adapter = PfeifferInvConfig(reduction_factor=c_rate)
                    model.add_adapter(adapter_name, config=config_adapter)
                    model.register_custom_head("UkrainePEThead", UkrainePEThead)
                    model.add_custom_head(head_type="UkrainePEThead", head_name=adapter_name, id2tokenid=data['id2tokenid'])
                    model.train_adapter(adapter_name)
                    trainer = AdapterTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=data['train_dataset']["train"]
                    )
                    #TODO: time
                    trainer.train()
                    model.save_adapter(output_dir, adapter_name)
                    #TODO: time
                    evaluate_model(model, data, output_dir)
                    # del model muss man glaube ich nicht callen
                    gc.collect()
                    torch.cuda.empty_cache() 
                    

if __name__ == '__main__':
    cli()
