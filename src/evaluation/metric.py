import json
import os

from src.evaluation.mrqa_eval import MRQAEvaluator


class SquadMetricWrapper:
    def __init__(self, name):
        self.predictions = dict()
        self.references = dict()
        self.name = name

    def __len__(self):
        return len(self.predictions)

    def add_batch(self, predictions, references):
        for prediction in predictions:
            qid = prediction['id']
            if (qid not in self.predictions) or (prediction['nll'] < self.predictions[qid]['nll']):
                self.predictions[qid] = prediction
        for reference in references:
            self.references[reference['id']] = reference

    def compute(self):
        all_predictions = list(map(lambda kv: kv[1], sorted(list(self.predictions.items()), key=lambda x: x[0])))
        [d.pop('nll') for d in all_predictions if 'nll' in d]
        all_references = list(map(lambda kv: kv[1], sorted(list(self.references.items()), key=lambda x: x[0])))
        print(f'total number of predictions test or val set: {len(all_predictions)}')
        if len(all_predictions)==0:
            return None
        else:
            return MRQAEvaluator.evaluate(answers=all_references, predictions=all_predictions)

    def save(self, path, metric_name):
        file_path = os.path.sep.join(path.split(os.path.sep)[:-1] + [metric_name+'_preds_and_refs.json'])
        all_predictions = list(map(lambda kv: kv[1], sorted(list(self.predictions.items()), key=lambda x: x[0])))
        [d.pop('nll') for d in all_predictions if 'nll' in d]
        all_references = list(map(lambda kv: kv[1], sorted(list(self.references.items()), key=lambda x: x[0])))
        with open(file_path, 'w+') as f:
            json.dump({'predictions': all_predictions, 'references': all_references}, f)
        print(f'saved prediction to {file_path}')

