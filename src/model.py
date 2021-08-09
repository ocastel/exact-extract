from __future__ import annotations
import itertools
import pickle
import sys
import os
from timeit import default_timer as timer
import traceback
from typing import List, Optional, Tuple, Union
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import Adafactor
import math
from transformers.data.data_collator import DataCollatorForSeq2Seq
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from argparse import ArgumentParser
import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
import os
import time

sys.path.insert(0,os.getcwd())
from src.decoding.exact_extract import mlspan, InputAndAttention
from src.evaluation.metric import SquadMetricWrapper
from src.preprocessing.data_loading import FewShotDataLoader
from src.preprocessing.encoding import DatasetProcessor
from src.utils.outputs import PredictionWithMetadata, ExtractionResults, AggregatedPredictionsOfType, SpanType



print('setting env variable "TOKENIZERS_PARALLELISM" to "false"')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class ValEveryNSteps(Callback):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step


    def on_batch_end(self, trainer, pl_module):
        if os.path.exists(os.path.join(pl_module.hparams.results_path)):
            print(f'Seems like another process have written results for {pl_module.hparams.exp_name} in {pl_module.hparams.results_path}, exiting.')
            raise RuntimeError(f'Seems like another process have written results for {pl_module.hparams.exp_name} in {pl_module.hparams.results_path}, existing.')

        if trainer.accumulate_grad_batches > 1 and (trainer.batch_idx % trainer.accumulate_grad_batches) != 0:
            return
        if self.every_n_step > 0:
            if (trainer.global_step+1) % self.every_n_step == 0 and trainer.global_step > 0:
                print(f'Finished step {trainer.global_step+1} (1-based), running evaluation:')
                self.run_evaluation_and_log(trainer=trainer, pl_module=pl_module, steps=trainer.global_step + 1)

    def on_train_start(self, trainer, pl_module):
        print(f'Running evaluation for non-finetuned model:')
        self.run_evaluation_and_log(trainer, pl_module, -1)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_step < pl_module.hparams.max_steps:
            return # skip saving if training end too early
        with open(pl_module.hparams.results_path, 'w+') as f:
            for r in pl_module.results:
                f.write(json.dumps(r)+'\n')

    def run_evaluation_and_log(self, trainer, pl_module, steps):
        start = timer()
        trainer.run_evaluation()
        for metric_name, metric_obj in pl_module.validation_metrics():
            pl_module.compute_and_log(metric_obj, metric_name, str(steps))
        pl_module.reset_metrics()
        end = timer()
        print(f'Evaluation took {end - start} seconds on step {steps}')


class SquadModel(LightningModule):

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adafactor_const':
            optimizer = Adafactor(self.model.parameters(), lr=self.hparams.lr, relative_step=False,
                                  scale_parameter=False)
            return optimizer
        else:
            raise RuntimeError(f'optimizer {self.hparams.optimizer} is not supported.')

    def checkpoint_path(self):
        return os.path.join(self.results_dir,'checkpoint.ckpt')


    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.results = []
        self.fewshot_dataloder = FewShotDataLoader()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.model_name, cache_dir=self.hparams.cache_dir)
        if self.hparams.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name, cache_dir=self.hparams.cache_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, cache_dir=self.hparams.cache_dir)
        self.results_dir = os.path.sep.join(self.hparams.results_path.split(os.path.sep)[:-1])
        self.reset_metrics()
        self.reset_test_metrics()
        # this map is used for dynamically trying out different chunk sizes per passage length
        self.context_size_to_chunks = dict([(i, 1) for i in range(0, 520, 10)])
        self.eos_token_id = self.tokenizer.additional_special_tokens_ids[1]


    def create_metric(self):
        return SquadMetricWrapper(name=self.hparams.exp_name)

    def reset_metrics(self):
        self.val_ml_span_metric = self.create_metric()
        self.val_ml_span_norm_metric = self.create_metric()
        self.val_greedy_metric = self.create_metric()
        self.val_beam_metric = self.create_metric()

    def reset_test_metrics(self):
        self.test_ml_span_metric = self.create_metric()
        self.test_ml_span_norm_metric = self.create_metric()
        self.test_greedy_metric = self.create_metric()
        self.test_beam_metric = self.create_metric()

    def validation_metrics(self):
        names = [f for f in vars(self).keys() if f.startswith('val_') and f.endswith('_metric')]
        return [(name.replace('_metric',''), getattr(self,name)) for name in names]

    def test_metrics(self):
        names = [f for f in vars(self).keys() if f.startswith('test_') and f.endswith('_metric')]
        return [(name.replace('_metric',''), getattr(self,name)) for name in names]

    def train_dataloader(self):
        print('loading train dataloader')
        if self.hparams.train_rss:
            print(f'loading {self.hparams.train_samples} from rss data')
            train_data_enc = self.fewshot_dataloder.load_rss(self.hparams.train_samples)
        else:
            print(f'loading train samples ({self.hparams.dataset}, {self.hparams.train_samples})')
            if self.hparams.train_samples == -1:
                train_samples = self.fewshot_dataloder.load_train_ds(self.hparams.dataset, cache_dir=self.hparams.cache_dir)
            else:
                train_samples = self.fewshot_dataloder.load_splinter_ds(self.hparams.splinter_data, self.hparams.dataset.lower(), self.hparams.seed,
                                 self.hparams.train_samples, hp_search=False, is_train=True, is_test=False, is_val=False)
            train_data_enc = DatasetProcessor.prepare_dataset(tokenizer=self.tokenizer, dataset=train_samples, template=self.hparams.pattern)
            print(f'total number of  original training samples (not multiple from a single (long) sample: {len(train_samples)}')
            print(f'total number of training samples (including multiple from a single (long) sample: {len(train_data_enc)}')
        return DataLoader(train_data_enc, num_workers=4, shuffle=True, batch_size=self.hparams.batch_size,
                          collate_fn=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True))

    def val_dataloader(self):
        print('Loading val split')
        if self.hparams.hp_search:
            print('loading validation from splinter val sample.')
            val_samples = self.fewshot_dataloder.load_splinter_ds(self.hparams.splinter_data, self.hparams.dataset.lower(), self.hparams.seed,
                             self.hparams.val_samples, hp_search=self.hparams.hp_search, is_val=True, is_test=False, is_train=False)
        else:
            if self.hparams.dataset.lower() in ('textbookqa', 'bioasq'):
                val_samples = self.fewshot_dataloder.load_splinter_ds(self.hparams.splinter_data, self.hparams.dataset.lower(),
                                                self.hparams.seed,
                                                self.hparams.val_samples, hp_search=False, is_val=True, is_test=False, is_train=False)
            else:
                val_samples = self.fewshot_dataloder.load_dev_ds(self.hparams.dataset, self.hparams.val_samples,
                                           cache_dir=self.hparams.cache_dir, seed=self.hparams.val_seed)
            print('loading validation from original dev split.')
        print(f'Loaded {len(val_samples)} raw val samples, encoding...')
        print('template is '+self.hparams.pattern)
        self.val_answers = val_samples.set_index(['qid']).answers
        val_data_enc = DatasetProcessor.prepare_dataset(self.tokenizer, val_samples, self.hparams.pattern)
        print(f'After encoding and duplicating long samples, validation set containes {len(val_data_enc)} entries')
        return DataLoader(val_data_enc, num_workers=4, shuffle=False, batch_size=self.hparams.val_batch_size,
                          collate_fn=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True))

    def test_dataloader(self):
        print('Loading test split')
        if self.hparams.dataset.lower() in ('textbookqa', 'bioasq'):
            test_samples = self.fewshot_dataloder.load_splinter_ds(self.hparams.splinter_data, self.hparams.dataset.lower(), self.hparams.test_seed,
                         self.hparams.test_samples, is_test=True, is_train=False, is_val=False, hp_search=False) # hp_search true will get data from dev split
        else:
            test_samples = self.fewshot_dataloder.load_dev_ds(self.hparams.dataset, self.hparams.test_samples,
                                      cache_dir=self.hparams.cache_dir, seed=self.hparams.test_seed)
        if self.hparams.test_samples > 0:
            test_samples = test_samples.sample(n=self.hparams.test_samples, random_state=self.hparams.test_seed)
        print(f'Loaded {len(test_samples)} raw test samples, encoding...')
        self.test_answers = test_samples.set_index(['qid']).answers#.apply(lambda x: x[0])
        test_data_enc = DatasetProcessor.prepare_dataset(self.tokenizer, test_samples, self.hparams.pattern)
        print(f'Resulted with {len(test_data_enc)} encoded test samples.')
        return DataLoader(test_data_enc, num_workers=4, shuffle=False, batch_size=self.hparams.val_batch_size,
                          collate_fn=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True))

    def forward(self, args):
        res = self.model.generate(input_ids=args['input_ids'], attention_mask=args['attention_mask'],
                                  max_length=self.hparams.trim_context, min_length=3)
        return res

    def training_step(self, batch, batch_idx):
        loss = self.model(input_ids=batch['input_ids'],  attention_mask=batch['attention_mask'],
                          labels=batch['labels']).loss
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def predict_from_span(self, batch) -> List[Tuple[ExtractionResults, int]]:
        span_selections = []
        for sample_id in range(len(batch['input_ids'])):
            input_ids = batch['input_ids'][sample_id]
            attention_mask = batch['attention_mask'][sample_id]
            context = batch['context_input_ids'][sample_id]
            context = context[context != self.tokenizer.pad_token_id]
            extraction_result = None
            context_rounded_size = context.shape[-1] // 10 * 10
            while extraction_result is None:
                try:
                    chunk_size = math.ceil(512 / self.context_size_to_chunks[context_rounded_size])
                    extraction_result = mlspan(model=self.model, tokenizer=self.tokenizer,
                                               encoder_input_attention=InputAndAttention(input_ids,attention_mask),
                                               context=context, chunk_size=chunk_size,
                                               device=self.device, trim_context=self.hparams.trim_context,
                                               eos_token_id=self.eos_token_id)
                except RuntimeError as e:
                    if 'out of memory' not in str(e):
                        traceback.print_exc()
                        raise e
                    next_chunks = self.context_size_to_chunks[context_rounded_size] + 1
                    next_chunk_size = math.ceil(512 / next_chunks)
                    while next_chunk_size == chunk_size:
                        next_chunks = next_chunks + 1
                        next_chunk_size = math.ceil(512 / next_chunks)
                    print(
                        f'Decreasing chunks size for context of size {context_rounded_size} from {chunk_size} to {next_chunk_size} ; {self.context_size_to_chunks[context_rounded_size]} -> {next_chunks}')
                    self.context_size_to_chunks[context_rounded_size] = next_chunks
                    for i in range(context_rounded_size, 520, 10):
                        self.context_size_to_chunks[i] = np.max(
                            (self.context_size_to_chunks[context_rounded_size], self.context_size_to_chunks[i]))
                    print(self.context_size_to_chunks)
                    if self.context_size_to_chunks[context_rounded_size] > context.shape[-1]:
                        raise RuntimeError(
                            "Tried calculating mlspan for chunk made of a single item but failed; consider "
                            "setting trim_context at a lower value.")
            span_selections.append((extraction_result, self.tokenizer.decode(batch['id'][sample_id], skip_special_tokens=True)))
        return span_selections

    def predict_greedy(self, batch) -> Tuple[List[List[int]], List[List[int]]]:
        pred_dicts = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                         eos_token_id=self.eos_token_id,
                                         return_dict_in_generate=True,
                                         output_scores=True,
                                         min_length=self.hparams.min_length
                                         )
        preds_ids = pred_dicts.sequences[:, 1:]
        pred_scores = pred_dicts.scores
        pred_probs = ( (preds_ids != 0).int() * torch.nn.CrossEntropyLoss(reduction='none')(input=torch.stack(pred_scores, dim=1).permute(0, 2, 1), target=preds_ids)).nan_to_num().sum(dim=1)
        return preds_ids, pred_probs.tolist()

    def predict_beam(self, batch) -> Tuple[List[List[int]], List[List[int]]]:
        pred_dicts = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                        eos_token_id=self.eos_token_id,
                                        early_stopping=False,
                                        num_beams=8,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        num_return_sequences=1
                                        )
        preds_ids = pred_dicts.sequences[:, 1:]
        pred_probs = -1*pred_dicts.sequences_scores # beam already process this for us, but returns maximized scores, so negate
        return preds_ids, pred_probs.tolist()

    def build_references(self, ids, is_validation):
        answers = self.val_answers if is_validation else self.test_answers
        references = [{'id': id, 'answers': {'text': list(answers.loc[id]), 'answer_start': [0]*len(answers.loc[id])}} for id in ids]
        return references

    def to_predictions(self, span_preds, field_name) -> Optional[List[dict]]:
        if getattr(span_preds[0][0],field_name) is None:
            return None
        return [{'id':str(span_pred[1]), 'prediction_text': getattr(span_pred[0],field_name).top.decoded,
                 'nll': getattr(span_pred[0],field_name).top.nll} for span_pred in span_preds]

    def validation_or_test_step(self, batch, is_validation) -> Tuple[dict,List[ExtractionResults]]:
        if is_validation:
            loss = self.model(input_ids=batch['input_ids'], labels=batch['labels']).loss
            self.log('val_loss', loss, sync_dist=True)
        ids = self.tokenizer.batch_decode(batch['id'], skip_special_tokens=True) #[str(i) for i in batch['num_id'].cpu().numpy().reshape(-1)]
        references = self.build_references(ids, is_validation)
        if self.hparams.decode_ml_span:
            span_predictions = self.predict_from_span(batch)
        else:
            span_predictions = [(ExtractionResults.empty(),self.tokenizer.decode(id_enc, skip_special_tokens=True)) for id_enc in batch['id']]
        if self.hparams.decode_greedy:
            batch_generate_greedy_ids, batch_generate_greedy_log_probs  = self.predict_greedy(batch)
        if self.hparams.decode_beam:
            batch_generate_beam_ids, batch_generate_beam_log_probs = self.predict_beam(batch)
        for i in range(len(batch['input_ids'])):
            span_prediction = span_predictions[i]
            if self.hparams.decode_greedy:
                decoded = self.tokenizer.decode(batch_generate_greedy_ids[i], skip_special_tokens=True)
                nll = batch_generate_greedy_log_probs[i]
                span_prediction[0].greedy = AggregatedPredictionsOfType(
                    top_k=[PredictionWithMetadata(tokens=[], tokens_ids=batch_generate_greedy_ids[i].tolist(),
                                tokens_nlls=[], decoded=decoded, nll=nll)], tpe=SpanType.GREEDY, k=1)
            if self.hparams.decode_beam:
                span_prediction[0].beam = AggregatedPredictionsOfType(
                    top_k=[PredictionWithMetadata(tokens=[], tokens_ids=batch_generate_beam_ids[i].tolist(),
                                tokens_nlls=[], decoded=self.tokenizer.decode(batch_generate_beam_ids[i],
                                                                              skip_special_tokens=True),
                                nll=batch_generate_beam_log_probs[i])], tpe=SpanType.BEAM, k=1)
        ret_dict = {}
        if self.hparams.decode_greedy:
            original_contexts = self.tokenizer.batch_decode(batch['context_input_ids'], skip_special_tokens=True)
            greedy_predictions = self.to_predictions(span_preds=span_predictions,field_name='greedy')
            greedy_in_context = [(greedy_prediction['prediction_text'] in original_context and len(greedy_prediction['prediction_text'])>0) for
                                 greedy_prediction, original_context in zip(greedy_predictions, original_contexts)]
            k = 'greedy_in_context_acc' if is_validation else 'test_greedy_in_context_acc'
            ret_dict[k] = greedy_in_context
        metrics = self.validation_metrics() if is_validation else self.test_metrics()
        for (metric_name, metric_obj) in metrics:
            metric_name = metric_name.replace('test_', '').replace('val_','')
            predictions = self.to_predictions(span_predictions, metric_name)
            if predictions is not None:
                metric_obj.add_batch(predictions=predictions, references=references)
        span_prediction_results = list(map(lambda x:x[0],span_predictions))
        return ret_dict, span_prediction_results

    def validation_step(self, batch, batch_idx):
        logs_dict, prediction_results = self.validation_or_test_step(batch=batch, is_validation=True)
        return logs_dict

    def test_step(self, batch, batch_idx):
        logs_dict, prediction_results = self.validation_or_test_step(batch=batch, is_validation=False)
        return logs_dict

    def compute_and_log(self, metric, metric_name, step=None):
        met = metric.compute()
        if met is not None:
            full_metric_name = metric_name if step is not None else str(step)+'_'+metric_name
            self.log(full_metric_name+'_f1', met['f1'])
            self.log(full_metric_name+'_EM', met['exact_match'])
            print(full_metric_name + ':' + str(met))
            json_dict = dict()
            for k in self.hparams:
                try: # avoid unserializable keys gracefully
                    json_dict[k] = json.dumps(self.hparams[k])
                except TypeError:
                    pass
            json_dict['metric_name'] = metric_name
            json_dict['step'] = step
            json_dict['f1'] = met['f1']
            json_dict['EM'] = met['exact_match']
            self.results.append(json_dict)

    def pickle_path(self, metric_name, rank):
        dir_path = os.path.sep.join(self.hparams.results_path.split(os.path.sep)[:-1] +
                                [metric_name])
        file_path = os.path.sep.join([dir_path, str(rank) + '.pkl'])
        return  dir_path, file_path

    def checkpoint_dir(self):
        dir_path = os.path.sep.join(self.hparams.results_path.split(os.path.sep)[:-1] +
                                ['final_checkpoint'])
        return dir_path

    def pickle_metric(self, metric_name, metric_obj):
        dir_path, file_path = self.pickle_path(metric_name, self.global_rank)
        print(f'pickling to {file_path}; contains {len(metric_obj)} predictions.')
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(metric_obj, f)

    def unpickle_metrics(self, metric_name, world_size):
        unpickled_metrics = []
        for rank in range(1, world_size):
            dir_path, file_path = self.pickle_path(metric_name, rank)
            while not os.path.exists(file_path):
                print(f'{file_path} does not exist, sleeping for 2 seconds')
                time.sleep(2)
            time.sleep(30) # to avoid race condition, sleep for some time
            with open(file_path, 'rb') as f:
                print(f'unpickling {file_path}')
                other_metric: SquadMetricWrapper = pickle.load(f)
                unpickled_metrics.append(other_metric)
        return unpickled_metrics

    # def update_metric_from_pickles(self, ):
    def test_epoch_end(self, outputs):
        gpus = len(self.hparams.gpus.split(',')) if type(self.hparams.gpus) == str else self.hparams.gpus
        world_size = gpus * self.hparams.num_nodes
        print(f'global rank is {self.global_rank}, world size is {world_size}')

        for metric_name, metric_obj in self.test_metrics():
            if world_size == 1:
                self.compute_and_log(metric_obj, metric_name)
                metric_obj.save(self.hparams.results_path, metric_name)
            else:
                self.pickle_metric(metric_name=metric_name, metric_obj=metric_obj)
                if self.global_rank == 0:
                    unpickled_metrics = self.unpickle_metrics(metric_name, world_size)
                    for unpickled_metric in unpickled_metrics:
                        metric_obj.add_batch(unpickled_metric.predictions.values(), unpickled_metric.references.values())
                    metric_obj.save(self.hparams.results_path, metric_name)
                    self.compute_and_log(metric_obj, metric_name)
        if self.hparams.decode_greedy:
            bool_list = list(itertools.chain.from_iterable([d['test_greedy_in_context_acc'] for d in outputs]))
            if world_size == 1:
                self.log('test_greedy_in_context_acc',np.mean(bool_list))
            else:
                self.pickle_metric('test_greedy_in_context',bool_list)
                if self.global_rank == 0:
                    in_context_bool_lists = self.unpickle_metrics('test_greedy_in_context', world_size)
                    for l in in_context_bool_lists:
                        bool_list += l
                    self.log('test_greedy_in_context_acc', np.mean(bool_list))
        with open(self.hparams.results_path, 'w+') as f:
            for r in self.results:
                f.write(json.dumps(r)+'\n')


    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='patrickvonplaten/t5-tiny-random')
        parser.add_argument('--tokenizer', default=None, type=str)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--train_samples', type=int, default=64, help="negative for full dataset")
        parser.add_argument('--val_samples', type=int, default=1024)
        parser.add_argument('--test_samples', type=int, default=10)
        parser.add_argument('--pattern', type=str, default="Text: <context>\nQuestion: <question>\nAnswer:<mask>.")
        parser.add_argument('--exp_name', type=str, default='test')
        parser.add_argument('--cache_dir', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--val_batch_size', type=int, default=2)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--trim_context', type=int, default=512)
        parser.add_argument('--dataset', type=str, default='SQuAD')
        parser.add_argument('--splinter_data', type=str, default='./data')
        parser.add_argument('--optimizer', default='adamw', type=str)
        parser.add_argument('--tags', default='', type=str)
        parser.add_argument('--decode_beam', default=False, action='store_true')
        parser.add_argument('--decode_greedy', default=False, action='store_true')
        parser.add_argument('--decode_ml_span', default=False, action='store_true')
        parser.add_argument('--val_seed',default=1, type=int)
        parser.add_argument('--test_seed',default=0, type=int)
        parser.add_argument('--check_val_every_n_steps', default=64, type=int)
        parser.add_argument('--results_path', type=str, default= './results')
        parser.add_argument('--hp_search', default=False, action='store_true')
        parser.add_argument('--min_length',default=None)
        parser.add_argument('--train_rss', default=False, action='store_true')
        parser.add_argument('--save_model', default=False, action='store_true')
        return parser


def neptune_tags(args):
    tags = args.tags.split(',')
    tags = tags + [args.optimizer, 'bsz_' + str(args.batch_size), 'accum_' + str(args.accumulate_grad_batches),
                   'seed_' + str(args.seed), 'lr_' + str(args.lr), 'steps' + str(args.max_steps)]
    return tags


def main(args):
    seed_everything(0)
    save_dir = os.path.sep.join((os.getcwd(), args.exp_name)) if args.cache_dir is None else os.path.sep.join(
        (args.cache_dir, args.exp_name))
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=0,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=args.exp_name
    )
    if args.max_steps is not None:
        args.max_epochs = 99999 # overriding to have full control through max_steps
    neptune_logger = NeptuneLogger(
        close_after_fit=False,
        experiment_name=args.exp_name,
        params=vars(args),
        tags=neptune_tags(args),
    )
    trainer = Trainer.from_argparse_args(args, deterministic=True,
                                         checkpoint_callback=checkpoint_callback,
                                         logger=neptune_logger,
                                         progress_bar_refresh_rate=64
                                         )
    model = SquadModel(args)
    if args.train_samples != 0:
        trainer.fit(model)
        if args.save_model:
            print(f'saving checkpoint to {model.checkpoint_dir()}')
            model.model.save_pretrained(model.checkpoint_dir())
    if args.test_samples!=0:
        trainer.test(model)


def update_local_args(args):
    os.makedirs('/'.join(args.results_path.split('/')[:-1]), exist_ok=True )
    return args


if __name__ == '__main__':
    start = timer()
    import platform
    parser = ArgumentParser(add_help=True)
    parser = Trainer.add_argparse_args(parser)
    parser = SquadModel.add_model_specific_args(parser)
    args = parser.parse_args()
    if platform.release() == '5.8.0-7642-generic':
        args = update_local_args(args)
    if args.val_samples == 0:
        args.val_percent_check = 0
    main(args)
    end = timer()
    print(end - start)

