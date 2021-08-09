import os
import pickle

import datasets
import pandas as pd

from src.preprocessing.mrqa_processor import MRQAProcessor

class FewShotDataLoader:

    dataset_to_hf_name = {'SearchQA':'SearchQA', 'TriviaQA':'TriviaQA-web', 'SQuAD':'SQuAD',
                          'NewsQA':'NewsQA', 'NaturalQuestions':'NaturalQuestionsShort',
                          'HotpotQA':'HotpotQA'}

    def splinter_ds_path(self, path, ds_name, seed, nsamples, hp_search, is_val, is_test, is_train):
        if hp_search and is_val:
            path = os.path.join(path, ds_name, f'{ds_name}-val-regular.jsonl')
            return path
        elif is_val or is_test:
            path = os.path.join(path, ds_name, f'dev-regular.jsonl')
            return path
        else:
            if nsamples > 0:
                if seed > 4: raise RuntimeError('seed need to be in range 0-5, but got ' + str(seed))
                path = os.path.join(path, ds_name, f'{ds_name}-train-seed-{42 + seed}-num-examples-{nsamples}-regular.jsonl')
            else:
                path = os.path.join(path, ds_name, f'{ds_name}-train-num-examples-full.jsonl')
            return path


    def load_splinter_ds(self, path, ds_name, seed, nsamples, hp_search, is_train, is_val, is_test):
        if ds_name == 'naturalquestionsshort':
            ds_name = 'naturalquestions'
        if ds_name == 'triviaqa-web':
            ds_name == 'triviaqa'
        path = self.splinter_ds_path(path=path, ds_name=ds_name, seed=seed, nsamples=nsamples, hp_search=hp_search,
                                is_val=is_val, is_test=is_test, is_train=is_train)
        print('loading from ' + path)
        processor = MRQAProcessor()
        df = processor.get_examples(path, is_train=is_train)
        return df


    def load_dev_ds(self, ds_name, size=None,cache_dir=None, seed=0):
        dataset = datasets.load_dataset('mrqa',cache_dir=cache_dir)
        validation_dataset = dataset['validation'].filter(lambda x: x['subset']==FewShotDataLoader.dataset_to_hf_name[ds_name])
        validatoin_df = validation_dataset.data.to_pandas()[['context', 'answers', 'question','qid']]
        if size>0:
            validatoin_df = validatoin_df.sample(n=size, random_state=seed)
        return validatoin_df

    def load_train_ds(self, ds_name, cache_dir=None):
        corrected_ds_name = FewShotDataLoader.dataset_to_hf_name[ds_name]
        dataset = datasets.load_dataset('mrqa',cache_dir=cache_dir, split='train').filter(lambda x: x['subset']==corrected_ds_name)
        processor = MRQAProcessor()
        train_df = processor.create_train_examples_from_hfdatasets(dataset)
        return train_df

    def load_rss(self, train_samples):
        pickle_path = f'data/rss/splinter-style-rs-{train_samples}-samples.pkl'
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data, columns=['input_ids', 'labels'])
        df['attention_mask'] = df.input_ids.apply(lambda x: [1]*len(x))
        ds = datasets.Dataset.from_pandas(df)
        return ds
