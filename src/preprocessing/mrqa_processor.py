""" Adapted from HuggingFace code for SQuAD """
import json
import pandas as pd


class MRQAProcessor:
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"

    def fix_casing(self, context, ans):
        text_lower = context.lower()
        ans_lower = ans.lower()
        if ans not in context and ans_lower in text_lower:
            idx = text_lower.index(ans_lower)
            return context[idx:idx + len(ans)]
        else:
            return ans

    def create_train_examples(self, input_data):
        examples = []
        for entry in input_data:
            context = entry["context"]
            for qa in entry["qas"]:
                res = {} #['context', 'answers', 'question','qid'
                res['qid'] = qa["id" if "id" in qa else "qid"]
                res['context'] = context
                res['question'] = qa["question"]
                # in some datasets the detected isn't cased as it appears in context; fix if possible
                res['answers'] = [self.fix_casing(context, answer['text']) for answer in qa["detected_answers"]]
                examples.append(res)
        return pd.DataFrame(examples)

    def create_train_examples_from_hfdatasets(self, input_data):
        df = input_data.data.to_pandas()
        df['answers'] = df.apply(
            lambda x: [self.fix_casing(x['context'], answer) for answer in x['detected_answers']['text']], axis=1)
        return df[['context', 'answers', 'question','qid']]

    def create_test_examples(self, input_data):
        examples = []
        for entry in input_data:
            context = entry["context"]
            for qa in entry["qas"]:
                res = {} #['context', 'answers', 'question','qid'
                res['qid'] = qa["id" if "id" in qa else "qid"]
                res['context'] = context
                res['question'] = qa["question"]
                res['answers'] = qa["answers"] # evaluation is done against the "answers" field as in mrqa official eval
                examples.append(res)
        return pd.DataFrame(examples)


    def get_examples(self, path, is_train, from_hfdatasets=False):
        """
        Returns the training examples from the data directory.
        """
        with open(path, "r", encoding="utf-8") as reader:
            print(reader.readline())
            input_data = [json.loads(line) for line in reader]
        if is_train:
            return self.create_train_examples(input_data)
        else:
            return self.create_test_examples(input_data)
