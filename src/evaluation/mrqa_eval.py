"""Adapted from mrqa_official_eval.py, which was adapted fromt the SQuAD v1.1 official evaluation script.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
import re
from collections import Counter


class MRQAEvaluator:

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = MRQAEvaluator.normalize_answer(prediction).split()
        ground_truth_tokens = MRQAEvaluator.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return (MRQAEvaluator.normalize_answer(prediction) == MRQAEvaluator.normalize_answer(ground_truth))

    @staticmethod
    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    @staticmethod
    def evaluate(answers, predictions, skip_no_answer=False):
        # structure need to be similar
        predictions_dict = {p['id']:p['prediction_text'] for p in predictions}
        f1 = exact_match = total = 0
        for ans in answers:
            qid = ans['id']
            ground_truths = ans['answers']['text']
            if qid not in predictions_dict:
                if not skip_no_answer:
                    raise RuntimeError('Unanswered question %s.' % qid)
                else:
                    print('Unanswered question %s will receive score 0.' % qid)
                    total += 1
                continue
            total += 1
            prediction = predictions_dict[qid]
            exact_match += MRQAEvaluator.metric_max_over_ground_truths(
                MRQAEvaluator.exact_match_score, prediction, ground_truths)
            f1 += MRQAEvaluator.metric_max_over_ground_truths(
                MRQAEvaluator.f1_score, prediction, ground_truths)

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        return {'exact_match': exact_match, 'f1': f1}

