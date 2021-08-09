import unittest
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from src.model import mlspan, extract_tokens_and_probs_from_prediction_ids, InputAndAttention, \
    extract_tokens_and_probs_from_prediction_ids_with_scores
from src.model import DatasetProcessor    # get_index_of_label_in_context


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('patrickvonplaten/t5-tiny-random')
        self.model = AutoModelForSeq2SeqLM.from_pretrained('patrickvonplaten/t5-tiny-random').to(0)
        self.model.eval()
        self.mask0 = self.tokenizer.additional_special_tokens_ids[0]
        self.mask1 = self.tokenizer.additional_special_tokens_ids[1]
        self.context = 'Barack Hussein Obama II (/bəˈrɑːk huːˈseɪn oʊˈbɑːmə/ (About this soundlisten) bə-RAHK hoo-SAYN oh-BAH-mə;[1] born August 4, 1961 is an American politician and attorney who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, Obama was the first African-American president of the United States. He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.'
        self.question = 'Which party is Obama a member of?'
        self.answer = 'Democratic Party'
        self.dummy_dataset = Dataset.from_dict({'question':[self.question], 'context':[self.context], 'answers':[self.answer]})

    def compare_ids_to_stri(self, ids, stri):
        return self.tokenizer.decode(ids, skip_special_tokens=True) == self.tokenizer.decode(self.tokenizer.encode(stri),skip_special_tokens=True)

    def test_prepare_dataset_template(self):
        template = '<context> <question> <mask>'
        ds = DatasetProcessor.prepare_dataset(self.tokenizer, dataset=self.dummy_dataset.data.to_pandas(), template=template,
                                              report_label_first_token_index=False)
        assert self.compare_ids_to_stri(ds['context_input_ids'][0],self.context)
        assert self.compare_ids_to_stri(ds['input_ids'][0],self.context + ' ' + self.question)
        template = 'Answer the question based on the paragraph. <context> <question> <mask>'
        ds = DatasetProcessor.prepare_dataset(self.tokenizer, dataset=self.dummy_dataset.data.to_pandas(), template=template,
                                   report_label_first_token_index=False)
        assert self.compare_ids_to_stri(ds['context_input_ids'][0],self.context)
        assert ds['input_ids'][0][-2] == self.tokenizer.additional_special_tokens_ids[0]
        assert self.compare_ids_to_stri(ds['input_ids'][0],'Answer the question based on the paragraph. ' + self.context + ' ' + self.question)

    def test_equality_of_different_chunk_size(self):
        input_ids = self.tokenizer.encode('The big dog Who? '+self.tokenizer.additional_special_tokens[0],
                                          return_tensors='pt').to(self.model.device)[0]
        context_ids = self.tokenizer.encode('The big dog'+self.tokenizer.additional_special_tokens[1], add_special_tokens=False,
                                            return_tensors='pt').to(self.model.device)[0]
        # check agnostic to chunk size
        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(self.model.device)
        f = lambda num_chunks: mlspan(model=self.model,tokenizer=self.tokenizer, encoder_input_attention=InputAndAttention(input_ids,attention_mask),
                                      context=context_ids, chunk_size=num_chunks,
                                      device=self.model.device, trim_context=15, k_top_spans=1,
                                      first_token_top_k=0, first_token_top_p=0, label_first_token_index=None)
        spans_by_chunks_size = [f(i) for i in range (1,10)]
        for i in range(1, len(spans_by_chunks_size)):
            assert spans_by_chunks_size[0] == spans_by_chunks_size[i]
        top_ml_span = spans_by_chunks_size[0].ml_span.top
        prediction_ids = torch.LongTensor(top_ml_span.tokens_ids).to(self.model.device)#self.tokenizer.encode(prediction, add_special_tokens=False, return_tensors='pt').to(self.model.device)
        extracted_nlls = extract_tokens_and_probs_from_prediction_ids(tokenizer=self.tokenizer, model=self.model,
                                                                      input_ids=input_ids.squeeze(0),
                                                                      attention_mask=attention_mask,
                                                                      prediction_ids=prediction_ids)
        assert top_ml_span == extracted_nlls

    def enc(self, s, add_eos):
        e = self.tokenizer(s + self.tokenizer.additional_special_tokens[0],return_tensors='pt', add_special_tokens=add_eos)
        return e.input_ids.to(self.model.device), e.attention_mask.to(self.model.device)

    def test_get_index_of_label_in_context(self):
        context_ids = [0, 1, 2, 3, 4, 5]
        def local_test(l, idx):
            padded_input_ids = [self.tokenizer.additional_special_tokens_ids[0]] + l + [self.tokenizer.additional_special_tokens_ids[1]]
            fidx = DatasetProcessor.get_index_of_label_in_context(padded_input_ids, context_ids)
            assert  fidx == idx
        local_test([0, 1, 2], 0)
        local_test([0], 0)
        local_test([1, 2, 3], 1)
        local_test([1, 2], 1)
        local_test([4, 5], 4)
        local_test([0, 2], -1)

    def test_greedy_decode_nlls(self):
        template = '<context> <question> <mask>'
        ds = DatasetProcessor.prepare_dataset(self.tokenizer, dataset=self.dummy_dataset.data.to_pandas(), template=template,
                                   report_label_first_token_index=True)
        assert self.compare_ids_to_stri(ds['context_input_ids'][0],self.context)
        assert ds['input_ids'][0][-2] == self.tokenizer.additional_special_tokens_ids[0]
        input_ids = torch.LongTensor(ds['input_ids']).to(0)
        attention_mask = torch.LongTensor(ds['attention_mask']).to(0)
        generated_result = self.model.generate(input_ids=input_ids, output_scores=True, return_dict_in_generate=True,
                                               attention_mask=attention_mask, decoder_start_token_id=self.model.config.decoder_start_token_id,
                                               min_length=4, max_length=4)
        hf_scores = generated_result.scores
        hf_scores = torch.stack(hf_scores).squeeze()
        generated_ids = generated_result.sequences[0]
        prediction, scores = extract_tokens_and_probs_from_prediction_ids_with_scores(self.tokenizer, self.model, input_ids=input_ids[0],
                                                                  prediction_ids=generated_ids[1:],
                                                                  attention_mask=attention_mask[0])
        scores = scores.squeeze()
        scores[:,1] = -float('inf')
        assert ((scores - hf_scores).abs() > 0.00001).sum()==0

    def test_build_encoding(self):
        def gen_text(length):
            return ' '.join(['dog']*length)
        template = '<context> <question> <mask>'
        short_context = gen_text(10)
        short_question = gen_text(10)
        short_label = gen_text(3)
        short_encs = DatasetProcessor.build_encoding(self.tokenizer, template=template, context=short_context,
                                        question=short_question, label=short_label, num_id=0)
        assert len(short_encs) == 1

        long_context = gen_text(512)
        long_question = gen_text(10)
        long_label = gen_text(3)
        long_encs = DatasetProcessor.build_encoding(self.tokenizer, template=template, context=long_context,
                                               question=long_question, label=long_label, num_id=0)
        assert len(long_encs) == 2

        slong_context = gen_text(512+100)
        slong_question = gen_text(10)
        slong_label = gen_text(3)
        slong_encs = DatasetProcessor.build_encoding(self.tokenizer, template=template, context=slong_context,
                                                    question=slong_question, label=slong_label, num_id=0)
        assert len(slong_encs) == 3

if __name__ == '__main__':
    unittest.main()
