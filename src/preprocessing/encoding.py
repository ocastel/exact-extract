import datasets
import pandas as pd


class DatasetProcessor:

    @classmethod
    def template_context_question(cls, template, context, question, tokenizer):
        return template.replace('<context>', context).replace('<question>', question).replace(
            '<mask>', tokenizer.additional_special_tokens[0])

    @classmethod
    def encode_context_question(cls, template, context, question, tokenizer, label_without_masks_enc,
                                label_with_masks_enc, id, is_input_splitted, label):
        '''

        :param template: Template to use, expected to have <context>, <question> and <mask> in it.
        :param context: the context to place inside the template.
        :param question: the quesiton to place inside the template.
        :param tokenizer: the tokenizer to use to build the encodings etc.
        :param label_without_masks_enc: The encoding of the label wihtout it paded with mask tokens.
                Used for checking if the label encodings contained inside the context encoding.
        :param label_with_masks_enc:  Label with the mask encodings, to be used at the target.
        :param id: encoded qid.
        :param is_input_splitted: flag if the context is "sliding window" over a too-large context.
        :param label: The label (str).
        :param replace_missing_label_enc: encofing to be used in case the label is missing in its string form.
                If None, then do not replace.
        :return: dict of encodings that can be used both by train and test steps.
        '''
        label_in_context = label in context
        templated_input = cls.template_context_question(template=template, context=context, question=question, tokenizer=tokenizer)
        encodings = tokenizer(templated_input, truncation=False, add_special_tokens=True)
        # added token needed only has placeholder for calc probability of last token
        context = context + tokenizer.additional_special_tokens[1]
        context_encodings = tokenizer(context, truncation=False, add_special_tokens=False, padding='max_length')
        label_enc_in_context_enc = str(label_without_masks_enc)[1:-1] in str(context_encodings.input_ids)
        encoded_id = tokenizer.encode(str(id),truncation=False, add_special_tokens=False, padding='max_length')
        return {'input_ids': encodings.input_ids,
                'attention_mask': encodings.attention_mask,
                'context_input_ids': context_encodings.input_ids,
                'context_attention_mask': context_encodings.attention_mask,
                'labels':label_with_masks_enc,
                'label_in_context': label_in_context,
                'label_enc_in_context_enc': label_enc_in_context_enc,
                'id': encoded_id,
                'is_input_splitted': is_input_splitted
                }

    @classmethod
    def is_encs_valid(cls, encs):
        return all([len(enc['input_ids']) <= 512 for enc in encs])



    @classmethod
    def build_encoding(cls, tokenizer, template, context, question, label, id):
        stride = 120
        context_window_size = None
        label_without_masks_enc = tokenizer.encode(label, add_special_tokens=False)
        label_with_masks = tokenizer.additional_special_tokens[0] + label + tokenizer.additional_special_tokens[1]
        label_with_masks_enc = tokenizer.encode(label_with_masks, add_special_tokens=False)
        encs = [cls.encode_context_question(template=template, context=context, question=question,
                                        tokenizer=tokenizer, label_without_masks_enc=label_without_masks_enc,
                                            label_with_masks_enc=label_with_masks_enc, id=id, is_input_splitted=False,
                                            label=label)]
        while not cls.is_encs_valid(encs):
            context_window_size = 512 if context_window_size is None else int(context_window_size * 0.9)
            context_splited = context.split(' ')
            encs = []
            for context_shard_start_index in range(0, len(context_splited), stride):
                context_shard = ' '.join(
                    context_splited[context_shard_start_index:context_shard_start_index + context_window_size])
                encs.append(cls.encode_context_question(template=template, context=context_shard, question=question,
                                                        tokenizer=tokenizer, label_without_masks_enc=label_without_masks_enc,
                                                        label_with_masks_enc=label_with_masks_enc, id=id, is_input_splitted=True,
                                                        label=label))
                if (context_shard_start_index + context_window_size) > len(context_splited):
                    break
        return encs

    @classmethod
    def prepare_dataset(cls, tokenizer, dataset, template):
        encs = []
        for row in dataset.iterrows():
            encs += DatasetProcessor.build_encoding(tokenizer=tokenizer, template=template, context=row[1]['context'],
                                       question=row[1]['question'], label=row[1]['answers'][0],
                                       id=row[1]['qid'])
        return datasets.Dataset.from_pandas(pd.DataFrame.from_dict(encs))


    @classmethod
    def get_index_of_label_in_context(cls, label_ids, context_ids):
        label_ids_str = str(label_ids[1:-1])[1:-1] #remove mask_0 and mask_1 from edges and then '(',')'.
        context_ids_str = str(context_ids)[1:-1]
        index_in_str = context_ids_str.find(label_ids_str)
        if index_in_str == -1:
            return -1
        index_in_context = context_ids_str[:index_in_str].count(',')
        return index_in_context
