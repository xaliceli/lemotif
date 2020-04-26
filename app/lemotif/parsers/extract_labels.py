"""
extract_labels.py
Load NLP model and perform inference
"""

import os
import pandas as pd
import tensorflow as tf

from bert import tokenization
from bert import modeling

import lemotif.parsers.utils as ut

class BERTClassifier():

    def __init__(self,
                 model_dir,
                 out_dir,
                 n_classes,
                 batch_size=4,
                 max_seq_len=128,
                 vocab='vocab.txt',
                 init_ckpt='bert_model.ckpt',
                 config='bert_config.json'):
        self.model_dir = model_dir
        self.out_dir = out_dir
        self.n_classes = n_classes
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.vocab = os.path.join(model_dir, vocab)
        self.init_ckpt = os.path.join(model_dir, init_ckpt)
        self.config = os.path.join(model_dir, config)

        self.model = self.load_models()

        print('BERT model initialized.')

    def load_models(self):
        bert_config = modeling.BertConfig.from_json_file(self.config)

        tokenization.validate_case_matches_checkpoint(True, self.init_ckpt)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab, do_lower_case=True)

        run_config = tf.estimator.RunConfig(
            model_dir=self.out_dir,
            save_summary_steps=0,
            keep_checkpoint_max=1,
            save_checkpoints_steps=0)
        model_fn = ut.model_fn_builder(
            bert_config=bert_config,
            num_labels=self.n_classes,
            init_checkpoint=self.init_ckpt,
            learning_rate=0,
            num_train_steps=0,
            num_warmup_steps=0,
            use_tpu=False,
            use_one_hot_embeddings=False)
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": self.batch_size})

        return estimator

    def predict(self, features):
        predict_examples = ut.create_examples(pd.DataFrame(features), False)
        test_features = ut.convert_examples_to_features(predict_examples, self.max_seq_len, self.tokenizer)
        predict_input_fn = ut.input_fn_builder(features=test_features, seq_length=self.max_seq_len, is_training=False,
                                               drop_remainder=False, num_labels=self.n_classes)
        predictions = self.model.predict(predict_input_fn)
        labels_s, labels_e = ut.create_output(predictions)

        return labels_s, labels_e
