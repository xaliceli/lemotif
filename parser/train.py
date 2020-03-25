"""
train.py
Fine-tune BERT for classification task.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

from bert import tokenization
from bert import modeling

import eval
import utils as ut

class BERTClassifier():

    def __init__(self,
                 model_dir,
                 out_dir,
                 n_classes,
                 max_seq_len=128,
                 vocab='vocab.txt',
                 init_ckpt='bert_model.ckpt',
                 config='bert_config.json'):
        self.model_dir = model_dir
        self.out_dir = out_dir
        self.n_classes = n_classes
        self.max_seq_len = max_seq_len
        self.vocab = os.path.join(model_dir, vocab)
        self.init_ckpt = os.path.join(model_dir, init_ckpt)
        self.config = os.path.join(model_dir, config)

        tokenization.validate_case_matches_checkpoint(True, self.init_ckpt)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab, do_lower_case=True)

    def train(self, df, batch_size=32, lr=2e-5, epochs=100, warmup=0.1, save_int=1000):
        num_train_steps = int(df.shape[0] / batch_size * epochs)
        num_warmup_steps = int(num_train_steps * warmup)

        bert_config = modeling.BertConfig.from_json_file(self.config)

        train_examples = ut.create_examples(df)
        train_features = ut.convert_examples_to_features(train_examples, self.max_seq_len, self.tokenizer)
        train_input_fn = ut.input_fn_builder(features=train_features,
                                             seq_length=self.max_seq_len,
                                             is_training=True,
                                             drop_remainder=True)

        run_config = tf.estimator.RunConfig(
            model_dir=self.out_dir,
            save_summary_steps=save_int,
            keep_checkpoint_max=1,
            save_checkpoints_steps=save_int)
        model_fn = ut.model_fn_builder(
            bert_config=bert_config,
            num_labels=self.n_classes,
            init_checkpoint=self.init_ckpt,
            learning_rate=lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False,
            use_one_hot_embeddings=False)
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": batch_size})

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        return estimator

    def predict(self, features, model):
        predict_examples = ut.create_examples(features, False)
        test_features = ut.convert_examples_to_features(predict_examples, self.max_seq_len, self.tokenizer)
        predict_input_fn = ut.input_fn_builder(features=test_features, seq_length=self.max_seq_len, is_training=False,
                                               drop_remainder=False)
        predictions = model.predict(predict_input_fn)
        output_df = ut.create_output(predictions)

        return output_df

    def eval_folds(self, df, folds, batch_size=32, lr=2e-5, epochs=100, warmup=0.1, save_int=1000):
        kf = KFold(n_splits=folds)
        split_n = -1
        all_gt, all_predictions = pd.DataFrame(), pd.DataFrame()
        for train_index, test_index in kf.split(df):
            split_n += 1
            x_train, x_test = df.iloc[train_index], df.iloc[test_index]

            print(f'Training split {split_n}')
            estimator = self.train(x_train, batch_size=batch_size, lr=lr, epochs=epochs, warmup=warmup, save_int=save_int)

            print(f'Beginning evaluation for split {split_n}')
            x_test = x_test.reset_index(drop=True)
            output_df = pd.concat([x_test.iloc[:, 0], self.predict(x_test, estimator)], axis=1)
            all_gt = pd.concat([all_gt, x_test], axis=0)
            all_predictions = pd.concat([all_predictions, output_df], axis=0)

        full_df = pd.concat([all_gt, all_predictions], axis=1)
        full_df.to_csv(os.path.join(self.out_dir, 'all_pred.csv'), index=False)

        self.score(all_gt, all_predictions, thresh='auto', metrics=['f1', 'precision', 'recall', 'norm_acc'], opt='norm_acc')


    def score(self, gt, pred, thresh, metrics, opt):
        scores = []

        if thresh == 'auto':
            thresholds = [v/100. for v in list(range(0, 100, 5))]
        else:
            thresholds = [thresh]

        gt_vals, pred_vals = gt.iloc[:, 1:], pred.iloc[:, 1:]
        all_gt, all_pred = np.empty((0)), np.empty((0))
        for label, values in gt_vals.iteritems():
            best_score, best_thresh, best_preds = 0, 0, None
            gt_values = values.values
            for t in thresholds:
                pred_values = pred_vals[label].values
                thresh_preds = np.where(pred_values >= t, 1, 0)
                row = [label, t, np.sum(gt_values), np.sum(thresh_preds),
                       np.sum(np.logical_and(gt_values == thresh_preds, gt_values == 1)),
                       np.sum(np.logical_and(gt_values == thresh_preds, gt_values == 0))]
                for m in metrics:
                    evaluator = getattr(eval, m)
                    score = evaluator(gt_values, thresh_preds)
                    if score > best_score and m == opt:
                        best_score, best_thresh, best_preds = score, t, thresh_preds
                    row.append(score)
                scores.append(row)
            all_gt, all_pred = np.concatenate((all_gt, gt_values)), np.concatenate((all_pred, best_preds))

        summary_row = ['all', 'auto', np.sum(all_gt), np.sum(all_pred),
                      np.sum(np.logical_and(all_gt == all_pred, all_pred == 1)),
                      np.sum(np.logical_and(all_gt == all_pred, all_pred == 0))]
        for m in metrics:
            evaluator = getattr(ut, m)
            score = evaluator(all_gt, all_pred)
            summary_row.append(score)
        scores.append(summary_row)

        for t in thresholds:
            thresh_preds = np.where(pred_vals.values >= t, 1, 0)
            row = ['all', t, np.sum(gt_vals.values), np.sum(thresh_preds),
                   np.sum(np.logical_and(gt_vals.values == thresh_preds, gt_vals.values == 1)),
                   np.sum(np.logical_and(gt_vals.values == thresh_preds, gt_vals.values == 0))]
            for m in metrics:
                evaluator = getattr(eval, m)
                score = evaluator(gt_vals.values, thresh_preds)
                row.append(score)
            scores.append(row)

        scores_df = pd.DataFrame(scores,
                                 columns=['label', 'threshold', 'n actual pos', 'n pred pos', 'n pred true pos', 'n pred true neg']
                                         + metrics)
        scores_df.to_csv(os.path.join(self.out_dir, 'acc_scores.csv'), index=False)