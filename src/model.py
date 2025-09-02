import os
import sys
import cv2
import editdistance
import time

from typing import List, Tuple
from path import Path

import numpy as np
import tensorflow as tf

from dataset import Batch, Dataset
from metric import CharacterErrorRate, Loss, PhraseAccuracy, WordErrorRate
from settings import Settings
from preprocessor import Preprocessor
from summary_writer import SummaryWriter, EpochSummary

class DecoderType:
    """CTC decoder types."""
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class Model:
    """Minimalistic TF model for HTR."""

    def __init__(self,
                 char_list: List[str],
                 decoder_type: str = DecoderType.BestPath,
                 must_restore: bool = False,
                 dump: bool = False) -> None:
        """Init model: add CNN, RNN and CTC and initialize TF."""
        self.dump = dump
        self.char_list = char_list
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.snap_ID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # input image batch
        self.input_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

        # setup CNN, RNN and CTC
        self.setup_cnn()
        self.setup_rnn()
        self.setup_ctc()

        # setup optimizer to train NN
        self.batches_trained = 0
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        # initialize TF
        self.sess, self.saver = self.setup_tf()

    def setup_cnn(self) -> None:
        """Create CNN layers."""
        cnn_in4d = tf.expand_dims(input=self.input_imgs, axis=3)

        # list of parameters for the layers
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)

        # create layers
        pool = cnn_in4d  # input to first CNN layer
        for i in range(num_layers):
            kernel = tf.Variable(
                tf.random.truncated_normal([kernel_vals[i], kernel_vals[i], feature_vals[i], feature_vals[i + 1]],
                                           stddev=0.1))
            conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool2d(input=relu, ksize=(1, pool_vals[i][0], pool_vals[i][1], 1),
                                    strides=(1, stride_vals[i][0], stride_vals[i][1], 1), padding='VALID')

        self.cnn_out_4d = pool

    def setup_rnn(self) -> None:
        """Create RNN layers."""
        rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=[2])

        # basic cells which is used to build RNN
        num_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True) for _ in
                 range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnn_in3d,
                                                                dtype=rnn_in3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.random.truncated_normal([1, 1, num_hidden * 2, len(self.char_list) + 1], stddev=0.1))
        self.rnn_out_3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'),
                                     axis=[2])

    def setup_ctc(self) -> None:
        """Create CTC loss and decoder."""
        # BxTxC -> TxBxC
        self.ctc_in_3d_tbc = tf.transpose(a=self.rnn_out_3d, perm=[1, 0, 2])
        # ground truth text as sparse tensor
        self.gt_texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                        tf.compat.v1.placeholder(tf.int32, [None]),
                                        tf.compat.v1.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seq_len = tf.compat.v1.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            input_tensor=tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.ctc_in_3d_tbc,
                                                  sequence_length=self.seq_len,
                                                  ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.saved_ctc_input = tf.compat.v1.placeholder(tf.float32,
                                                        shape=[None, None, len(self.char_list) + 1])
        self.loss_per_element = tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.saved_ctc_input,
                                                         sequence_length=self.seq_len, ctc_merge_repeated=True)

        # best path decoding or beam search decoding
        if self.decoder_type == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len)
        elif self.decoder_type == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len,
                                                         beam_width=50)
        # word beam search decoding (see https://github.com/githubharald/CTCWordBeamSearch)
        elif self.decoder_type == DecoderType.WordBeamSearch:
            # prepare information about language (dictionary, characters in dataset, characters forming words)
            chars = ''.join(self.char_list)
            word_chars = open(Settings.WORD_CHAR_LIST_FILE_PATH).read().splitlines()[0]
            corpus = open(Settings.CORPUS_FILE_PATH).read()

            # decode using the "Words" mode of word beam search
            from word_beam_search import WordBeamSearch
            self.decoder = WordBeamSearch(50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
                                          word_chars.encode('utf8'))

            # the input to the decoder must have softmax already applied
            self.wbs_input = tf.nn.softmax(self.ctc_in_3d_tbc, axis=2)

    def setup_tf(self) -> Tuple[tf.compat.v1.Session, tf.compat.v1.train.Saver]:
        """Initialize TF."""
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.compat.v1.Session()  # TF session

        saver = tf.compat.v1.train.Saver(max_to_keep=1)  # saver saves model to file
        
        latest_snapshot = tf.train.latest_checkpoint(Settings.MODEL_DIR)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.must_restore and not latest_snapshot:
            raise Exception('No saved model found in: ' + Settings.MODEL_DIR)

        # load saved model if available
        if latest_snapshot:
            print('Init with stored values from ' + latest_snapshot)
            saver.restore(sess, latest_snapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return sess, saver

    def to_sparse(self, texts: List[str]) -> Tuple[List[List[int]], List[int], List[int]]:
        """Put ground truth texts into sparse tensor for ctc_loss."""
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for batchElement, text in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_str = [self.char_list.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # put each label into sparse tensor
            for i, label in enumerate(label_str):
                indices.append([batchElement, i])
                values.append(label)

        return indices, values, shape

    def decoder_output_to_text(self, ctc_output: tuple, batch_size: int) -> List[str]:
        """Extract texts from output of CTC decoder."""

        # word beam search: already contains label strings
        if self.decoder_type == DecoderType.WordBeamSearch:
            label_strs = ctc_output

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctc_output[0][0]

            # contains string of labels for each batch element
            label_strs = [[] for _ in range(batch_size)]

            # go over all indices and save mapping: batch -> values
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batch_element = idx2d[0]  # index according to [b,t]
                label_strs[batch_element].append(label)

        # map labels to chars for all batch elements
        return [''.join([self.char_list[c] for c in labelStr]) for labelStr in label_strs]

    def train_batch(self, batch: Batch) -> float:
        """Feed a batch into the NN to train it."""
        num_batch_elements = len(batch.imgs)
        max_text_len = batch.imgs[0].shape[0] // 4
        sparse = self.to_sparse(batch.gt_texts)
        eval_list = [self.optimizer, self.loss]
        feed_dict = {self.input_imgs: batch.imgs, self.gt_texts: sparse,
                     self.seq_len: [max_text_len] * num_batch_elements, self.is_train: True}
        _, loss_val = self.sess.run(eval_list, feed_dict)
        self.batches_trained += 1
        return loss_val

    @staticmethod
    def dump_nn_output(rnn_output: np.ndarray) -> None:
        """Dump the output of the NN to CSV file(s)."""
        if not os.path.isdir(Settings.DUMP_DIR):
            os.mkdir(Settings.DUMP_DIR)

        # iterate over all batch elements and create a CSV file for each one
        max_t, max_b, max_c = rnn_output.shape
        for b in range(max_b):
            csv = ''
            for t in range(max_t):
                csv += ';'.join([str(rnn_output[t, b, c]) for c in range(max_c)]) + ';\n'
            fn = Settings.DUMP_DIR + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

    def infer_batch(self, batch: Batch, calc_probability: bool = False, probability_of_gt: bool = False):
        """Feed a batch into the NN to recognize the texts."""

        # decode, optionally save RNN output
        num_batch_elements = len(batch.imgs)

        # put tensors to be evaluated into list
        eval_list = []
        
        if self.decoder_type == DecoderType.WordBeamSearch:
            eval_list.append(self.wbs_input)
        else:
            eval_list.append(self.decoder)

        if self.dump or calc_probability:
            eval_list.append(self.ctc_in_3d_tbc)

        eval_list.append(self.loss)

        # sequence length depends on input image size (model downsizes width by 4)
        max_text_len = batch.imgs[0].shape[0] // 4

        # dict containing all tensor fed into the model
        feed_dict = {self.input_imgs: batch.imgs, self.gt_texts: self.to_sparse(batch.gt_texts), 
                     self.seq_len: [max_text_len] * num_batch_elements, self.is_train: False}

        # evaluate model
        eval_res = self.sess.run(eval_list, feed_dict)
        loss = eval_res[2]

        # TF decoders: decoding already done in TF graph
        if self.decoder_type != DecoderType.WordBeamSearch:
            decoded = eval_res[0]
        # word beam search decoder: decoding is done in C++ function compute()
        else:
            decoded = self.decoder.compute(eval_res[0])

        # map labels (numbers) to character string
        texts = self.decoder_output_to_text(decoded, num_batch_elements)

        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calc_probability:
            sparse = self.to_sparse(batch.gt_texts) if probability_of_gt else self.to_sparse(texts)
            ctc_input = eval_res[1]
            eval_list = self.loss_per_element
            feed_dict = {self.saved_ctc_input: ctc_input, self.gt_texts: sparse,
                         self.seq_len: [max_text_len] * num_batch_elements, self.is_train: False}
            loss_vals = self.sess.run(eval_list, feed_dict)
            probs = np.exp(-loss_vals)

        # dump the output of the NN to CSV file(s)
        if self.dump:
            self.dump_nn_output(eval_res[1])

        return texts, probs, loss

    def save(self) -> None:
        """Save model to file."""
        self.snap_ID += 1
        self.saver.save(self.sess, '../model/snapshot', global_step=self.snap_ID)

    def train(self,
            train_set: Dataset,
            validation_set: Dataset,
            early_stopping: int = 25,
            max_epoch: int = 42) -> None:
        """Trains NN."""
        epoch = 0  # number of training epochs since start
        summary_writer = SummaryWriter(Settings.SUMMARY_FILE_PATH)

        train_loss_in_epoch = []

        best_char_error_rate = float('inf')  # best validation character error rate
        no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
        # stop training after this number of epochs without improvement
        while True:
            epoch += 1
            print('Epoch:', epoch)
            if epoch > max_epoch:
                print(f'Max epoch {max_epoch} reached. Trainig stopped')
                break

            # train
            print('Train NN')
            start_time = time.time()
            train_set.reset_iterator()
            while train_set.has_next():
                iter_info = train_set.get_iterator_info()
                batch = train_set.get_next()
                loss = self.train_batch(batch)
                print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')
                train_loss_in_epoch.append(loss)

            train_CER, train_WER, train_phrase_acc, _ = self.validate(train_set)
            end_time = time.time()
            # validate
            val_CER, val_WER, val_phrase_acc, val_loss = self.validate(validation_set)

            # write summary
            epoch_summary = EpochSummary(
                epoch = epoch,
                train_cer=train_CER,
                val_cer = val_CER,
                train_wer=train_WER,
                val_wer= val_WER,
                train_phrase_acc=train_phrase_acc,
                val_phrase_acc = val_phrase_acc,
                train_loss = sum(train_loss_in_epoch) / len(train_loss_in_epoch),
                val_loss=val_loss,
                time_to_train_epoch = end_time-start_time
            )
            summary_writer.append(epoch_summary)

            # reset train loss list
            train_loss_in_epoch = []

            # if best validation accuracy so far, save model parameters
            if val_CER < best_char_error_rate:
                print('Character error rate improved, save model')
                best_char_error_rate = val_CER
                no_improvement_since = 0
                self.save()
            else:
                print(f'Character error rate not improved, best so far: {best_char_error_rate * 100.0}%')
                no_improvement_since += 1

            # stop training if no more improvement in the last x epochs
            if no_improvement_since >= early_stopping:
                print(f'No more improvement for {early_stopping} epochs. Training stopped.')
                break


    def validate(self, validation_set: Dataset) -> Tuple[float, float, float]:
        """Validates NN."""
        print('Validate NN')
        validation_set.reset_iterator()
        cer_metric = CharacterErrorRate()
        wer_metric = WordErrorRate()
        phrase_acc_metric = PhraseAccuracy()
        loss_metric = Loss()
        
        while validation_set.has_next():
            iter_info = validation_set.get_iterator_info()
            print(f'Batch: {iter_info[0]} / {iter_info[1]}')
            batch = validation_set.get_next()
            recognized, _, batch_loss = self.infer_batch(batch)
            loss_metric.update_state(batch_loss)

            print('Ground truth -> Recognized')
            for i in range(len(recognized)):
                phrase_acc_metric.update_state(batch.gt_texts[i], recognized[i])
                wer_metric.update_state(batch.gt_texts[i], recognized[i])
                cer_metric.update_state(batch.gt_texts[i], recognized[i])
                
                print(cer_metric.get_info(),
                      wer_metric.get_info(),
                      phrase_acc_metric.get_info(),
                      f"{batch.gt_texts[i]} -> {recognized[i]}")

        # print validation result
        char_error_rate = cer_metric.result()
        word_error_rate = wer_metric.result()
        phrase_accuracy = phrase_acc_metric.result()
        val_loss = loss_metric.result()
        cer_metric.reset_states()
        wer_metric.reset_states()
        phrase_acc_metric.reset_states()
        loss_metric.reset_states()
        print(f'Character error rate: {char_error_rate * 100.0}%.')
        print(f'Word error rate: {word_error_rate * 100.0}%')
        print(f'Phrase accuracy: {phrase_accuracy * 100.0}%.')
        return char_error_rate, word_error_rate, phrase_accuracy, val_loss


    def infer(self, fn_img: Path) -> None:
        """Recognizes text in image provided by file path."""
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        assert img is not None

        preprocessor = Preprocessor(self.get_img_size(), dynamic_width=True, padding=16)
        img = preprocessor.process_img(img)

        batch = Batch([img], None, 1)
        recognized, probability, _ = self.infer_batch(batch, True)
        print(f'Recognized: "{recognized[0]}"')
        print(f'Probability: {probability[0]}')
