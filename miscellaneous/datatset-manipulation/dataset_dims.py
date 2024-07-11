import sys
sys.path.append('../../../../../../')

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

import my_dataset_builder.napierone_tiny_no_pdf # Register Dataset
import numpy as np
import datetime
import csv
from itertools import product
import time
import string

def main():
    #mirrored_strategy = tf.distribute.MirroredStrategy() #MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    with tf.device('/GPU:0'):

        #DEFINE GLOBAL VARIABLES
        NUM_SAMPLES = 'ALL'

        SEG_LEN = 256
        SUB_LEN = 8
        SUB_PARTS = int(SEG_LEN/SUB_LEN)
        SEG = 4
        DISTANCES = tf.constant(list(range(129)), dtype=tf.float64)
        NUM_DISTANCES = len(DISTANCES)
        THRESHOLDS = tf.constant(list(range(129)), dtype=tf.float64)
        NUM_THRESHOLDS = len(THRESHOLDS)
        STATISTICS = ['ENT','DAA','ENT+DAA']
        best_dists_threshs = []


        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        histrory_dir = './history/'
        fieldnames = ['Dataset', 'Split', 'Filter Length', 'Total', 'Ransomware', 'Legitimate']
        dataset_dims = []

        CYCLE_LENGTH = 2048

        g = tf.random.Generator.from_non_deterministic_state()

        @tf.function(reduce_retracing=True)
        def entropy(labels):

            n_labels = tf.shape(labels)[0]

            if n_labels <= 1:
                return tf.constant(0., dtype=tf.float64)

            value, _, counts = tf.unique_with_counts(labels)
            probs = tf.math.divide(counts, n_labels)
            n_classes = tf.math.count_nonzero(probs)

            if n_classes <= 1:
                return tf.constant(0., dtype=tf.float64)

            ent = tf.constant(0., dtype=tf.float64)

            # Compute binary entropy
            i = tf.constant(0)
            c = lambda i, ent: tf.less(i, tf.shape(probs)[0])
            b = lambda i, ent: [i+1, tf.math.subtract(ent,tf.math.multiply(probs[i],tf.math.divide(tf.math.log(probs[i]),tf.math.log(tf.constant(2., dtype=tf.float64)))))]

            output = tf.while_loop(
                c, b, loop_vars=[i, ent],
                shape_invariants=[i.get_shape(), ent.get_shape()])[1]

            return output

        @tf.function
        def compute_ideal_entropy(file):
            ent = tf.cast(tf.reshape((), (0,)), dtype=tf.float64)
            for j in range(1,SUB_PARTS+1):
                ent = tf.concat([ent, [entropy(file[:j*SUB_LEN])]], 0)
            return ent

        @tf.function
        def filter_len(example):

            if (tf.shape(example['file'])[0] >= SEG_LEN * SEG):
                return True

            return False


        @tf.function
        def get_segments(example):

            file = example['file']
            file_len = tf.shape(file)[0]
            file_segments = tf.reshape(tf.slice(file, begin=[0], size=[SEG_LEN]), [1, SEG_LEN])

            n = tf.constant(0)
            i = g.uniform(shape=(), minval=0, maxval=file_len - SEG_LEN, dtype=tf.int32)
            c = lambda n, i, file_segs, file: tf.less(n, SEG-1)
            b = lambda n, i, file_segs, file: [n+1, g.uniform(shape=(), minval=0, maxval=file_len - SEG_LEN, dtype=tf.int32), tf.concat([file_segs, tf.reshape(tf.slice(file, begin=[i], size=[SEG_LEN]), [1, SEG_LEN])], axis=0), file]

            output = tf.while_loop(
                c, b, loop_vars=[n, i, file_segments, file],
                shape_invariants=[n.get_shape(), tf.TensorShape([]), tf.TensorShape([None, SEG_LEN]), file.get_shape()])[2]

            return {'file_segments': output, 'label':example['label']}

        @tf.function
        def compute_entropy(file_segs):

            ents = tf.cast(tf.reshape((), (0,SUB_PARTS)), dtype=tf.float64)
            for i in range(SEG):
                ent = tf.cast(tf.reshape((), (0,)), dtype=tf.float64)
                for j in range(1,SUB_PARTS+1):
                    ent = tf.concat([ent, [entropy(file_segs[i][:j*SUB_LEN])]], 0)
                ents = tf.concat([ents, [ent]], axis=0)
            return ents


        @tf.function
        def compute_daa(segs_ents):
            daas = tf.cast(tf.reshape((), (0,SUB_PARTS)), dtype=tf.float64)
            da = tf.math.abs(tf.math.subtract(segs_ents, IDEAL_ENTROPIES))

            for i in range(SEG):
                daa = tf.cast(tf.reshape((), (0,)), dtype=tf.float64)
                for j in range(1,SUB_PARTS+1):
                    daa = tf.concat([daa, [tfp.math.trapz(tf.concat([[0], da[i][:j]], 0), dx=SUB_LEN)]], 0)
                daas =  tf.concat([daas, [daa]], axis=0)
            return daas


        @tf.function
        def statistics(example):
            ent = compute_entropy(example['file_segments'])
            daa = compute_daa(ent)

            return {'ent': ent, 'daa': daa, 'label': example['label']}

        IDEAL_FILE = tf.constant(list(range(256)), dtype=tf.float64)
        IDEAL_ENTROPIES = compute_ideal_entropy(IDEAL_FILE)

        @tf.function
        def low_ent_random(example):

            random_byte = g.uniform(shape=(), minval=0, maxval=255, dtype=tf.int32)
            random_bytes = tf.constant([random_byte for i in range(256)], dtype=tf.uint8)

            modified_file = tf.concat([random_bytes,example['file']], 0)

            return {'file': modified_file, 'label': example['label']}

        @tf.function
        def low_ent(example):

            rep_a = tf.constant([97 for i in range(256)], dtype=tf.uint8)
            modified_file = tf.concat([rep_a,example['file']], 0)

            return {'file': modified_file, 'label': example['label']}

        @tf.function
        def rep_bytes(example):
            first_8_bytes = tf.cast(tf.reshape((), (0,)), dtype=tf.uint8)

            for i in range(32):
                first_8_bytes = tf.concat([first_8_bytes, example['file'][:8]], 0)

            #first_8_bytes = [example['file'][:8] for i in range(32)]
            modified_file = tf.concat([first_8_bytes,example['file']], 0)

            return {'file': modified_file, 'label': example['label']}

        @tf.function
        def com_seq(example):

            up_case_abc = tf.constant([ord(c) for c in string.ascii_uppercase], dtype = tf.uint8)
            low_case_abc = tf.constant([ord(c) for c in string.ascii_lowercase], dtype = tf.uint8)
            common_words = tf.constant([ord(c) for word in ['the','of','to','and','a','in','is','it','you','that','he','was','for'] for c in word], dtype = tf.uint8)
            numbers = tf.constant(list(range(22)), dtype = tf.uint8)
            zeros = tf.constant([0 for i in range(32)], dtype = tf.uint8)
            modified_file = tf.concat([up_case_abc,low_case_abc,common_words,numbers,zeros, example['file']], 0)

            return {'file': modified_file, 'label': example['label']}

        @tf.function
        def filter_ransom(example):
            if (example['label'] != 0.):
                return True
            return False

        @tf.function
        def filter_legit(example):
            if (example['label'] == 0.):
                return True
            return False

        @tf.function
        def compute_train_scores(example):

            FINAL_DIM = (SEG-1)*NUM_DISTANCES*NUM_THRESHOLDS
            ones = tf.ones([FINAL_DIM, SUB_PARTS], dtype=tf.int32)
            zeros = tf.zeros([FINAL_DIM, SUB_PARTS], dtype=tf.int32)
            h_ones = tf.ones([NUM_THRESHOLDS, SUB_PARTS], dtype=tf.int32)
            h_zeros = tf.zeros([NUM_THRESHOLDS, SUB_PARTS], dtype=tf.int32)

            TP = tf.cast(tf.reshape((), (0,SUB_PARTS)), dtype=tf.int32)
            FP = tf.cast(tf.reshape((), (0,SUB_PARTS)), dtype=tf.int32)
            TN = tf.cast(tf.reshape((), (0,SUB_PARTS)), dtype=tf.int32)
            FN = tf.cast(tf.reshape((), (0,SUB_PARTS)), dtype=tf.int32)

            DAAs = example['daa']
            label = example['label']
            label_check = label == 1.
            header = DAAs[0]
            t = tf.transpose(tf.expand_dims(THRESHOLDS,0))
            h_check = tf.less_equal(header,t)
            h_mask_under = tf.where(h_check, h_ones, h_zeros)
            h_mask_over = tf.where(h_check, h_zeros, h_ones)

            if(label_check):
                TP = tf.concat([TP, h_mask_under],0)
                FN = tf.concat([FN, h_mask_over],0)
                TN = tf.concat([TN, h_zeros],0)
                FP = tf.concat([FP, h_zeros],0)
            else:
                TP = tf.concat([TP, h_zeros],0)
                FN = tf.concat([FN, h_zeros],0)
                TN = tf.concat([TN, h_mask_over],0)
                FP = tf.concat([FP, h_mask_under],0)

            if (SEG > 1):
                d = tf.transpose(tf.expand_dims(DISTANCES,0))
                header_dist = tf.math.subtract(header,d)

                avg = tf.cast(tf.reshape((), (0,SUB_PARTS)), dtype=tf.float64)
                for i in range(2, SEG+1):
                    avg = tf.concat([avg, [tf.reduce_mean(DAAs[1:i], 0)]],0)

                avg = tf.repeat(avg, repeats=[NUM_DISTANCES], axis=0)
                min_areas = tf.where(tf.less(tf.tile(header_dist, [SEG-1,1]), avg), tf.tile(tf.reshape(header, [1, SUB_PARTS]), [(SEG-1)*NUM_DISTANCES,1]), avg)
                #min_areas = tf.minimum(tf.tile(header_dist, [SEG-1,1]), avg)
                min_areas_expanded = tf.repeat(min_areas, repeats=[NUM_THRESHOLDS], axis=0)
                t_expanded = tf.tile(t, [(SEG-1)*NUM_DISTANCES,1])
                thresh_check = tf.less_equal(min_areas_expanded, t_expanded)

                if(label_check):
                    tp = tf.where(thresh_check, ones, zeros)
                    fn = tf.where(thresh_check, zeros, ones)
                    tn = zeros
                    fp = zeros
                else:
                    tp = zeros
                    fn = zeros
                    tn = tf.where(thresh_check, zeros, ones)
                    fp = tf.where(thresh_check, ones, zeros)

                TP = tf.concat([TP, tp],0)
                FN = tf.concat([FN, fn],0)
                TN = tf.concat([TN, tn],0)
                FP = tf.concat([FP, fp],0)

            return TP,FP,TN,FN

        @tf.function
        def get_best_dists_threshs(acc):
            dists_threshs = tf.cast(tf.reshape((), (0,2)), dtype=tf.float64)
            indexes =  tf.cast(tf.reshape((), (0,)), dtype=tf.int32)

            #max_acc = tf.squeeze(tf.math.top_k(acc,k=1)[0])
            max_acc = tf.reduce_mean(acc, 1)
            idx = tf.squeeze(tf.math.top_k(max_acc[:NUM_THRESHOLDS],k=1)[1])
            indexes = tf.concat([indexes, [idx]],0)
            t_h = THRESHOLDS[idx]
            dists_threshs = tf.concat([dists_threshs, [[0,t_h]]],0)

            for i in range(NUM_THRESHOLDS, (SEG-1)*NUM_DISTANCES*NUM_THRESHOLDS, NUM_DISTANCES*NUM_THRESHOLDS):
                idx = tf.squeeze(tf.math.top_k(max_acc[i:i+NUM_DISTANCES*NUM_THRESHOLDS],k=1)[1])
                indexes = tf.concat([indexes, [idx+i]],0)
                d = DISTANCES[tf.cast(tf.math.floordiv(idx,NUM_THRESHOLDS), dtype=tf.int32)]
                t = THRESHOLDS[idx%NUM_THRESHOLDS]
                dists_threshs = tf.concat([dists_threshs, [[d,t,]]],0)

            return (dists_threshs,indexes)

        @tf.function
        def compute_test_scores(example):

            ones = tf.ones([SEG, SUB_PARTS], dtype=tf.int32)
            zeros = tf.zeros([SEG, SUB_PARTS], dtype=tf.int32)
            DAAs = example['daa']
            label = example['label']
            d,t = tf.transpose(tf.expand_dims(best_dists_threshs[0][:, 0], 0)),tf.transpose(tf.expand_dims(best_dists_threshs[0][:, 1], 0))
            label_check = label == 1.

            segments_dist = tf.math.subtract(DAAs,d)
            thresh_check = tf.less_equal(segments_dist, t)

            if(label_check):
                tp = tf.where(thresh_check, ones, zeros)
                fn = tf.where(thresh_check, zeros, ones)
                tn = zeros
                fp = zeros
            else:
                tp = zeros
                fn = zeros
                tn = tf.where(thresh_check, zeros, ones)
                fp = tf.where(thresh_check, ones, zeros)

            return (tp,fp,tn,fn)

        @tf.function
        def compute_metrics(TP,FP,TN,FN):
            acc = tf.math.divide(tf.math.add(TP,TN),tf.math.add_n([TP,FP,TN,FN]))
            precision = tf.math.divide(TP,tf.math.add(TP,FP))
            recall = tf.math.divide(TP,tf.math.add(TP,FN))
            f1_score = tf.math.scalar_mul(2, tf.math.divide(tf.math.multiply(precision,recall),tf.math.add(precision,recall)))
            return acc,f1_score,precision,recall


        datasets, info = tfds.load('napierone_tiny_no_pdf',
                                data_dir='/mydatasets',#'../../../../../../my_datasets/downloads/manual/Tiny',#/mydatasets
                                split=['train[:60%]', 'train[60%:80%]','train[80%:100%]'],
                                as_supervised=False,
                                with_info=True,)

        train, val, test = datasets[0],datasets[1],datasets[2]

        print(info)



        #NAPIERONE DATASET##
        train_neg, train_pos = np.bincount(tf.squeeze(list(train.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        train_total = train_neg + train_pos
        dataset_dims.append({
            'Dataset': 'napierone',
            'Split': 'train',
            'Filter Length': 'no',
            'Total': train_total,
            'Ransomware': train_pos,
            'Legitimate': train_neg})

        val_neg, val_pos = np.bincount(tf.squeeze(list(val.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        val_total = val_neg + val_pos
        dataset_dims.append({
            'Dataset': 'napierone',
            'Split': 'val',
            'Filter Length': 'no',
            'Total': val_total,
            'Ransomware': val_pos,
            'Legitimate': val_neg})

        test_neg, test_pos = np.bincount(tf.squeeze(list(test.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        test_total = test_neg + test_pos
        dataset_dims.append({
            'Dataset': 'napierone',
            'Split': 'test',
            'Filter Length': 'no',
            'Total': test_total,
            'Ransomware': test_pos,
            'Legitimate': test_neg})




        ##NAPIERONE FILTER LENGTH DATASET##
        train_dataset = train.filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        train_neg, train_pos = np.bincount(tf.squeeze(list(train_dataset)))
        train_total = train_neg + train_pos
        dataset_dims.append({
            'Dataset': 'napierone',
            'Split': 'train',
            'Filter Length': 'yes',
            'Total': train_total,
            'Ransomware': train_pos,
            'Legitimate': train_neg})

        val_dataset = val.filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_neg, val_pos = np.bincount(tf.squeeze(list(val_dataset)))
        val_total = val_neg + val_pos
        dataset_dims.append({
            'Dataset': 'napierone',
            'Split': 'val',
            'Filter Length': 'yes',
            'Total': val_total,
            'Ransomware': val_pos,
            'Legitimate': val_neg})

        test_dataset = test.filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        test_neg, test_pos = np.bincount(tf.squeeze(list(test_dataset)))
        test_total = test_neg + test_pos
        dataset_dims.append({
            'Dataset': 'napierone',
            'Split': 'test',
            'Filter Length': 'yes',
            'Total': test_total,
            'Ransomware': test_pos,
            'Legitimate': test_neg})




        ##FILTERED LOW ENTROPY DATASET##
        train_low_ent = train.filter(filter_ransom).map(low_ent, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        train_neg, train_pos = np.bincount(tf.squeeze(list(train_low_ent)))
        train_total = train_neg + train_pos
        dataset_dims.append({
            'Dataset': 'low entropy',
            'Split': 'train',
            'Filter Length': 'yes',
            'Total': train_total,
            'Ransomware': train_pos,
            'Legitimate': train_neg})

        val_low_ent = val.filter(filter_ransom).map(low_ent, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_neg, val_pos = np.bincount(tf.squeeze(list(val_low_ent)))
        val_total = val_neg + val_pos
        dataset_dims.append({
            'Dataset': 'low entropy',
            'Split': 'val',
            'Filter Length': 'yes',
            'Total': val_total,
            'Ransomware': val_pos,
            'Legitimate': val_neg})

        test_low_ent = test.filter(filter_ransom).map(low_ent, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        test_neg, test_pos = np.bincount(tf.squeeze(list(test_low_ent)))
        test_total = test_neg + test_pos
        dataset_dims.append({
            'Dataset': 'low entropy',
            'Split': 'test',
            'Filter Length': 'yes',
            'Total': test_total,
            'Ransomware': test_pos,
            'Legitimate': test_neg})

        ##FILTERED REPETION BYTES DATASET##
        train_rep_bytes = train.filter(filter_ransom).map(rep_bytes, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        train_neg, train_pos = np.bincount(tf.squeeze(list(train_rep_bytes)))
        train_total = train_neg + train_pos
        dataset_dims.append({
            'Dataset': 'repetion bytes',
            'Split': 'train',
            'Filter Length': 'yes',
            'Total': train_total,
            'Ransomware': train_pos,
            'Legitimate': train_neg})

        val_rep_bytes = val.filter(filter_ransom).map(rep_bytes, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_neg, val_pos = np.bincount(tf.squeeze(list(val_rep_bytes)))
        val_total = val_neg + val_pos
        dataset_dims.append({
            'Dataset': 'repetion bytes',
            'Split': 'val',
            'Filter Length': 'yes',
            'Total': val_total,
            'Ransomware': val_pos,
            'Legitimate': val_neg})

        test_rep_bytes = test.filter(filter_ransom).map(rep_bytes, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        test_neg, test_pos = np.bincount(tf.squeeze(list(test_rep_bytes)))
        test_total = test_neg + test_pos
        dataset_dims.append({
            'Dataset': 'repetion bytes',
            'Split': 'test',
            'Filter Length': 'yes',
            'Total': test_total,
            'Ransomware': test_pos,
            'Legitimate': test_neg})

        ##FILTERED COMMON SEQUENCE DATASET##
        train_com_seq = train.filter(filter_ransom).map(com_seq, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        train_neg, train_pos = np.bincount(tf.squeeze(list(train_com_seq)))
        train_total = train_neg + train_pos
        dataset_dims.append({
            'Dataset': 'common sequence',
            'Split': 'train',
            'Filter Length': 'yes',
            'Total': train_total,
            'Ransomware': train_pos,
            'Legitimate': train_neg})

        val_com_seq = val.filter(filter_ransom).map(com_seq, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_neg, val_pos = np.bincount(tf.squeeze(list(val_com_seq)))
        val_total = val_neg + val_pos
        dataset_dims.append({
            'Dataset': 'common sequence',
            'Split': 'val',
            'Filter Length': 'yes',
            'Total': val_total,
            'Ransomware': val_pos,
            'Legitimate': val_neg})

        test_com_seq = test.filter(filter_ransom).map(com_seq, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        test_neg, test_pos = np.bincount(tf.squeeze(list(test_com_seq)))
        test_total = test_neg + test_pos
        dataset_dims.append({
            'Dataset': 'common sequence',
            'Split': 'test',
            'Filter Length': 'yes',
            'Total': test_total,
            'Ransomware': test_pos,
            'Legitimate': test_neg})



        ##LOW ENTROPY DATASET##
        train_low_ent = train.filter(filter_ransom).map(low_ent, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        train_neg, train_pos = np.bincount(tf.squeeze(list(train_low_ent.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        train_total = train_neg + train_pos
        dataset_dims.append({
            'Dataset': 'low entropy',
            'Split': 'train',
            'Filter Length': 'no',
            'Total': train_total,
            'Ransomware': train_pos,
            'Legitimate': train_neg})

        val_low_ent = val.filter(filter_ransom).map(low_ent, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        val_neg, val_pos = np.bincount(tf.squeeze(list(val_low_ent.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        val_total = val_neg + val_pos
        dataset_dims.append({
            'Dataset': 'low entropy',
            'Split': 'val',
            'Filter Length': 'no',
            'Total': val_total,
            'Ransomware': val_pos,
            'Legitimate': val_neg})

        test_low_ent = test.filter(filter_ransom).map(low_ent, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        test_neg, test_pos = np.bincount(tf.squeeze(list(test_low_ent.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        test_total = test_neg + test_pos
        dataset_dims.append({
            'Dataset': 'low entropy',
            'Split': 'test',
            'Filter Length': 'no',
            'Total': test_total,
            'Ransomware': test_pos,
            'Legitimate': test_neg})

        ##REPETION BYTES DATASET##
        train_rep_bytes = train.filter(filter_ransom).map(rep_bytes, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        train_neg, train_pos = np.bincount(tf.squeeze(list(train_rep_bytes.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        train_total = train_neg + train_pos
        dataset_dims.append({
            'Dataset': 'repetion bytes',
            'Split': 'train',
            'Filter Length': 'no',
            'Total': train_total,
            'Ransomware': train_pos,
            'Legitimate': train_neg})

        val_rep_bytes = val.filter(filter_ransom).map(rep_bytes, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        val_neg, val_pos = np.bincount(tf.squeeze(list(val_rep_bytes.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        val_total = val_neg + val_pos
        dataset_dims.append({
            'Dataset': 'repetion bytes',
            'Split': 'val',
            'Filter Length': 'no',
            'Total': val_total,
            'Ransomware': val_pos,
            'Legitimate': val_neg})

        test_rep_bytes = test.filter(filter_ransom).map(rep_bytes, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        test_neg, test_pos = np.bincount(tf.squeeze(list(test_rep_bytes.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        test_total = test_neg + test_pos
        dataset_dims.append({
            'Dataset': 'repetion bytes',
            'Split': 'test',
            'Filter Length': 'no',
            'Total': test_total,
            'Ransomware': test_pos,
            'Legitimate': test_neg})

        ##COMMON SEQUENCE DATASET##
        train_com_seq = train.filter(filter_ransom).map(com_seq, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        train_neg, train_pos = np.bincount(tf.squeeze(list(train_com_seq.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        train_total = train_neg + train_pos
        dataset_dims.append({
            'Dataset': 'common sequence',
            'Split': 'test',
            'Filter Length': 'no',
            'Total': train_total,
            'Ransomware': train_pos,
            'Legitimate': train_neg})

        val_com_seq = val.filter(filter_ransom).map(com_seq, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        val_neg, val_pos = np.bincount(tf.squeeze(list(val_com_seq.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        val_total = val_neg + val_pos
        dataset_dims.append({
            'Dataset': 'common sequence',
            'Split': 'val',
            'Filter Length': 'no',
            'Total': val_total,
            'Ransomware': val_pos,
            'Legitimate': val_neg})

        test_com_seq = test.filter(filter_ransom).map(com_seq, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        test_neg, test_pos = np.bincount(tf.squeeze(list(test_com_seq.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        test_total = test_neg + test_pos
        dataset_dims.append({
            'Dataset': 'common sequence',
            'Split': 'test',
            'Filter Length': 'no',
            'Total': test_total,
            'Ransomware': test_pos,
            'Legitimate': test_neg})




        ##ATTACK DAA##
        train_daa = tf.data.Dataset.sample_from_datasets([train,train_low_ent,train_rep_bytes,train_com_seq], rerandomize_each_iteration=True)
        train_neg, train_pos = np.bincount(tf.squeeze(list(train_daa.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        train_total = train_neg + train_pos
        dataset_dims.append({
            'Dataset': 'attackdaa',
            'Split': 'train',
            'Filter Length': 'no',
            'Total': train_total,
            'Ransomware': train_pos,
            'Legitimate': train_neg})

        val_daa = tf.data.Dataset.sample_from_datasets([val,val_low_ent,val_rep_bytes,val_com_seq], rerandomize_each_iteration=True)
        val_neg, val_pos = np.bincount(tf.squeeze(list(val_daa.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        val_total = val_neg + val_pos
        dataset_dims.append({
            'Dataset': 'attackdaa',
            'Split': 'val',
            'Filter Length': 'no',
            'Total': val_total,
            'Ransomware': val_pos,
            'Legitimate': val_neg})

        test_daa = tf.data.Dataset.sample_from_datasets([test,test_low_ent,test_rep_bytes,test_com_seq], rerandomize_each_iteration=True)
        test_neg, test_pos = np.bincount(tf.squeeze(list(test_daa.map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE))))
        test_total = test_neg + test_pos
        dataset_dims.append({
            'Dataset': 'attackdaa',
            'Split': 'test',
            'Filter Length': 'no',
            'Total': test_total,
            'Ransomware': test_pos,
            'Legitimate': test_neg})




        ##ATTACKDAA FILTER LENGTH##
        train_dataset = train_daa.filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        train_neg, train_pos = np.bincount(tf.squeeze(list(train_dataset)))
        train_total = train_neg + train_pos
        dataset_dims.append({
            'Dataset': 'attackdaa',
            'Split': 'train',
            'Filter Length': 'yes',
            'Total': train_total,
            'Ransomware': train_pos,
            'Legitimate': train_neg})

        val_dataset = val_daa.filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_neg, val_pos = np.bincount(tf.squeeze(list(val_dataset)))
        val_total = val_neg + val_pos
        dataset_dims.append({
            'Dataset': 'attackdaa',
            'Split': 'val',
            'Filter Length': 'yes',
            'Total': val_total,
            'Ransomware': val_pos,
            'Legitimate': val_neg})

        test_dataset = test_daa.filter(filter_len).map(lambda x: tf.cast(x['label'], tf.int32), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        test_neg, test_pos = np.bincount(tf.squeeze(list(test_dataset)))
        test_total = test_neg + test_pos
        dataset_dims.append({
            'Dataset': 'attackdaa',
            'Split': 'test',
            'Filter Length': 'yes',
            'Total': test_total,
            'Ransomware': test_pos,
            'Legitimate': test_neg})

        #SAVE TEST RESULTS INTO CSV FILE
        print('Saving dims')
        result_csv_file = current_time+ '_' +str(NUM_SAMPLES)+'_'+str(list(range(1,SEG+1)))+'F_'+str(SEG_LEN)+'_dataset_dims.csv'
        with open(result_csv_file, mode='w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset_dims)

    return


if __name__=='__main__':
    main()
