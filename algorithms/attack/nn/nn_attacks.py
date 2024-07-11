import sys
sys.path.append('../../../../../../')

import tensorflow as tf
#tf.config.set_soft_device_placement(True)
#tf.debugging.set_log_device_placement(True)

import tensorflow_datasets as tfds
import tensorflow_probability as tfp

import my_dataset_builder.napierone_tiny_no_pdf # Register Dataset

import datetime
from itertools import product
import json
import os
import numpy as np
import csv
import string

def main():
    with tf.device('/GPU:0'):

        #DEFINE GLOBAL VARIABLES
        NUM_SAMPLES = 'ALL'
        EPOCHS = 500
        BATCH_SIZE = 256

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
        save_dir = './model_save/'
        save_format = '.keras'
        #fieldnames = ['Data Type', 'Segments', 'Segments Length', 'Statistics','Accuracy', 'F1_Score', 'Precision', 'Recall']
        fieldnames = ['Data Type', 'Segments', 'Distance','Threshold','Segments Length','Accuracy', 'F1_Score', 'Precision', 'Recall']
        nn_test_results = []
        daa_train_results = []
        daa_test_results = []

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
            #daa = compute_daa(example['file_segments'])
            return {'ent': ent, 'daa': daa, 'label': example['label']}

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

        IDEAL_FILE = tf.constant(list(range(256)), dtype=tf.float64)
        IDEAL_ENTROPIES = compute_ideal_entropy(IDEAL_FILE)

        METRICS = [
                tf.keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
                tf.keras.metrics.MeanSquaredError(name='Brier score'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn'),
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.F1Score(average='micro',name='f1_score'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
                ]

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

        train_low_ent = train.filter(filter_ransom).map(low_ent, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        val_low_ent = val.filter(filter_ransom).map(low_ent, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        test_low_ent = test.filter(filter_ransom).map(low_ent, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

        train_rep_bytes = train.filter(filter_ransom).map(rep_bytes, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        val_rep_bytes = val.filter(filter_ransom).map(rep_bytes, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        test_rep_bytes = test.filter(filter_ransom).map(rep_bytes, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

        train_com_seq = train.filter(filter_ransom).map(com_seq, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        val_com_seq = val.filter(filter_ransom).map(com_seq, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        test_com_seq = test.filter(filter_ransom).map(com_seq, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

        train_daa = tf.data.Dataset.sample_from_datasets([train,train_low_ent,train_rep_bytes,train_com_seq], rerandomize_each_iteration=True)
        val_daa = tf.data.Dataset.sample_from_datasets([val,val_low_ent,val_rep_bytes,val_com_seq], rerandomize_each_iteration=True)
        test_daa = tf.data.Dataset.sample_from_datasets([test,test_low_ent,test_rep_bytes,test_com_seq], rerandomize_each_iteration=True)

        train_dataset = train_daa.filter(filter_len).map(get_segments, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).map(statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE).cache()#.cache('./cache/daa_attacks_svc_train_stats')

        val_dataset = val_daa.filter(filter_len).map(get_segments, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).map(statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE).cache()#.cache('./cache/daa_attacks_svc_val_stats')

        test_dataset = test_daa.filter(filter_len).map(get_segments, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).map(statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE).cache()#.cache('./cache/daa_attacks_svc_test_stats')

        print('Start DAA training')
        #t0=time.time()
        train_scores = train_dataset.map(compute_train_scores, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

        print('Computing train scores')
        TP = tf.zeros([(SEG-1)*NUM_DISTANCES*NUM_THRESHOLDS+NUM_THRESHOLDS,SUB_PARTS], dtype=tf.int32)
        FP = tf.zeros([(SEG-1)*NUM_DISTANCES*NUM_THRESHOLDS+NUM_THRESHOLDS,SUB_PARTS], dtype=tf.int32)
        TN = tf.zeros([(SEG-1)*NUM_DISTANCES*NUM_THRESHOLDS+NUM_THRESHOLDS,SUB_PARTS], dtype=tf.int32)
        FN = tf.zeros([(SEG-1)*NUM_DISTANCES*NUM_THRESHOLDS+NUM_THRESHOLDS,SUB_PARTS], dtype=tf.int32)

        for e in train_scores:
            TP = tf.math.add(TP,e[0])
            FP = tf.math.add(FP,e[1])
            TN = tf.math.add(TN,e[2])
            FN = tf.math.add(FN,e[3])

        #t1=time.time()
        #print('time: ',t1-t0)


        print('Computed train scores')
        daa_train_metrics = compute_metrics(TP,FP,TN,FN)

        #best_dists_threshs = get_best_dists_threshs(daa_train_metrics[0])
        #d_t_indexes = best_dists_threshs[1]

        #for seg,seg_len in product(range(SEG),range(SUB_PARTS)):
            #daa_train_results.append({
            #'Data Type': 'standard',
            #'Segments': seg+1,
            #'Segments Length': (seg_len+1)*SUB_LEN,
            #'Statistics': 'Dist: '+str(best_dists_threshs[0][seg][0].numpy())+', Thresh: '+str(best_dists_threshs[0][seg][1].numpy()),
            #'Accuracy': daa_train_metrics[0][d_t_indexes[seg]][seg_len].numpy(),
            #'F1_Score': daa_train_metrics[1][d_t_indexes[seg]][seg_len].numpy(),
            #'Precision': daa_train_metrics[2][d_t_indexes[seg]][seg_len].numpy(),
            #'Recall': daa_train_metrics[3][d_t_indexes[seg]][seg_len].numpy()})

        for idx in range(NUM_THRESHOLDS):
            t = THRESHOLDS[idx%NUM_THRESHOLDS]
            for seg_len in range(SUB_PARTS):
                daa_train_results.append({
                    'Data Type': 'napierone',
                    'Segments': 1,
                    'Distance': 0,
                    'Threshold': t.numpy(),
                    'Segments Length': (seg_len+1)*SUB_LEN,
                    'Accuracy': daa_train_metrics[0][idx][seg_len].numpy(),
                    'F1_Score': daa_train_metrics[1][idx][seg_len].numpy(),
                    'Precision': daa_train_metrics[2][idx][seg_len].numpy(),
                    'Recall': daa_train_metrics[3][idx][seg_len].numpy()})

        #print(daa_train_results)

        #for idx in range((SEG-1)*NUM_DISTANCES*NUM_THRESHOLDS):
            #seg = tf.cast(tf.math.floordiv(idx,NUM_DISTANCES*NUM_THRESHOLDS), dtype=tf.int32)
            #d = DISTANCES[tf.cast(tf.math.floordiv(idx,NUM_THRESHOLDS), dtype=tf.int32)%NUM_DISTANCES]
            #t = THRESHOLDS[idx%NUM_THRESHOLDS]
            #print(seg,d,t)
            #for seg_len in range(SUB_PARTS):
                #daa_train_results.append({
                    #'Data Type': 'napierone',
                    #'Segments': seg.numpy()+1,
                    #'Distance': d.numpy(),
                    #'Threshold': t.numpy(),
                    #'Segments Length': (seg_len+1)*SUB_LEN,
                    #'Accuracy': daa_train_metrics[0][idx+NUM_THRESHOLDS][seg_len].numpy(),
                    #'F1_Score': daa_train_metrics[1][idx+NUM_THRESHOLDS][seg_len].numpy(),
                    #'Precision': daa_train_metrics[2][idx+NUM_THRESHOLDS][seg_len].numpy(),
                    #'Recall': daa_train_metrics[3][idx+NUM_THRESHOLDS][seg_len].numpy()})


        #print(daa_train_results)
        #SAVE TEST RESULTS INTO CSV FILE
        print('Saving train results')
        result_csv_file = str(NUM_SAMPLES)+'_'+'header_'+str(SEG_LEN)+'_daa_attacks_train_tot_results.csv'
        with open(histrory_dir + current_time + '_' + result_csv_file, mode='w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(daa_train_results)

        ##SAVE TEST RESULTS INTO CSV FILE
        #print('Saving train results')
        #result_csv_file = str(NUM_SAMPLES)+'_'+str(list(range(1,SEG+1)))+'F_'+str(SEG_LEN)+'_daa_attacks_train_results.csv'
        #with open(histrory_dir + current_time + '_' + result_csv_file, mode='w') as csvfile:
            #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #writer.writeheader()
            #writer.writerows(daa_train_results)

        #print('Computed best dists and thresh')

        ##t0=time.time()
        #test_scores = test_dataset.map(compute_test_scores, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

        #TP = tf.zeros([SEG,SUB_PARTS], dtype=tf.int32)
        #FP = tf.zeros([SEG,SUB_PARTS], dtype=tf.int32)
        #TN = tf.zeros([SEG,SUB_PARTS], dtype=tf.int32)
        #FN = tf.zeros([SEG,SUB_PARTS], dtype=tf.int32)

        #for e in test_scores:
            #TP = tf.math.add(TP,e[0])
            #FP = tf.math.add(FP,e[1])
            #TN = tf.math.add(TN,e[2])
            #FN = tf.math.add(FN,e[3])

        ##t1=time.time()
        ##print('time: ',t1-t0)

        #print('Computed test scores')
        #daa_test_metrics = compute_metrics(TP,FP,TN,FN)

        #for seg,seg_len in product(range(SEG),range(SUB_PARTS)):
            #daa_test_results.append({
            #'Data Type': 'standard',
            #'Segments': seg+1,
            #'Segments Length': (seg_len+1)*SUB_LEN,
            #'Statistics': 'DAA',
            #'Accuracy': daa_test_metrics[0][seg][seg_len].numpy(),
            #'F1_Score': daa_test_metrics[1][seg][seg_len].numpy(),
            #'Precision': daa_test_metrics[2][seg][seg_len].numpy(),
            #'Recall': daa_test_metrics[3][seg][seg_len].numpy()})

        ##SAVE TEST RESULTS INTO CSV FILE
        #print('Saving test results')
        #result_csv_file = str(NUM_SAMPLES)+'_'+str(list(range(1,SEG+1)))+'F_'+str(SEG_LEN)+'_daa_attacks_test_results.csv'
        #with open(histrory_dir + current_time + '_' + result_csv_file, mode='w') as csvfile:
            #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #writer.writeheader()
            #writer.writerows(daa_test_results)

        #print('Start machine learning training')

        #for seg,seg_len,stat in product(range(1,SEG+1),range(1,SUB_PARTS+1),STATISTICS):

            #model_name = 'no_pdf_segmented_input_statistics_'+current_time+'_'+str(seg)+'_'+str(seg_len*SUB_LEN)+'_'+str(stat)
            #checkpoint_dir = 'training_checkpoints'
            #checkpoint_train = os.path.join(checkpoint_dir,current_time,str(seg)+'_'+str(seg_len*SUB_LEN)+'_'+str(stat),'ckpt.weights.h5')
            #log_dir = 'tb_logs'
            #log_prefix= os.path.join(log_dir,current_time,str(seg)+'_'+str(seg_len*SUB_LEN)+'_'+str(stat))


            ##Put all the callbacks together.tf.keras.callbacks.LearningRateScheduler(decay)
            #CALLBACKS = [
                #tf.keras.callbacks.TensorBoard(log_dir=log_prefix),
                #tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_train,save_weights_only=True),
                #tf.keras.callbacks.EarlyStopping(monitor='val_prc',verbose=1,patience=10,mode='max',restore_best_weights=True),
                ##MetricsF1(),
            #]

            #if(stat == 'ENT' or stat == 'DAA'):
                #model = tf.keras.Sequential([
                    #tf.keras.Input(shape=(seg*seg_len)),
                    #tf.keras.layers.Dense(256, activation='relu'),
                    ##tf.keras.layers.Dropout(0.5),
                    #tf.keras.layers.Dense(1, activation='sigmoid'),
                #])
            #else:
                #model = tf.keras.Sequential([
                    #tf.keras.Input(shape=(2*seg*seg_len)),
                    #tf.keras.layers.Dense(256, activation='relu'),
                    ##tf.keras.layers.Dropout(0.5),
                    #tf.keras.layers.Dense(1, activation='sigmoid'),
                #])

            #model.compile(
                #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                #loss=tf.keras.losses.BinaryCrossentropy(),
                #metrics=METRICS
            #)

            #if(stat == 'ENT'):

                #@tf.function
                #def get_statistics(example):
                    #ent = tf.reshape(tf.slice(example['ent'],begin=[0,0],size=[seg,seg_len]), [-1])
                    #return (ent, example['label'])

            #elif(stat == 'DAA'):

                #@tf.function
                #def get_statistics(example):
                    #daa = tf.reshape(tf.slice(example['daa'],begin=[0,0],size=[seg,seg_len]), [-1])
                    #return (daa, example['label'])
            #else:
                #@tf.function
                #def get_statistics(example):
                    #ent = tf.reshape(tf.slice(example['ent'],begin=[0,0],size=[seg,seg_len]), [-1])
                    #daa = tf.reshape(tf.slice(example['daa'],begin=[0,0],size=[seg,seg_len]), [-1])
                    #stats = tf.concat([ent,daa], 0)
                    #return (stats,example['label'])

            #train_ransom_stats = train_dataset.filter(filter_ransom).map(get_statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            #train_legit_stats = train_dataset.filter(filter_legit).map(get_statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            #train_stats = tf.data.Dataset.sample_from_datasets([train_ransom_stats,train_legit_stats], stop_on_empty_dataset=True,rerandomize_each_iteration=True).prefetch(tf.data.AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True,num_parallel_calls=tf.data.AUTOTUNE)

            #val_stats = val_dataset.map(get_statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE).batch(BATCH_SIZE,drop_remainder=True,num_parallel_calls=tf.data.AUTOTUNE)

            #test_stats = test_dataset.map(get_statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE).batch(BATCH_SIZE,drop_remainder=True,num_parallel_calls=tf.data.AUTOTUNE)

            #model.fit(train_stats,epochs=EPOCHS,callbacks=CALLBACKS,validation_data=val_stats)

            #results = model.evaluate(test_stats,callbacks=CALLBACKS,return_dict=True)

            #if not os.path.exists(save_dir):
                #os.makedirs(save_dir)
            #model.save(save_dir + model_name + save_format)

            #nn_test_results.append({
                    #'Data Type': 'attack_daa',
                    #'Segments': seg,
                    #'Segments Length': seg_len*SUB_LEN,
                    #'Statistics': stat,
                    #'Accuracy': results['accuracy'],
                    #'F1_Score': results['f1_score'],
                    #'Precision': results['precision'],
                    #'Recall': results['recall']})

    ##SAVE TEST RESULTS INTO CSV FILE
    #print('Saving test results')
    #result_csv_file = current_time+ '_' +str(NUM_SAMPLES)+'_'+str(list(range(1,SEG+1)))+'F_'+str(SEG_LEN)+'_daa_attacks_nn_test_results.csv'
    #with open(histrory_dir + result_csv_file, mode='w') as csvfile:
        #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #writer.writeheader()
        #writer.writerows(nn_test_results)

    return


if __name__ == '__main__':
    main()
