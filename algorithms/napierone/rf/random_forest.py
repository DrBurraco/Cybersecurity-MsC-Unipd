import sys
sys.path.append('../../../../../../')

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

import my_dataset_builder.napierone_tiny_no_pdf # Register Dataset

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from joblib import parallel_backend

import numpy as np
import datetime
import csv
from itertools import product
import time


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
        fieldnames = ['Data Type', 'Segments', 'Segments Length', 'Statistics','Accuracy', 'F1_Score', 'Precision', 'Recall']
        randomforest_val_results = []
        randomforest_test_results = []
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

            return {'ent': ent, 'daa': daa, 'label': example['label']}

        IDEAL_FILE = tf.constant(list(range(256)), dtype=tf.float64)
        IDEAL_ENTROPIES = compute_ideal_entropy(IDEAL_FILE)

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

        train_dataset = train.interleave(lambda x: tf.data.Dataset.from_tensors(x).filter(filter_len).map(get_segments, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).map(statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE).cache()

        val_dataset = val.interleave(lambda x: tf.data.Dataset.from_tensors(x).filter(filter_len).map(get_segments, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).map(statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE).cache()

        test_dataset = test.interleave(lambda x: tf.data.Dataset.from_tensors(x).filter(filter_len).map(get_segments, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).map(statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE).cache()

        print('Start DAA training')
        #t0=time.time()
        train_scores = train_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(compute_train_scores, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

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

        best_dists_threshs = get_best_dists_threshs(daa_train_metrics[0])
        d_t_indexes = best_dists_threshs[1]

        for seg,seg_len in product(range(SEG),range(SUB_PARTS)):
            daa_train_results.append({
            'Data Type': 'standard',
            'Segments': seg+1,
            'Segments Length': (seg_len+1)*SUB_LEN,
            'Statistics': 'Dist: '+str(best_dists_threshs[0][seg][0].numpy())+', Thresh: '+str(best_dists_threshs[0][seg][1].numpy()),
            'Accuracy': daa_train_metrics[0][d_t_indexes[seg]][seg_len].numpy(),
            'F1_Score': daa_train_metrics[1][d_t_indexes[seg]][seg_len].numpy(),
            'Precision': daa_train_metrics[2][d_t_indexes[seg]][seg_len].numpy(),
            'Recall': daa_train_metrics[3][d_t_indexes[seg]][seg_len].numpy()})

        #SAVE TEST RESULTS INTO CSV FILE
        print('Saving train results')
        result_csv_file = str(NUM_SAMPLES)+'_'+str(list(range(1,SEG+1)))+'F_'+str(SEG_LEN)+'_daa_train_results.csv'
        with open(histrory_dir + current_time + '_' + result_csv_file, mode='w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(daa_train_results)

        print('Computed best dists and thresh')

        #t0=time.time()
        test_scores = test_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(compute_test_scores, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

        TP = tf.zeros([SEG,SUB_PARTS], dtype=tf.int32)
        FP = tf.zeros([SEG,SUB_PARTS], dtype=tf.int32)
        TN = tf.zeros([SEG,SUB_PARTS], dtype=tf.int32)
        FN = tf.zeros([SEG,SUB_PARTS], dtype=tf.int32)

        for e in test_scores:
            TP = tf.math.add(TP,e[0])
            FP = tf.math.add(FP,e[1])
            TN = tf.math.add(TN,e[2])
            FN = tf.math.add(FN,e[3])

        #t1=time.time()
        #print('time: ',t1-t0)

        print('Computed test scores')
        daa_test_metrics = compute_metrics(TP,FP,TN,FN)

        for seg,seg_len in product(range(SEG),range(SUB_PARTS)):
            daa_test_results.append({
            'Data Type': 'standard',
            'Segments': seg+1,
            'Segments Length': (seg_len+1)*SUB_LEN,
            'Statistics': 'DAA',
            'Accuracy': daa_test_metrics[0][seg][seg_len].numpy(),
            'F1_Score': daa_test_metrics[1][seg][seg_len].numpy(),
            'Precision': daa_test_metrics[2][seg][seg_len].numpy(),
            'Recall': daa_test_metrics[3][seg][seg_len].numpy()})

        #SAVE TEST RESULTS INTO CSV FILE
        print('Saving test results')
        result_csv_file = str(NUM_SAMPLES)+'_'+str(list(range(1,SEG+1)))+'F_'+str(SEG_LEN)+'_daa_test_results.csv'
        with open(histrory_dir + current_time + '_' + result_csv_file, mode='w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(daa_test_results)

        print('Start machine learning training')

        for seg,seg_len,stat in product(range(1,SEG+1),range(1,SUB_PARTS+1),STATISTICS):

            if(stat == 'ENT'):

                @tf.function
                def get_statistics(example):
                    ent = tf.reshape(tf.slice(example['ent'],begin=[0,0],size=[seg,seg_len]), [-1])
                    return {'stats': ent, 'label': example['label']}

            elif(stat == 'DAA'):

                @tf.function
                def get_statistics(example):
                    daa = tf.reshape(tf.slice(example['daa'],begin=[0,0],size=[seg,seg_len]), [-1])
                    return {'stats': daa, 'label': example['label']}
            else:

                @tf.function
                def get_statistics(example):
                    ent = tf.reshape(tf.slice(example['ent'],begin=[0,0],size=[seg,seg_len]), [-1])
                    daa = tf.reshape(tf.slice(example['daa'],begin=[0,0],size=[seg,seg_len]), [-1])
                    stats = tf.concat([ent,daa], 0)
                    return {'stats': stats, 'label': example['label']}


            train_stats = train_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(get_statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

            val_stats = val_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(get_statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

            test_stats = test_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(get_statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

            train_features = list(train_stats.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(lambda y: y['stats'],num_parallel_calls=tf.data.AUTOTUNE), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE))
            train_labels= list(train_stats.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(lambda z: z['label'],num_parallel_calls=tf.data.AUTOTUNE), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE))

            val_features = list(val_stats.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(lambda y: y['stats'],num_parallel_calls=tf.data.AUTOTUNE), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE))
            val_labels= list(val_stats.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(lambda z: z['label'],num_parallel_calls=tf.data.AUTOTUNE), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE))

            test_features = list(test_stats.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(lambda y: y['stats'],num_parallel_calls=tf.data.AUTOTUNE), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE))
            test_labels= list(test_stats.interleave(lambda x: tf.data.Dataset.from_tensors(x).map(lambda z: z['label'],num_parallel_calls=tf.data.AUTOTUNE), cycle_length=CYCLE_LENGTH,num_parallel_calls=tf.data.AUTOTUNE))


            with parallel_backend('threading', n_jobs=24):

                random_forest = RandomForestClassifier()#DecisionTreeClassifier()

                random_forest.fit(train_features,np.ravel(train_labels))

                val_labels_pred = random_forest.predict(val_features)
                val_accuracy = metrics.accuracy_score(val_labels,val_labels_pred)
                val_f1 = metrics.f1_score(val_labels,val_labels_pred)
                val_precision = metrics.precision_score(val_labels,val_labels_pred)
                val_recall = metrics.recall_score(val_labels,val_labels_pred)

                randomforest_val_results.append({
                    'Data Type': 'standard',
                    'Segments': seg,
                    'Segments Length': seg_len*SUB_LEN,
                    'Statistics': stat,
                    'Accuracy': val_accuracy,
                    'F1_Score': val_f1,
                    'Precision': val_precision,
                    'Recall': val_recall})

                test_labels_pred = random_forest.predict(test_features)
                test_accuracy = metrics.accuracy_score(test_labels,test_labels_pred)
                test_f1 = metrics.f1_score(test_labels,test_labels_pred)
                test_precision = metrics.precision_score(test_labels,test_labels_pred)
                test_recall = metrics.recall_score(test_labels,test_labels_pred)

                randomforest_test_results.append({
                    'Data Type': 'standard',
                    'Segments': seg,
                    'Segments Length': seg_len*SUB_LEN,
                    'Statistics': stat,
                    'Accuracy': test_accuracy,
                    'F1_Score': test_f1,
                    'Precision': test_precision,
                    'Recall': test_recall})

    #SAVE VAL RESULTS INTO CSV FILE
    print('Saving validation results')
    result_csv_file = str(NUM_SAMPLES)+'_'+str(list(range(1,SEG+1)))+'F_'+str(SEG_LEN)+'_randomforest_val_results.csv'
    with open(histrory_dir + current_time + '_' + result_csv_file, mode='w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(randomforest_val_results)

    #SAVE TEST RESULTS INTO CSV FILE
    print('Saving test results')
    result_csv_file = str(NUM_SAMPLES)+'_'+str(list(range(1,SEG+1)))+'F_'+str(SEG_LEN)+'_randomforest_test_results.csv'
    with open(histrory_dir + current_time + '_' + result_csv_file, mode='w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(randomforest_test_results)

    return


if __name__ == '__main__':
    main()
