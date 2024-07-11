import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import os
import re

ml_directories = [
    '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf/segmented_input/statistics/linear_svc/history/Ten_Runs',
    '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf/segmented_input/statistics/nn/history/Ten_Runs',
    '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf/segmented_input/statistics/randomforest/history/Ten_Runs',
    '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf_low_entropy_ransom/segmented_input/statistics/linear_svc/history/Ten Runs',
    '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf_low_entropy_ransom/segmented_input/statistics/nn/history/Ten_Runs',
    '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf_low_entropy_ransom/segmented_input/statistics/randomforest/history/Ten_Runs'
    ]

daa_directories = [
    '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf_low_entropy_ransom/segmented_input/statistics/daa',
    '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf/segmented_input/statistics/daa',
    ]


# Define the columns and titles for the plots
metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']
statistics = ['ENT','DAA','ENT+DAA']

# Function to select rows where any column matches a specific value
def select_rows_by_value(df, value):
    # Select rows where any column contains the value
    selected_rows = df.loc[df['Segments'] == value]
    return selected_rows

# Function to select rows where any column matches a specific value
def select_rows_by_seg_dist(df, seg, dist):
    # Select rows where any column contains the value
    selected_rows = df.loc[df['Segments'] == seg & df['Distance'] == dist]
    return selected_rows

def select_rows_by_seg_thresh(df, seg, thresh):
    # Select rows where any column contains the value
    selected_rows = df.loc[df['Segments'] == seg & df['Threshold'] == thresh]
    return selected_rows

def plot_daa_header_metrics(df, csv_pdf):

    #segs_len = list(range(32,257,32))
    segs_len = [32,48,56,128,152,192,256]
    #segs_len.insert(0,24)

    # Ensure 'Segments Length' and 'Statistics' columns are present
    if 'Segments Length' not in df.columns:
        print("Columns 'Segments Length' not found in DataFrame")
        return

    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    xticks = list(range(0,121,10))
    xticks.append(128)

    for i, metric in enumerate(metrics):
        if metric in df.columns:
            for seg_len in segs_len:
                subset = df.loc[df['Segments Length'] == seg_len]
                axs[i].plot(subset['Threshold'], subset[metric]*100, label=seg_len) #, marker='o'
            axs[i].set_title(metric)
            axs[i].set_xlabel('Threshold (Bit-Bytes)')
            axs[i].set_ylabel(metric + '(%)')
            axs[i].set_ylim(0, 100)  # Set y-axis range from 0.0 to 1.0
            axs[i].set_yticks(np.arange(0, 101, 5))
            axs[i].set_xlim(0, 128)    # Set x-axis range from 8 to 256
            axs[i].set_xticks(xticks)
            axs[i].legend(title='Segment Length')
            #axs[i].legend(bbox_to_anchor=(1.0, 1), loc="upper left",title='Segment Length')#,prop={'size': 8})
            axs[i].grid(True)
        else:
            print(f"Column '{metric}' not found in DataFrame")

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a PDF file
    plt.savefig(csv_pdf)
    #plt.show()

    # Close the plot to free memory
    plt.close()

    return

def plot_daa_randomsegments_metrics(df=None, value=None, distance=None, threshold=None, csv_pdf=None, segs_lens=None):
    # Select rows containing the specified value
    #segs_len = list(range(32,257,32))
    if segs_lens is None: segs_lens = [40,72]

    # Generate each plot
    if distance is not None:
        # Create subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))
        selected_rows = select_rows_by_seg_dist(df, value, distance)
        # Ensure 'Segments Length' and 'Statistics' columns are present
        if 'Segments Length' not in selected_rows.columns:
            print("Columns 'Segments Length' not found in DataFrame")
            return
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                for seg_len in segs_len:
                    subset = selected_rows[selected_rows['Segments Length'] == seg_len]
                    axs[i].plot(subset['Threshold'], subset[metric]*100, label=seg_len) #, marker='o'
                axs[i].set_title(metric)
                axs[i].set_xlabel('Bit-Byte Area')
                axs[i].set_ylabel(metric)
                axs[i].set_ylim(-0.05, 105)  # Set y-axis range from 0.0 to 1.0
                axs[i].set_yticks(np.arange(0, 110, 20))
                axs[i].set_xlim(0, 130)    # Set x-axis range from 8 to 256
                axs[i].set_xticks(range(0, 130, 8))
                axs[i].legend(bbox_to_anchor=(1.0, 1), loc="upper left",title='Segment Length')#,prop={'size': 8})
                axs[i].grid(True)
            else:
                print(f"Column '{metric}' not found in DataFrame")

        # Adjust layout
        plt.tight_layout()

        # Save the plot to a PDF file
        #plt.savefig(csv_pdf)
        plt.show()

        # Close the plot to free memory
        plt.close()

        return

    if threshold is not None:
        # Create subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))
        selected_rows = select_rows_by_seg_dist(df, value, threshold)
        # Ensure 'Segments Length' and 'Statistics' columns are present
        if 'Segments Length' not in selected_rows.columns:
            print("Columns 'Segments Length' not found in DataFrame")
            return
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                for seg_len in segs_len:
                    subset = selected_rows[selected_rows['Segments Length'] == seg_len]
                    axs[i].plot(subset['Distance'], subset[metric]*100, label=seg_len) #, marker='o'
                axs[i].set_title(metric)
                axs[i].set_xlabel('Bytes')
                axs[i].set_ylabel(metric)
                axs[i].set_ylim(-0.05, 105)  # Set y-axis range from 0.0 to 1.0
                axs[i].set_yticks(np.arange(0, 110, 20))
                axs[i].set_xlim(0, 130)    # Set x-axis range from 8 to 256
                axs[i].set_xticks(range(0, 130, 8))
                axs[i].legend(bbox_to_anchor=(1.0, 1), loc="upper left",title='Segment Length')#,prop={'size': 8})
                axs[i].grid(True)
            else:
                print(f"Column '{metric}' not found in DataFrame")

        # Adjust layout
        plt.tight_layout()

        # Save the plot to a PDF file
        #plt.savefig(csv_pdf)
        plt.show()

        # Close the plot to free memory
        plt.close()
        return

    selected_rows = select_rows_by_value(df, value)

    for seg_len in segs_lens:
        fig = plt.figure(figsize=(12, 10))#figsize=plt.figaspect(0.5)
        fig.suptitle(f'Segments Length {seg_len}')
        subset = selected_rows[selected_rows['Segments Length'] == seg_len]
        X = subset['Threshold'].unique()
        Y = subset['Distance'].unique()
        X, Y = np.meshgrid(X, Y)
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax = fig.add_subplot(2, 2, i+1, projection='3d')
                ax.set_title(metric, y=1)
                # Plot the surface.
                Z = subset.pivot_table(metric, 'Distance', 'Threshold',dropna=False)
                surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=0, vmax=100)
                ax.set_xlabel('Distance')
                ax.set_ylabel('Threshold')
                ax.set_xticks(range(0, 129, 16))
                ax.set_xlim(0, 128)
                ax.set_ylim(0, 128)
                ax.set_yticks(range(0, 129, 16))
                ax.set_zlim(0, 100)
                ax.set_zticks(range(0,101,10))
                #ax.zaxis.set_major_locator(LinearLocator(10))
                # A StrMethodFormatter is used automatically
                #ax.zaxis.set_major_formatter('{x:.02f}')

                # Add a color bar which maps values to colors.
                fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)


        # set the spacing between subplots
        #plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.5,wspace=0.4,hspace=0.4)
        plt.tight_layout()
        #plt.show()
        #print(csv_pdf[:-5]+str(seg_len)+csv_pdf[-6:])
        plt.savefig(csv_pdf[:-5]+str(seg_len)+csv_pdf[-6:])
        plt.close()
    return



# Function to plot 4 line graphs with different colors based on 'Statistics' column
def plot_ml_metrics(df, value, save_dir, ml_model=None,csv_pdf=None):
    # Select rows containing the specified value
    selected_rows = select_rows_by_value(df, value)

    # Ensure 'Segments Length' and 'Statistics' columns are present
    if 'Segments Length' not in selected_rows.columns or 'Statistics' not in selected_rows.columns:
        print("Columns 'Segments Length' or 'Statistics' not found in DataFrame")
        return

    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    x_ticks = list(range(8, 257, 16))
    x_ticks.append(256)

    # Generate each plot
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            for statistic in selected_rows['Statistics'].unique(): #df['Segments Length'].unique()[range(7,32,8)]:
                subset = selected_rows.loc[selected_rows['Statistics'] == statistic]
                axs[i].plot(subset['Segments Length'], subset[metric]*100, label=statistic) #, marker='o'
            #axs[i].set_title(metric)
            if (value == 1):
                axs[i].set_xlabel('Header Length')
            else:
                axs[i].set_xlabel('Segments Length')
            axs[i].set_ylabel(metric + ' (%)')
            axs[i].set_ylim(-0.05, 100)  # Set y-axis range from 0.0 to 1.0
            axs[i].set_yticks(np.arange(0, 101, 5))
            axs[i].set_xlim(0, 264)    # Set x-axis range from 8 to 256
            axs[i].set_xticks(x_ticks)
            axs[i].tick_params(axis='x', which='major', labelsize=8)
            #axs[i].legend(bbox_to_anchor=(1.0, 1), loc="upper left",title='Statistics')#,prop={'size': 8})
            if 'daa' not in save_dir:
                axs[i].legend(title='Statistics')#,prop={'size': 8})
            axs[i].grid(True)
        else:
            print(f"Column '{metric}' not found in DataFrame")

    plt.tight_layout()
    #plt.show()
    plt.savefig(save_dir)
    plt.close()

    #color = {'ENT':'r', 'DAA':'g', 'ENT+DAA':'b'}
    #for statistic in statistics:
        #mean_subset = selected_rows.loc[selected_rows['Statistics'] == 'MEAN '+statistic]
        #std_subset = selected_rows.loc[selected_rows['Statistics'] == 'STD '+statistic]
        ##minus_std_subset = selected_rows.loc[selected_rows['Statistics'] == 'MINUS_STD '+statistic]
        ##print(mean_subset,std_subset)
        #fig, axs = plt.subplots(4, 1, figsize=(10, 20))
        ##fig.suptitle(save_dir, y=1)
        #x_ticks = list(range(8, 257, 16))
        #x_ticks.append(256)
        #print(save_dir +' - '+ ml_model + ' - '+ str(value)+'F'+' '+statistic)
        #for i, metric in enumerate(metrics):
            #if metric in df.columns:
                ##axs[i].plot(mean_subset['Segments Length'], mean_subset[metric]*100, c=color[statistic])
                #axs[i].errorbar(mean_subset['Segments Length'], mean_subset[metric]*100, yerr=std_subset[metric]*100 , c=color[statistic], ecolor='gray', capsize=3) #, marker='o'
                ##axs[i].plot(plus_std_subset['Segments Length'], plus_std_subset[metric]*100, label='STD '+statistic) #, marker='o'
                ##axs[i].plot(minus_std_subset['Segments Length'], minus_std_subset[metric]*100, label='STD '+statistic) #, marker='o'
                ##axs[i].set_title(save_dir +' - '+ml_model)
                #if (value == 1):
                    #axs[i].set_xlabel('Header Length')
                #else:
                    #axs[i].set_xlabel('Segments Length')
                #axs[i].set_ylabel(metric+' (%)')
                #axs[i].set_ylim(-0.05, 100)  # Set y-axis range from 0.0 to 1.0
                #axs[i].set_yticks(np.arange(0, 101, 5))
                #axs[i].set_xlim(0, 264)    # Set x-axis range from 8 to 256
                #axs[i].set_xticks(x_ticks)
                #axs[i].tick_params(axis='x', which='major', labelsize=8)
                ##axs[i].legend(bbox_to_anchor=(1.0, 1), loc="upper left",title=statistic)#,prop={'size': 8})
                #axs[i].grid(True)
            #else:
                #print(f"Column '{metric}' not found in DataFrame")

        ## Adjust layout
        ##plt.title(ml_model)
        #plt.tight_layout()

        ## Save the plot to a PDF file
        #plt.savefig(os.path.join(csv_pdf, save_dir +'-'+ ml_model + '-'+ str(value)+'F'+'-'+statistic+ '.pdf'))
        ##plt.show()

        ## Close the plot to free memory
        #plt.close()
    return

def daa_run(directory_in_str):
    directory = os.fsencode(directory_in_str)
    csv_pdfs = []
    if ('no_pdf' == re.search('tiny/(.*)/segmented_input', directory_in_str).group(1)):
        save_dir = 'napierone'
        indexes = [6]
    else:
        save_dir = 'attackdaa'
        indexes = [7]

    for file in os.listdir(directory):
        csv_file_path = os.fsdecode(file)
        if('[2, 3, 4]' in csv_file_path):
            df = pd.read_csv(os.path.join(directory_in_str, csv_file_path))
            df.loc[:, 'Segments'] = df['Segments'] + 1
            for metric in metrics:
                df.loc[:, metric] = df[metric] * 100
            for value_to_find in df['Segments'].unique():
                csv_pdf = [csv_file_path.split('_')[i] for i in indexes]
                file_pdf = [save_dir,'daa',csv_pdf[0],str(value_to_find)+'F']
                csv_pdf = '_'.join(map(str, file_pdf))
                idx = csv_pdfs.count(csv_pdf) + 1
                csv_pdfs.append(csv_pdf)
                csv_pdf = csv_pdf +'_'+str(idx)+'.pdf'
                plot_daa_randomsegments_metrics(df=df, value=value_to_find, csv_pdf=os.path.join('.',save_dir,'daa',csv_pdf))
    return

def daa_max_f1(directory_in_str):
    directory = os.fsencode(directory_in_str)
    csv_pdfs = []
    if ('no_pdf' == re.search('tiny/(.*)/segmented_input', directory_in_str).group(1)):
        save_dir = 'napierone'
        indexes = [6]
    else:
        save_dir = 'attackdaa'
        indexes = [7]

    df_1f = pd.DataFrame()
    df_2f = pd.DataFrame()
    df_3f = pd.DataFrame()
    df_4f = pd.DataFrame()
    runs = []
    i=0
    files = [ele for ele in os.listdir(directory) if ele != b'Excess']
    #print(os.listdir(directory))
    #print(files)
    for file in files:
        if os.path.isfile(os.path.join(directory,file)):
            #print(file)
            csv_file_path = os.fsdecode(file)
            df = pd.read_csv(os.path.join(directory_in_str, csv_file_path))
            if(save_dir == 'attackdaa'):
                df.loc[:, 'Data Type'] = 'attackdaa'
            for metric in metrics:
                df.loc[:, metric] = df[metric] * 100
            if('[2, 3, 4]' in csv_file_path):
                df.loc[:, 'Segments'] = df['Segments'] + 1
            for value_to_find in df['Segments'].unique():
                selected_rows = select_rows_by_value(df, value_to_find)
                max_f1 = selected_rows.loc[selected_rows['F1_Score'] == selected_rows['F1_Score'].max()]
                runs.append({
                    'file': file,
                    'run': i,
                    'segment': value_to_find,
                    'f1_score': max_f1['F1_Score'].unique()[0]})
                max_f1.insert(0,"run", i, True)
                match (value_to_find):
                    case 1:
                        df_1f = pd.concat([df_1f, max_f1])
                    case 2:
                        df_2f = pd.concat([df_2f, max_f1])
                    case 3:
                        df_3f = pd.concat([df_3f, max_f1])
                    case 4:
                        df_4f = pd.concat([df_4f, max_f1])
            i+=1

    #for run in runs:
        #print(f"File name: {run['file']},\n Order of pick: {run['run']},\n  Segments:{run['segment']}\n   F1 Score: {run['f1_score']}")

    #print(df_1f[['Distance','Threshold','Segments Length']],df_2f[['Distance','Threshold','Segments Length']],df_3f[['Distance','Threshold','Segments Length']],df_4f[['Distance','Threshold','Segments Length']])
    max_f1_1f = df_1f.loc[df_1f['F1_Score'] == df_1f['F1_Score'].max()]
    max_f1_2f = df_2f.loc[df_2f['F1_Score'] == df_2f['F1_Score'].max()]
    max_f1_3f = df_3f.loc[df_3f['F1_Score'] == df_3f['F1_Score'].max()]
    max_f1_4f = df_4f.loc[df_4f['F1_Score'] == df_4f['F1_Score'].max()]
    #print(max_f1_1f,max_f1_2f,max_f1_3f,max_f1_4f)
    print(max_f1_1f[['Distance','Threshold','Segments Length']],max_f1_2f[['Distance','Threshold','Segments Length']],max_f1_3f[['Distance','Threshold','Segments Length']],max_f1_4f[['Distance','Threshold','Segments Length']])

    df_1f = pd.DataFrame()
    df_2f = pd.DataFrame()
    df_3f = pd.DataFrame()
    df_4f = pd.DataFrame()
    for file in files:
        if os.path.isfile(os.path.join(directory,file)):
            #print(file)
            csv_file_path = os.fsdecode(file)
            df = pd.read_csv(os.path.join(directory_in_str, csv_file_path))
            if(save_dir == 'attackdaa'):
                df.loc[:, 'Data Type'] = 'attackdaa'
            for metric in metrics:
                df.loc[:, metric] = df[metric] * 100
            if('[2, 3, 4]' in csv_file_path):
                df.loc[:, 'Segments'] = df['Segments'] + 1
            for segment in df['Segments'].unique():
                    match (segment):
                        case 1:
                            distance = max_f1_1f['Distance'].values[0]
                            threshold = max_f1_1f['Threshold'].values[0]
                            seg_len = max_f1_1f['Segments Length'].values[0]
                        case 2:
                            distance = max_f1_2f['Distance'].values[0]
                            threshold = max_f1_2f['Threshold'].values[0]
                            seg_len = max_f1_2f['Segments Length'].values[0]
                        case 3:
                            distance = max_f1_3f['Distance'].values[0]
                            threshold = max_f1_3f['Threshold'].values[0]
                            seg_len = max_f1_3f['Segments Length'].values[0]
                        case 4:
                            distance = max_f1_4f['Distance'].values[0]
                            threshold = max_f1_4f['Threshold'].values[0]
                            seg_len = max_f1_4f['Segments Length'].values[0]
                    print(segment,distance,threshold,seg_len)
                    selected_row = df.loc[(df['Segments'] == segment) & (df['Distance'] == distance) & (df['Threshold'] == threshold) & (df['Segments Length'] == seg_len)]
                    match (segment):
                        case 1:
                            df_1f = pd.concat([df_1f, selected_row])
                        case 2:
                            df_2f = pd.concat([df_2f, selected_row])
                        case 3:
                            df_3f = pd.concat([df_3f, selected_row])
                        case 4:
                            df_4f = pd.concat([df_4f, selected_row])

    print(df_1f.mean(numeric_only=True),df_2f.mean(numeric_only=True),df_2f.std(numeric_only=True),df_3f.mean(numeric_only=True),df_3f.std(numeric_only=True),df_4f.mean(numeric_only=True),df_4f.std(numeric_only=True))

    idxs = [max_f1_2f['run'].unique()[0],max_f1_3f['run'].unique()[0],max_f1_4f['run'].unique()[0]]
    seg_len = [max_f1_2f['Segments Length'].unique()[0],max_f1_3f['Segments Length'].unique()[0],max_f1_4f['Segments Length'].unique()[0]]
    selected_files = [files[i] for i in idxs]
    value_to_find = 2
    for idx, file in enumerate(selected_files):
        csv_file_path = os.fsdecode(file)
        df = pd.read_csv(os.path.join(directory_in_str, csv_file_path))
        if(save_dir == 'attackdaa'):
            df.loc[:, 'Data Type'] = 'attackdaa'
        for metric in metrics:
            df.loc[:, metric] = df[metric] * 100
        df.loc[:, 'Segments'] = df['Segments'] + 1
        #print(df.loc[(df['Segments'] == value_to_find) & (df['Segments Length'] == seg_len[idx])]['F1_Score'].max())
        csv_pdf = [csv_file_path.split('_')[i] for i in indexes]
        file_pdf = [save_dir,'daa',csv_pdf[0],str(value_to_find)+'F']
        csv_pdf = '_'.join(map(str, file_pdf))
        idx = csv_pdfs.count(csv_pdf) + 1
        csv_pdfs.append(csv_pdf)
        csv_pdf = csv_pdf +'_'+str(idx)+'.pdf'
        plot_daa_randomsegments_metrics(df=df, value=value_to_find, csv_pdf=os.path.join('.',save_dir,'daa',csv_pdf), segs_lens=[seg_len[idx]])
        value_to_find += 1

    return

def machinelearning_run(directory_in_str):
    directory = os.fsencode(directory_in_str)
    csv_pdfs = []
    if ('no_pdf' == re.search('tiny/(.*)/segmented_input', directory_in_str).group(1)):
        save_dir = 'napierone'
        indexes = [4,5]
    else:
        save_dir = 'attackdaa'
        indexes = [6,7]
    ml_algorithm = re.search('statistics/(.*)/history', directory_in_str).group(1)
    ml_model = '_'
    match(ml_algorithm):
        case 'linear_svc':
            ml_model = 'svc'
        case 'randomforest':
            ml_model = 'randomforest'
        case 'nn':
            ml_model = 'nn'
        case '_':
            print('Not machine learning algorithms found')
            return

    total_df = pd.DataFrame()
    for file in os.listdir(directory):
        csv_file_path = os.fsdecode(file)

        if ml_model in csv_file_path:
            df = pd.read_csv(os.path.join(directory_in_str, csv_file_path))
            df = df.replace("DAA", "DA", regex=False)
            df = df.replace("ENT+DAA", "ENT+DA", regex=False)
            if ml_model == 'nn':
                for index, row in df.iterrows():
                    #print(row['F1_Score'])#df.loc[5, 'Name'] = 'SHIV CHANDRA'
                    if (row['Precision']+row['Recall'] != 0):
                        df.loc[index, 'F1_Score'] = 2*row['Precision']*row['Recall']/(row['Precision']+row['Recall'])
                    else:
                        df.loc[index, 'F1_Score'] = None
            if save_dir == 'attackdaa':
                df.loc[:, 'Data Type'] = 'daa_attacks'
            total_df = pd.concat([total_df,df])


        #df = pd.read_csv(os.path.join(directory_in_str, csv_file_path))
        #for value_to_find in df['Segments'].unique():
            #csv_pdf = [csv_file_path.split('_')[i] for i in indexes]
            ##csv_pdf.insert(0, str(value_to_find)+'F')
            #if save_dir == 'attackdaa':
                #if ml_model in csv_pdf:
                    #file_pdf = [save_dir,ml_model,csv_pdf[1],str(value_to_find)+'F']
                #else:
                    #file_pdf = [save_dir,'daa',csv_pdf[0],str(value_to_find)+'F']
            #else:
                #file_pdf = [save_dir,csv_pdf[0],csv_pdf[1],str(value_to_find)+'F']
            #csv_pdf = '_'.join(map(str, file_pdf))
            #idx = csv_pdfs.count(csv_pdf) + 1
            #csv_pdfs.append(csv_pdf)
            #csv_pdf = csv_pdf +'_'+str(idx)+'.pdf'
            ##print(os.path.join('.',save_dir,ml_model,csv_pdf))
            #plot_ml_metrics(df, value_to_find, os.path.join('.',save_dir,ml_model,csv_pdf))
    stats_df = pd.DataFrame()
    for segment in total_df['Segments'].unique():
        for seg_len in total_df['Segments Length'].unique():
            subset = total_df.loc[(total_df['Segments'] == segment) & (total_df['Segments Length'] == seg_len)]
            for statistic in subset['Statistics'].unique():
                stat_df = subset.loc[subset['Statistics'] == statistic]
                if('nn' not in ml_model):
                    mean = pd.DataFrame([[
                        save_dir,
                        segment,
                        seg_len,
                        'MEAN '+statistic,
                        stat_df['Accuracy'].mean(skipna=False),
                        stat_df['F1_Score'].mean(skipna=False),
                        stat_df['Precision'].mean(skipna=False),
                        stat_df['Recall'].mean(skipna=False)]],
                        columns=['Data Type','Segments','Segments Length','Statistics','Accuracy','F1_Score','Precision','Recall'])
                    std = pd.DataFrame([[
                        save_dir,
                        segment,
                        seg_len,
                        'STD '+statistic,
                        stat_df['Accuracy'].std(skipna=False),
                        stat_df['F1_Score'].std(skipna=False),
                        stat_df['Precision'].std(skipna=False),
                        stat_df['Recall'].std(skipna=False)]],
                        columns=['Data Type','Segments','Segments Length','Statistics','Accuracy','F1_Score','Precision','Recall'])
                else:
                    mean = pd.DataFrame([[
                            save_dir,
                            segment,
                            seg_len,
                            'MEAN '+statistic,
                            stat_df['Accuracy'].mean(skipna=False),
                            2*stat_df['Precision'].mean(skipna=False)*stat_df['Recall'].mean(skipna=False)/(stat_df['Precision'].mean(skipna=False)+stat_df['Recall'].mean(skipna=False)),
                            stat_df['Precision'].mean(skipna=False),
                            stat_df['Recall'].mean(skipna=False)]],
                            columns=['Data Type','Segments','Segments Length','Statistics','Accuracy','F1_Score','Precision','Recall'])
                    std = pd.DataFrame([[
                        save_dir,
                        segment,
                        seg_len,
                        'STD '+statistic,
                        stat_df['Accuracy'].std(skipna=False),
                        2*stat_df['Precision'].std(skipna=False)*stat_df['Recall'].std(skipna=False)/(stat_df['Precision'].std(skipna=False)+stat_df['Recall'].std(skipna=False)),
                        stat_df['Precision'].std(skipna=False),
                        stat_df['Recall'].std(skipna=False)]],
                        columns=['Data Type','Segments','Segments Length','Statistics','Accuracy','F1_Score','Precision','Recall'])
                stats_df = pd.concat([stats_df,mean,std])

    df_max_f1 = pd.DataFrame()
    output_values = []
    print(ml_model)
    for segment in stats_df['Segments'].unique():
        for statistic in stats_df['Statistics'].unique():
            selected_rows = stats_df.loc[stats_df['Segments'] == segment]
            #print(selected_rows)
            if 'MEAN' in statistic:
                mean_rows = selected_rows.loc[selected_rows['Statistics'] == statistic]
                max_f1_mean_row = mean_rows.loc[mean_rows['F1_Score'] == mean_rows['F1_Score'].max()]
                #print(max_f1_mean_row)
                output_values.append({
                    'dataset': max_f1_mean_row['Data Type'].iloc[0],
                    'model': ml_model,
                    'segments': max_f1_mean_row['Segments'].iloc[0],
                    'segments length': max_f1_mean_row['Segments Length'].iloc[0],
                    'statistic': max_f1_mean_row['Statistics'].iloc[0],
                    'f1_score': max_f1_mean_row['F1_Score'].iloc[0]*100,
                    })
                std_rows = selected_rows.loc[selected_rows['Statistics'] == statistic.replace('MEAN', 'STD')]
                max_f1_std_row = std_rows.loc[std_rows['Segments Length'] == max_f1_mean_row['Segments Length'].iloc[0]]
                #print(max_f1_std_row)
                output_values.append({
                    'dataset': max_f1_std_row['Data Type'].iloc[0],
                    'model': ml_model,
                    'segments': max_f1_std_row['Segments'].iloc[0],
                    'segments length': max_f1_std_row['Segments Length'].iloc[0],
                    'statistic': max_f1_std_row['Statistics'].iloc[0],
                    'f1_score': max_f1_std_row['F1_Score'].iloc[0]*100,
                    })
                #print(selected_rows.loc[selected_rows['Statistics'] == statistic.replace('MEAN', 'STD')])
                #print(selected_rows.loc[(selected_rows['Segments Length'] == max_f1_row['Segments Length'].iloc[0]) & (selected_rows['Statistics'] == statistic.replace('MEAN', 'STD'))])
        #plot_ml_metrics(stats_df,segment, save_dir, ml_model, os.path.join('.', save_dir,ml_model))

    return output_values

def divide_df(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df_header = df.loc[4128:]
    #print(df_header)
    df_header.to_csv('/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf_low_entropy_ransom/segmented_input/statistics/daa/20240611-175737_ALL_[2, 3, 4]F_256_daa_attacks_train_tot_results.csv', encoding='utf-8', index=False, na_rep='nan')

    return

def plot_best_f1_ml(f1_mean, f1_std, save_dir):
    seg1_mean = [f1_mean[x] for x in [0,4,8]]
    seg2_mean = [f1_mean[x] for x in [1,5,9]]
    seg3_mean = [f1_mean[x] for x in [2,6,10]]
    seg4_mean = [f1_mean[x] for x in [3,7,11]]


    seg1_std = [f1_std[x] for x in [0,4,8]]
    seg2_std = [f1_std[x] for x in [1,5,9]]
    seg3_std = [f1_std[x] for x in [2,6,10]]
    seg4_std = [f1_std[x] for x in [3,7,11]]

    fig, ax = plt.subplots()
    barWidth = 0.20
    br1 = np.arange(len(seg1_mean))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    ax.bar(br1, seg1_mean, yerr=seg1_std, color ='r', width = barWidth, edgecolor ='grey', align='center', alpha=0.5, ecolor='black', capsize=2, label='H')
    ax.bar(br2, seg2_mean, yerr=seg2_std, color ='g', width = barWidth, edgecolor ='grey', align='center', alpha=0.5, ecolor='black', capsize=2, label='H+RS')
    ax.bar(br3, seg3_mean, yerr=seg3_std, color ='pink', width = barWidth, edgecolor ='grey', align='center', alpha=0.5, ecolor='black', capsize=2, label='H+2RS')
    ax.bar(br4, seg4_mean, yerr=seg4_std, color ='b', width = barWidth, edgecolor ='grey', align='center', alpha=0.5, ecolor='black', capsize=2, label='H+3RS')

    ax.set_ylabel('F1 Score (%)')
    ax.set_ylim(0, 100)
    ax.set_xticks([r + barWidth for r in range(len(seg1_mean))], ['SVC', 'NN', 'RF'])
    #ax.set_xticklabels(materials)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_dir+'.pdf')
    #plt.show()


if __name__=='__main__':
    best_mean_f1 = []
    for directory_in_str in ml_directories:
        best_mean_f1.append(machinelearning_run(directory_in_str))
    #print(best_mean_f1)

    nap_ent_f1_mean,nap_da_f1_mean,nap_entda_f1_mean = [],[],[]
    nap_ent_f1_std,nap_da_f1_std,nap_entda_f1_std = [],[],[]

    attack_ent_f1_mean,attack_da_f1_mean,attack_entda_f1_mean = [],[],[]
    attack_ent_f1_std,attack_da_f1_std,attack_entda_f1_std = [],[],[]

    for ml_model in best_mean_f1:
        for stat in ml_model:
            print(stat['model'])
            if stat['dataset'] == 'napierone':
                match(stat['statistic']):
                    case 'MEAN ENT':
                        #print(stat)
                        nap_ent_f1_mean.append(stat['f1_score'])
                    case 'STD ENT':
                        #print(stat)
                        nap_ent_f1_std.append(stat['f1_score'])
                    case 'MEAN DA':
                        #print(stat)
                        nap_da_f1_mean.append(stat['f1_score'])
                    case 'STD DA':
                        #print(stat)
                        nap_da_f1_std.append(stat['f1_score'])
                    case 'MEAN ENT+DA':
                        #print(stat)
                        nap_entda_f1_mean.append(stat['f1_score'])
                    case 'STD ENT+DA':
                        #print(stat)
                        nap_entda_f1_std.append(stat['f1_score'])

            else:
                 match(stat['statistic']):
                    case 'MEAN ENT':
                        #print(stat)
                        attack_ent_f1_mean.append(stat['f1_score'])
                    case 'STD ENT':
                        #print(stat)
                        attack_ent_f1_std.append(stat['f1_score'])
                    case 'MEAN DA':
                        #print(stat)
                        attack_da_f1_mean.append(stat['f1_score'])
                    case 'STD DA':
                        #print(stat)
                        attack_da_f1_std.append(stat['f1_score'])
                    case 'MEAN ENT+DA':
                        #print(stat)
                        attack_entda_f1_mean.append(stat['f1_score'])
                    case 'STD ENT+DA':
                        #print(stat)
                        attack_entda_f1_std.append(stat['f1_score'])
            seg = stat['segments']
            #print(seg)
    #print(nap_ent_f1_mean,nap_ent_f1_std)
    #print(nap_da_f1_mean,nap_da_f1_std)
    #print(nap_entda_f1_mean,nap_entda_f1_std)
    #print(attack_ent_f1_mean,attack_ent_f1_std)
    #print(attack_da_f1_mean,attack_da_f1_std)
    #print(attack_entda_f1_mean,attack_entda_f1_std)

    plot_best_f1_ml(nap_ent_f1_mean, nap_ent_f1_std, './napierone/nap_ent_f1')
    plot_best_f1_ml(nap_da_f1_mean,nap_da_f1_std, './napierone/nap_da_f1')
    plot_best_f1_ml(nap_entda_f1_mean,nap_entda_f1_std, './napierone/nap_ent+da_f1')
    plot_best_f1_ml(attack_ent_f1_mean,attack_ent_f1_std, './attackdaa/attack_ent_f1')
    plot_best_f1_ml(attack_da_f1_mean,attack_da_f1_std, './attackdaa/attack_da_f1')
    plot_best_f1_ml(attack_entda_f1_mean,attack_entda_f1_std, './attackdaa/attack_ent+da_f1')

    #best_nap_svc,best_nap_nn,best_nap_rf = 0,0,0
    #best_att_svc,best_att_nn,best_att_rf = 0,0,0

    #best_nap_svc_f1_score,best_nap_nn_f1_score,best_nap_rf_f1_score = 0,0,0
    #best_att_svc_f1_score,best_att_nn_f1_score,best_att_rf_f1_score = 0,0,0

    #for ml_model in best_mean_f1:
        #for stat in ml_model:
            #if 'MEAN' in stat['statistic']:
                #if stat['dataset'] == 'napierone':
                    #match(stat['model']):
                        #case 'svc':
                            #if (stat['f1_score'] > best_nap_svc_f1_score):
                                #best_nap_svc_f1_score = stat['f1_score']
                                #best_nap_svc = stat
                        #case 'nn':
                            #if stat['f1_score'] > best_nap_nn_f1_score:
                                #best_nap_nn_f1_score = stat['f1_score']
                                #best_nap_nn = stat
                        #case 'randomforest':
                            #if stat['f1_score'] > best_nap_rf_f1_score:
                                #best_nap_rf_f1_score = stat['f1_score']
                                #best_nap_rf = stat
                #else:
                    #match(stat['model']):
                        #case 'svc':
                            #if stat['f1_score'] > best_att_svc_f1_score:
                                #best_att_svc_f1_score = stat['f1_score']
                                #best_att_svc = stat
                        #case 'nn':
                            #if stat['f1_score'] > best_att_nn_f1_score:
                                #best_att_nn_f1_score = stat['f1_score']
                                #best_att_nn = stat
                        #case 'randomforest':
                            #if stat['f1_score'] > best_att_rf_f1_score:
                                #best_att_rf_f1_score = stat['f1_score']
                                #best_att_rf = stat

    #print(best_nap_svc,best_nap_nn,best_nap_rf)
    #print(best_att_svc,best_att_nn,best_att_rf)

    #for ml_model in best_mean_f1:
        #for stat in ml_model:
            #if 'STD' in stat['statistic']:
                #if stat['dataset'] == 'napierone':
                    #match(stat['model']):
                        #case 'svc':
                            #if (stat['segments length'] == best_nap_svc['segments length'] and stat['segments'] == best_nap_svc['segments']):
                                #print(stat)
                        #case 'nn':
                            #if (stat['segments length'] == best_nap_nn['segments length'] and stat['segments'] == best_nap_nn['segments']):
                                #print(stat)
                        #case 'randomforest':
                            #if (stat['segments length'] == best_nap_rf['segments length'] and stat['segments'] == best_nap_rf['segments']):
                                #print(stat)
                #else:
                    #match(stat['model']):
                        #case 'svc':
                            #if (stat['segments length'] == best_att_svc['segments length'] and stat['segments'] == best_att_svc['segments']):
                                #print(stat)
                        #case 'nn':
                            #if (stat['segments length'] == best_att_nn['segments length'] and stat['segments'] == best_att_nn['segments']):
                                #print(stat)
                        #case 'randomforest':
                            #if (stat['segments length'] == best_att_rf['segments length'] and stat['segments'] == best_att_rf['segments']):
                                #print(stat)

    #divide_df(csv_file_path)
    #for directory_in_str in daa_directories:
        #daa_run(directory_in_str)
    #for directory_in_str in daa_directories:
        #daa_max_f1(directory_in_str)


    #csv_file_path = '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf/segmented_input/statistics/old_daa/history/20240624-113932_ALL_[1]F_256_daa_old_train_tot_results.csv'
    #csv_file_path = '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf/segmented_input/statistics/daa/20240612-154706_ALL_[1]F_256_daa_train_tot_results.csv'
    #csv_file_path = '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf_low_entropy_ransom/segmented_input/statistics/daa/20240612-154934_ALL_[1]F_256_daa_attacks_train_tot_results.csv'
    #df = pd.read_csv(csv_file_path)
    #max_f1_header_oldnap = df.loc[df['F1_Score'] == df['F1_Score'].max()]
    #print(max_f1_header_oldnap.to_string())
    #csv_pdf = './napierone/old_daa/napierone_daa_old_train_tot_header.pdf'
    #csv_pdf = './attackdaa/daa/attackdaa_daa_train_tot_header.pdf'
    #plot_daa_header_metrics(df, csv_pdf)







# Load DataFrame from CSV file
#csv_file_path = '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf/segmented_input/statistics/daa/20240611-125212_ALL_[1, 2, 3, 4]F_256_daa_train_tot_results.csv'

#csv_file_path = '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf/segmented_input/statistics/daa/20240612-154706_ALL_[0]F_256_daa_train_tot_results.csv'
#csv_file_path = '/home/emanuele/Documents/Tesi/Code/Machine_Learning/algorithms/tiny/no_pdf_low_entropy_ransom/segmented_input/statistics/daa/20240612-154934_ALL_[0]F_256_daa_attacks_train_tot_results.csv'

#df = pd.read_csv(csv_file_path)

#for index, row in df.iterrows():
    ##print(row['F1_Score'])#df.loc[5, 'Name'] = 'SHIV CHANDRA'
    #if (row['Precision']+row['Recall'] != 0):
        #df.loc[index, 'F1_Score'] = 2*row['Precision']*row['Recall']/(row['Precision']+row['Recall'])
        ##row['F1_Score'] = 2*row['Precision']*row['Recall']/(row['Precision']+row['Recall'])
    #else:
        #df.loc[index, 'F1_Score'] = 0.

#df.loc[:, 'Segments'] = df['Segments'] + 1

#for index, row in df.iterrows():
    #if (row['Segments'] == 1):
#print((df.loc[4128:])['Distance','Threshold','Segments Length'])
#print(df[(df['Segments'] == 1) & (df['Distance'] == 0)])
#for index, row in df.iterrows():
    #df.loc[index, 'Segments'] = row['Segments'] + 1



#csv_pdf = f'./standard/nn/test/4_nn_test_statistics_{value_to_find}F.pdf'
#plot_performance_metrics(df, 1, csv_pdf)


# Iterate through unique values in 'Segments' column
#for value_to_find in df['Segments Length'].unique():
    #Create a filename for the PDF
    #csv_pdf = f'./standard/nn/test/4_nn_test_statistics_{value_to_find}F.pdf'
#csv_pdf = f'./attackdaa/daa/attackdaa_daa_test_header.pdf'
    #Call the plotting function
#plot_performance_metrics(df, csv_pdf)

