#!/usr/bin/env python3

import zipfile as zf
import io
from os import listdir
from os.path import isfile, join
import numpy as np
import math
import re

LENGTH = 10000
def main():
    exit = ''
    mypath = '/home/emanuele/Documents/Tesi/Code/Machine_Learning/my_datasets/downloads/manual/Tiny/NapierOne-tiny-NO-PDF/'
    ransom_length,legit_length = [],[]
    total_ransom_count,small_ransom,equal_ransom,big_ransom, total_legit_count = 0,0,0,0,0
    min_lengths = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    ransom_cat, legit_cat = [], []
    for file in onlyfiles:
        if file.startswith('RANSOMWARE'):
            with zf.ZipFile(mypath + '/' + file, mode='r') as archive: #open NapierOne archive in read mode
                file_list = archive.namelist() #get names of files
                ransom_count = 0
                for file_name in file_list: #iterate through names
                    with archive.open(file_name) as read_file: #open specific ransom archive (ex. Netwalker)
                        file_data = np.frombuffer(read_file.read(), dtype=np.ubyte)
                        #print('File data standard: ',file_data)
                        #file_data_ubyte = np.frombuffer(file_data, dtype=np.ubyte) #read bytes of archive
                        #print('File data np.ubyte: ',file_data_ubyte)
                        #file_data_byte = np.frombuffer(file_data, dtype=np.byte) #read bytes of archive
                        #print('File data np.byte: ',file_data_byte)
                        #file_data_int8 = np.frombuffer(file_data, dtype=np.int8) #read bytes of archive
                        #print('File data np.int8: ',file_data_int8)
                        #daa =[]
                        #for byte in file_data:
                            #daa.append(int.from_bytes(file_data, byteorder="big"))
                        #print('File data in DAA: ',daa)
                        #if (file_name.endswith('.cmb')):
                            #if(np.sum(file_data[:262145]) <= 0):
                                #min_lengths.append(len(file_data))
                                #print(len(file_data))
                                #print(file_data,np.sum(file_data[:262145]),np.bincount(file_data[:262145]))
                        ransom_count += 1
                        ransom_length.append(len(file_data))
                        read_file.close()
                ransom_cat.append(re.search('-(.*?)-', file).group(1))
                if(ransom_count != 100):
                    print(file+f': {ransom_count} files <-- ATTENTION!! {ransom_count} FILES!!')
                else:
                    print(file+f': {ransom_count} files')
                archive.close()
            total_ransom_count += ransom_count
        else:
            with zf.ZipFile(mypath + '/' + file, mode='r') as archive: #open NapierOne archive in read mode
                file_list = archive.namelist() #get names of files
                legit_count = 0
                for file_name in file_list: #iterate through names
                    with archive.open(file_name) as read_file: #open specific ransom archive (ex. Netwalker)
                        file_data = np.frombuffer(read_file.read(), dtype=np.ubyte)
                        legit_length.append(len(file_data))
                        read_file.close()
                        legit_count += 1
                legit_cat.append(re.search('(.*?)-', file).group(1))
                if(legit_count != 100):
                    print(file+f': {legit_count} files <-- ATTENTION!! {legit_count} FILES!!')
                else:
                    print(file+f': {legit_count} files')
                archive.close()
            total_legit_count += legit_count

    print(legit_cat, ransom_cat)
    print(sorted(legit_cat), sorted(ransom_cat))
    print(sorted(legit_cat) + sorted(ransom_cat))
    print(sorted(list(set(legit_cat))) + sorted(list(set(ransom_cat))))
    #print('Min 0 length: ', np.min(min_lengths))
    for length in ransom_length:
        if(length < LENGTH):
            small_ransom += 1
        elif(length == LENGTH):
            equal_ransom +=1
        else:
            big_ransom += 1

    total_files = total_ransom_count+total_legit_count
    print('Total files: ', total_files)
    print(f'Total legit files: {total_legit_count}, {(total_legit_count/total_files)*100:.2f}% of total')
    print(f'Total ransom files: {total_ransom_count}, {(total_ransom_count/total_files)*100:.2f}% of total')
    print('Min length: ', min(ransom_length))
    print('Max length: ', max(ransom_length))
    print('Mean length: ', np.mean(ransom_length))
    print('Num of fragments: ', math.ceil(np.sum(ransom_length)/LENGTH))
    print(f'Ransom meno di {LENGTH}: {small_ransom}')
    print(f'Ransom piu di {LENGTH}: {big_ransom}')
    print(f'Ransom uguali a {LENGTH}: {equal_ransom}')
    print('Min legit length: ', min(legit_length))
    print('Max legit length: ', max(legit_length))
    print('Mean legit length: ', np.mean(legit_length))
    print('Num of legit fragments: ', math.ceil(np.sum(legit_length)/LENGTH))
    return

if __name__ == '__main__':
    main()
