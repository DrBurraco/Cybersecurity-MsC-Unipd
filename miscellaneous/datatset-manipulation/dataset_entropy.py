import numpy as np
import zipfile
import os
import glob
import matplotlib.pyplot as plt
from collections import Counter

def entropy(byte_array):
    # Convert byte array to numpy array
    data = np.array(byte_array)
    # Calculate the probabilities of each byte value
    probabilities = np.bincount(data) / len(data)
    # Filter out zero probabilities to avoid log2(0)
    probabilities = probabilities[probabilities > 0]
    # Compute the entropy
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def process_zip_files(folder_path):
    # List all zip files in the folder
    zip_files = glob.glob(os.path.join(folder_path, '*.zip'))

    # Dictionary to hold entropy arrays for each category
    entropy_dict = {}
    small_file = {}
    for zip_file in zip_files:
        # Extract category name from zip file name
        category_name = os.path.basename(zip_file).replace('-tiny-NO-PDF.zip', '')

        #print(category_name)
        with zipfile.ZipFile(zip_file, 'r') as z:
            for file_name in z.namelist():
                #print(file_name)
                with z.open(file_name) as f:
                    # Read the first 256 bytes
                    #byte_content = list(f.read(256))
                    byte_content = list(f.read())
                    # Ensure the byte content is at least 256 bytes
                    if len(byte_content) < 256 * 4:
                        if category_name not in small_file:
                            small_file[category_name] = 0
                        small_file[category_name] += 1
                        #continue

                    # Divide the bytes into chunks of 8
                    #byte_chunks = [byte_content[:i+8] for i in range(0, 256, 8)]

                    # Compute entropy for each chunk and store in the category's list
                    #entropies = [entropy(chunk) for chunk in byte_chunks]

                    #if category_name not in entropy_dict:
                        #entropy_dict[category_name] = []
                    #entropy_dict[category_name].append(entropies)

    # Calculate the mean entropy for each category
    #mean_entropy_dict = {category: np.mean(entropies, axis=0) for category, entropies in entropy_dict.items()}

    #return mean_entropy_dict
    return small_file
    return

def plot_mean_entropies(mean_entropies):
    segment_lengths = list(range(8, 257, 8))

    plt.figure(figsize=(16, 10))

    print_y = []

    for category, mean_entropy in mean_entropies.items():
        if category.startswith('RANSOMWARE-'):
            plt.plot(segment_lengths, mean_entropy, label=category.replace('RANSOMWARE-', '', 1))
            plt.text(segment_lengths[-1], mean_entropy[-1], category.replace('RANSOMWARE-', '', 1), verticalalignment='center', fontsize=8)
        #if not category.startswith('RANSOMWARE-'):
            #plt.plot(segment_lengths, mean_entropy, label=category)
            #element_to_check = round(mean_entropy[-1], 2)
            #if (element_to_check not in print_y):
                #print_y.append(element_to_check)
            #else:
                #count_dict = Counter(print_y)
                #count_of_element = count_dict.get(element_to_check, 0)
                #y = element_to_check - count_of_element
                #print_y.append(y)
            #plt.text(segment_lengths[-1], category.replace('RANSOMWARE-', '', 1), verticalalignment='bottom', fontsize=8)
        #line, = plt.plot(segment_lengths, mean_entropy, label=category)
        #if category.startswith('RANSOMWARE'):
            #ransomware_lines.append(line)
        #else:
            #other_lines.append(line)

    plt.xlabel('Segment Length')
    plt.ylabel('Entropy')
    plt.ylim(-1, 8)
    plt.xlim(0, 264)
    plt.yticks(list(range(10)))
    plt.xticks(segment_lengths)
    plt.title('Entropy vs. Segment Length for Different Categories')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=6,prop={'size': 8})
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Define the folder path
folder_path = '/home/emanuele/Documents/Tesi/Code/Machine_Learning/my_datasets/downloads/manual/Tiny/NapierOne-tiny-NO-PDF'

# Process the zip files and compute the mean entropy for each category
#mean_entropies = process_zip_files(folder_path)
small_file = process_zip_files(folder_path)
print(small_file)

# Plot the mean entropies
#plot_mean_entropies(mean_entropies)
