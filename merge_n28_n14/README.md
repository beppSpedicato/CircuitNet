# Merge N14 and N28 for DRC violation

merge the N28 and N14 datasets for DRC violation task

1. Setup (config variables)
- n28_csv_file_test n28_csv_file_train
- n14_csv_file_test n28_csv_file_train
- n28_training_set folder 
- n14_training_set folder
- output_folder


2. Read csv files and check for duplicates
each line of the CSV file has in each row has two filepath: the first path is about the features and the second about the label

3. in the output folder creates two folders: /features, /labels

4. take separated the two csv files (dataframes)
for each not duplicated element merge one dataframe for each part

merge the files from n28 and n14 folder to outputfolder


