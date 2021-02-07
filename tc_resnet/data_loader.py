from os import listdir
from os.path import join, isfile, isdir, normpath
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
import numpy as np
from itertools import zip_longest

from data_preprocessor import DataPreprocessor
from tc_resnet import PROJECT_PATH


class DataLoader:

    def __init__(self, classes, path, sample_size):
        """
        Given the path and sample size DataLoader loads the data and applies preprocessing steps described in DataPreprocessor class

        :param classes: list of classes to load from the dataset
        :type path: list
        :param path: dataset path
        :type path: str
        :param sample_size: number of files per class used for training
        :type sample_size: int
        """
        self.classes = classes
        self.dataset_path = PROJECT_PATH / path
        self.sample_size = sample_size

    def __load_audio_filenames_with_class__(self):
        filenames = []
        class_ids = []
        for i in range(len(self.classes)):
            c = self.classes[i]
            class_filenames = self.__load_audio_filenames__(join(self.dataset_path, c))
            filenames.extend(class_filenames)
            class_ids.extend([i] * len(class_filenames))
        return filenames, class_ids, self.classes

    def __load_audio_filenames__(self, class_folder):
        filenames = []
        folder_content = listdir(class_folder)
        amount = self.sample_size if self.sample_size is not None else len(folder_content)
        for entry in listdir(class_folder)[:amount]:
            full_path = join(class_folder, entry)
            if isfile(full_path):
                if entry.endswith('.wav'):
                    filenames.append(full_path)
            else:
                filenames.extend(self.__load_audio_filenames__(full_path))
        return filenames

    @staticmethod
    def __load_subset_filenames__(root_folder, filename):
        subset_list = []
        with open(join(root_folder, filename)) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                subset_list.append(normpath(join(root_folder, line)))
        return set(subset_list)

    def load_data_from_folder(self):
        filenames, class_ids, classes = self.__load_audio_filenames_with_class__()
        dataset_size = len(filenames)
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        X_validation = []
        y_validation = []
        pool = Pool(cpu_count() - 1)
        preprocessor = DataPreprocessor(dataset_path=self.dataset_path)
        for (results, filepath, class_id, random_roll) in tqdm(pool.imap_unordered(preprocessor.process_file, zip_longest(filenames, class_ids)),
                                                               total=dataset_size):
            filepath = normpath(filepath)
            is_testing = 1 <= random_roll <= 10
            is_validation = 11 <= random_roll <= 20
            for item in results:
                if is_testing:
                    X_test.append(item)
                    y_test.append(class_id)
                elif is_validation:
                    X_validation.append(item)
                    y_validation.append(class_id)
                else:
                    X_train.append(item)
                    y_train.append(class_id)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_validation = np.array(X_validation)
        y_validation = np.array(y_validation)
        return X_train, y_train, X_test, y_test, X_validation, y_validation, classes
