import sys
import time
from collections import Counter
import multiprocessing
from functools import partial
import math

import pandas as pd

from log import get_logger
from origami_greedy import Origami
from origamiprepostprocess import OrigamiPrePostProcess


class ProcessFile(Origami):
    """
    Decoding and encoding will call this class. And this class will call
    the origami method to handle individual origami. This file will also call
    """

    def __init__(self, redundancy, verbose, degree="new"):
        """
        This will combine all the origami and reconstruct the file
        :param redundancy: This is minimum number of redundancy
        :param verbose:
        :param degree: is it old or new degree distribution
        """
        super().__init__(verbose=verbose)
        self.minimum_redundancy = redundancy
        self.degree = degree
        self.verbose = verbose
        self.logger = get_logger(verbose, __name__)
        # Will be updated later during checking number how much redundancy we will need
        self.number_of_bit_per_origami = 16

    def encode(self, file_in, file_out, formatted_output=False):
        """
        Encode the file
        :param file_in: File that need to be encoded
        :param file_out: File where output will be saved
        :param formatted_output: Output will written as a matrix
        :return:
        """
        try:
            file_in = open(file_in, 'rb')
            file_out = open(file_out, "w")
        except Exception as e:
            self.logger.exception(e)
            self.logger.error("Error opening the file")
            return -1, -1, -1, -1  # simulation file expect this format
        data = file_in.read()
        file_in.close()
        # Converting data into binary
        data_in_binary = ''.join(format(letter, '08b') for letter in data)
        preprocess = OrigamiPrePostProcess(self.verbose)
        segments, xored_data_list, data_bit_per_origami, required_red = \
            preprocess.encode(data_in_binary, min_required_redundancy=self.minimum_redundancy, degree=self.degree)
        # write the encoded file in the file
        for origami_index, single_origami in enumerate(xored_data_list):
            encoded_stream = self._encode(single_origami, origami_index, data_bit_per_origami)
            if formatted_output:
                print("Matrix -> " + str(origami_index), file=file_out)
                self.print_matrix(self.data_stream_to_matrix(encoded_stream), in_file=file_out)
            else:
                file_out.write(encoded_stream + '\n')
        file_out.close()
        print("Encoded done")
        return segments, len(xored_data_list), data_bit_per_origami, required_red

    def single_origami_decode(self, origami_row, ior_file_name, correct_dictionary, common_parity_index,
                              minimum_temporary_weight, maximum_number_of_error, false_positive):
        origami_id = origami_row[0]
        origami_row = origami_row[1]
        origami = origami_row['Binary String']
        origami_intensity = origami_row['intensity string']
        intensity_threshold = origami_row['threshold']
        fit_quality = origami_row['Fit Quality']
        ior_text = None
        # print(origami_id)
        current_time = time.time()
        self.logger.info("Working on origami(%d): %s", origami_id, origami)
        if len(origami) != 48:
            self.logger.warning("Data point is missing in the origami")
            return [None, None, None]
        try:
            decoded_matrix = super().decode(origami, origami_intensity, intensity_threshold, common_parity_index,
                                            minimum_temporary_weight, maximum_number_of_error, false_positive)
        except Exception as e:
            self.logger.exception(e)
            return [None, None, None]

        if decoded_matrix == -1:
            return [None, None, None]

        self.logger.info("Recovered a origami with index: %s and data: %s", decoded_matrix['index'],
                         decoded_matrix['binary_data'])

        if decoded_matrix['total_probable_error'] > 0:
            self.logger.info("Total %d errors found in locations: %s", decoded_matrix['total_probable_error'],
                             str(decoded_matrix['probable_error_locations']))
        else:
            self.logger.info("No error found")
        # Storing information in individual origami report

        if ior_file_name:
            # Checking correct value
            if correct_dictionary:
                try:
                    status = int(correct_dictionary[int(decoded_matrix['index'])] == decoded_matrix['binary_data'])
                except Exception as e:
                    self.logger.warning(str(e))
                    status = -1
            else:
                status = " "
            decoded_time = round(time.time() - current_time, 3)
            ior_text = f"{origami_id},{origami},{status},{decoded_matrix['total_probable_error']}," \
                       f"{str(decoded_matrix['probable_error_locations']).replace(',', ' ')}," \
                       f"{decoded_matrix['orientation']},{decoded_matrix['index']}," \
                       f"{self.matrix_to_data_stream(decoded_matrix['matrix'])},{decoded_matrix['binary_data']}," \
                       f"{decoded_time},{fit_quality}\n"
        return [decoded_matrix, status, ior_text]

    def decode(self, file_in, file_out, file_size, threshold_data, threshold_parity, maximum_number_of_error,
               individual_origami_info, false_positive, correct_file=False):
        correct_origami = 0
        incorrect_origami = 0
        total_error_fixed = 0
        # Read the file
        try:
            data_df = pd.read_csv(file_in)
            # data_file = open(file_in, "r")
            # data = data_file.readlines()
            # data_file.close()
            # File to store individual origami information
            if individual_origami_info:
                ior_file_name = file_out + "_ior.csv"
                with open(ior_file_name, "w") as ior_file:
                    ior_file.write(
                        "Line number in file, origami,status,error,error location,orientation,decoded index,"
                        "decoded origami, decoded data,decoding time,fit quality\n")
            else:
                ior_file_name = False
        except Exception as e:
            self.logger.error("%s", e)
            return
        preprocess = OrigamiPrePostProcess(self.verbose)
        segments, total_origami_with_red, redundancy, self.number_of_bit_per_origami, xored_map = \
            preprocess.recoverable_red(file_size * 8, self.minimum_redundancy, self.degree)
        self.matrix_details, self.parity_bit_relation, self.checksum_bit_relation = \
            self._matrix_details(self.number_of_bit_per_origami)
        self.data_bit_to_parity_bit = self.data_bit_to_parity_bit(self.parity_bit_relation)

        decoded_dictionary = {}
        # If user pass correct file we will create a correct key value pair from that and will compare with our decoded
        # data.
        correct_dictionary = {}
        if correct_file:
            with open(correct_file) as cf:
                for so in cf:
                    ci, cd = self._extract_text_and_index(self.data_stream_to_matrix(so.rstrip("\n")))
                    correct_dictionary[ci] = cd
        # Decoded dictionary with number of occurrence of a single origami
        decoded_dictionary_wno = {}
        # origami_data = [(i, single_origami.rstrip("\n")) for i, single_origami in enumerate(data)]
        p_single_origami_decode = partial(self.single_origami_decode, ior_file_name=ior_file_name,
                                          correct_dictionary=correct_dictionary,
                                          common_parity_index=threshold_data,
                                          minimum_temporary_weight=threshold_parity,
                                          maximum_number_of_error=maximum_number_of_error,
                                          false_positive=false_positive)
        optimum_number_of_process = int(math.ceil(multiprocessing.cpu_count()))
        pool = multiprocessing.Pool(processes=optimum_number_of_process)
        return_value = pool.map(p_single_origami_decode, data_df.iterrows())
        # return_value = map(p_single_origami_decode, data_df.iterrows())
        pool.close()
        pool.join()
        if ior_file_name:
            ior_file = open(ior_file_name, "a")
        for decoded_matrix, status, ior_text in return_value:
            if ior_text is not None:
                ior_file.write(ior_text)
            if not decoded_matrix is None:
                # Checking status
                if correct_file:
                    if status:
                        correct_origami += 1
                    else:
                        incorrect_origami += 1
                total_error_fixed += int(decoded_matrix['total_probable_error'])
                decoded_dictionary_wno.setdefault(decoded_matrix['index'], []).append(
                    decoded_matrix['binary_data'])
        if ior_file_name:
            ior_file.close()
        majority_vote_queue = {}
        recovered_origami = set()
        origami_list = set(list(range(total_origami_with_red)))
        for index, single_index_origami in decoded_dictionary_wno.items():
            if index >= total_origami_with_red:
                # This origami doesn't exists. It's a garbage
                continue
            most_common = Counter(single_index_origami).most_common()
            # Majority voting
            # If a origami has same number of majority voting but different data then we will discard that
            if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
                continue
            # If a origami has single majority voting we will consider that as recovered
            recovered_origami.add(index)
            # Initially, we will start the fountain code decoding with origami having majority voting more than 2.
            # Which ever origami have majority voting less 3, we will add them in the queue
            # We we can not decode the file with majority voting 3 then we will keep using other origamies with lower
            # majority voting
            if most_common[0][1] >= 3:
                decoded_dictionary[index] = most_common[0][0]
            else:
                if most_common[0][1] not in majority_vote_queue:
                    # queue will have a key value pair of majoirty voting
                    # { 'majority_vote' : 'origami_data' }
                    majority_vote_queue[most_common[0][1]] = {}
                majority_vote_queue[most_common[0][1]][index] = most_common[0][0]
        # We will calculate the number of missing origami here
        missing_origami = list(origami_list.difference(recovered_origami))
        # Sorting the dictionary according to the index
        decoded_elements = {}
        current_data_map = {}
        try:
            # Initialize the fountain code decoding using the origami having majority vote of >= 3
            decoded_elements, current_data_map = preprocess.decode(decoded_dictionary, segments, xored_map,
                                                                   decoded_elements, current_data_map)
            # we will keep adding element from the queue until the queue is empty or we recover the file
            while not len(decoded_elements) == segments:
                if len(majority_vote_queue) == 0:
                    # If no more item in the queue then the file is not decodable
                    self.logger.warning("Could not decode the file")
                    self.logger.info("Number of missing origami: " + str(missing_origami))
                    return -1, incorrect_origami, correct_origami, total_error_fixed, missing_origami
                else:
                    # get the first element from the majority vote queue
                    # Add it with the current data map
                    decoded_dictionary = majority_vote_queue[max(majority_vote_queue.keys())]
                    del majority_vote_queue[max(majority_vote_queue.keys())]
                decoded_elements, current_data_map = preprocess.decode(decoded_dictionary, segments, xored_map,
                                                                       decoded_elements, current_data_map)
        except Exception as e:
            self.logger.warning("Couldn't decode the file")
            self.logger.info("Number of missing origami: " + str(missing_origami))
            self.logger.exception(e)
            return -1, incorrect_origami, correct_origami, total_error_fixed, missing_origami
        # Combine all the file segments
        recovered_binary = "".join(str(decoded_elements[index]) for index in sorted(decoded_elements.keys()))
        # remove partial padding
        recovered_binary = recovered_binary[:8 * (len(recovered_binary) // 8)]
        with open(file_out, "wb") as result_file:
            for start_index in range(0, len(recovered_binary), 8):
                bin_data = recovered_binary[start_index:start_index + 8]
                # convert bin data into decimal
                decimal = int(''.join(str(i) for i in bin_data), 2)
                if decimal == 0 and start_index + 8 == len(
                        recovered_binary):  # This will remove the padding. If the padding is whole byte.
                    continue
                decimal_byte = bytes([decimal])
                result_file.write(decimal_byte)
        self.logger.info("Number of missing origami :" + str(missing_origami))
        self.logger.info("Total error fixed: " + str(total_error_fixed))
        print("File recovery was successfull")
        return 1, incorrect_origami, correct_origami, total_error_fixed, missing_origami


# This is for debugging purpose
if __name__ == '__main__':
    pass
