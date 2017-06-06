import math
import numpy as np
import random
import itertools


class Simple_Seq:
    def next_batch(self, batch_size=1, seq_length=10, encoding_length=2):
        X, Y = [], []
        for b in range(batch_size):
            if b%100 == 0:
                print("preparing dataset... %s/%s" % (b, batch_size))
            random_nums = [random.randint(0,encoding_length-1) for i in range(seq_length)]
            seq_of_inputs = np.eye(encoding_length)[random_nums]
            target = self.calc_target(random_nums)
            X.append(seq_of_inputs), Y.append(target)

        return np.asarray(X), np.asarray(Y)

    def calc_target(self, random_nums):
        to_find = 0
        if to_find in random_nums:
            return random_nums.index(to_find)
        else:
            return len(random_nums)-1

if __name__ == "__main__":
    p = Simple_Seq()
    sample_batch = p.next_batch(1)
    print(sample_batch)
