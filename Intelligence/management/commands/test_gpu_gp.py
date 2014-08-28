from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import Intelligence.GP_GPU.gp_cuda as gp_cuda

import numpy as np
import sys
import time


def test_gpu_gp(data, test_input_path):
    totaltime = 0
    for i in range(63):
        inputprefix = test_input_path + str(i)
        feedback = np.load(str(inputprefix) + '_feedback.npy')
        feedback_indices = np.load(str(inputprefix) + '_feedback_indices.npy')
        K_diag_noise = np.load(str(inputprefix) + '_random_K.npy')
        K_xx_noise = np.load(str(inputprefix) + '_random_K_xx.npy')

        t = time.time()
        mean, variance = gp_cuda.gaussian_process(data, feedback, feedback_indices, K_noise=K_diag_noise, K_xx_noise=K_xx_noise)
        itertime = time.time() - t
        print(str(i) + '\t' + str(itertime) + '\t' + str(len(feedback)))
        totaltime += itertime
    print('Total time:', str(totaltime))


class Command(BaseCommand):
    def handle(self, *args, **options):
        data = np.asfarray(np.load(settings.DATA_PATH + "cl25000.npy"), dtype="float32")
        input_path = settings.DATA_PATH + 'speedtest_input/'
        #print('sys.argv[1] == test')
        test_gpu_gp(data, input_path)
        exit()
