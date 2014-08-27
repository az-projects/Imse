from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import Intelligence.GP_GPU.gp_cuda as gp_cuda

import numpy as np
import sys


def test_gpu_gp():
    for i in range(10):
        feedback = np.load(str(i) + '_feedback.npy')
        feedback_indices = np.load(str(i) + '_feedback_indices.npy')
        K_diag_noise = np.load(str(i) + '_random_K.npy')
        K_xx_noise = np.load(str(i) + '_random_K_xx.npy')

        mean, variance = gp_cuda.gaussian_process(data, feedback, feedback_indices, K_noise=K_diag_noise, K_xx_noise=K_xx_noise)


class Command(Basecommand):
    def handle(self, *args, **options):
        data = np.asfarray(np.load(settings.DATA_PATH + "cl25000.npy"), dtype="float32")
        if len(sys.argv) > 1:
            print('sys.argv length:', len(sys.argv))
            if sys.argv[1] == 'debug':
                print('sys.argv[1] == debug')
                for i in range(20):
                    feedback = np.load(str(i) + '_feedback.npy')
                    feedback_indices = np.load(str(i) + '_feedback_indices.npy')
                    K_diag_noise = np.load(str(i) + '_random_K.npy')
                    K_xx_noise = np.load(str(i) + '_random_K_xx.npy')
                    mean, variance = gp_cuda.gaussian_process(data, feedback, feedback_indices, debug=True)
                exit()

            if sys.argv[1] == 'test':
                #print('sys.argv[1] == test')
                test gpu_gp()
                exit()
