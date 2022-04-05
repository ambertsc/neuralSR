#!/usr/bin/env python
# coding: utf-8

# set up logging
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# load libraries
import os
import sys
import glob
import json
import math
import random
import numpy as np
# from tqdm import tqdm
from numpy import *  # to override the math functions

import torch
import torch.nn as nn
from torch.nn import functional as F
# from torch.utils.data import Dataset

from utils import set_seed, sample
from trainer_loss_xx_multi import Trainer, TrainerConfig
from model_nonorm_padd import GPT, GPTConfig, PointNetConfig
from scipy.optimize import minimize, least_squares
from utils import processDataFiles, CharDataset, relativeErr, mse, sqrt, divide, lossFunc

unique_seed = sys.argv[1]
# set the random seed and ID
uniqueID = "1011"+str(unique_seed)
set_seed(int(uniqueID))

# config
device = 'gpu'
numEpochs = 50  # number of epochs to train the GPT+PT model
embeddingSize = 512  # the hidden dimension of the representation of both GPT and PT
numPoints = 30  # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars = 1  # the dimenstion of input points x, if you don't know then use the maximum
numYs = 1  # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 200  # spatial extent of the model for its context
batchSize = 64  # batch size of training data
dataDir = './datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton'  # 'Skeleton' #'EQ'
dataFolder = 'nox2_x10'
addr = './SavedModels/'  # where to save model
method = 'EMB_SUM'  # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation.
variableEmbedding = 'NOT_VAR'  # NOT_VAR/LEA_EMB/STR_VAR
# NOT_VAR: Do nothing, will not pass any information from the number of variables in the equation to the GPT
# LEA_EMB: Learnable embedding for the variables, added to the pointNET embedding
# STR_VAR: Add the number of variables to the first token
addVars = True if variableEmbedding == 'STR_VAR' else False
maxNumFiles = 30  # maximum number of file to load in memory for training the neural network
bestLoss = None  # if there is any model to load as pre-trained one
fName = '{}_SymbolicGPT_50epochs_padding.txt'.format(uniqueID)
ckptPath = '{}/{}.pt'.format(addr, fName.split('.txt')[0])
try:
    os.mkdir(addr)
except:
    print('Folder already exists!')

# load the train dataset
path = '{}/{}/Train/*.json'.format(dataDir, dataFolder)
files = glob.glob(path)[:maxNumFiles]
text = processDataFiles(files)
chars = sorted(list(set(text)) + ['_', 'T', '<', '>',
                                  ':'])  # extract unique characters from the text before converting the text to a list, # T is for the test data
text = text.split('\n')  # convert the raw text to a set of examples
text = text[:-1] if len(text[-1]) == 0 else text
random.shuffle(text)  # shuffle the dataset, it's important specailly for the combined number of variables experiment
train_dataset = CharDataset(text, blockSize, chars, numVars=numVars,
                            numYs=numYs, numPoints=numPoints, target=target, addVars=addVars)

# print a random sample
idx = np.random.randint(train_dataset.__len__())
inputs, outputs, points, variables = train_dataset.__getitem__(idx)
print('inputs:{}'.format(inputs))
inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx, inputs, outputs, points, variables))

# load the val dataset
path = '{}/{}/Val/*.json'.format(dataDir, dataFolder)
files = glob.glob(path)
print("val files: ", files)
textVal = processDataFiles(files)
textVal = textVal.split('\n')  # convert the raw text to a set of examples
val_dataset = CharDataset(textVal, blockSize, chars, numVars=numVars,
                          numYs=numYs, numPoints=numPoints, target=target, addVars=addVars)

# print a random sample
print("val: ", textVal[0])
idx = np.random.randint(val_dataset.__len__())
print(val_dataset.__len__())
print("idx is", idx)
inputs, outputs, points, variables = val_dataset.__getitem__(idx)
print(points.min(), points.max())
inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx, inputs, outputs, points, variables))

# load the test data
path = '{}/{}/Test/*.json'.format(dataDir, dataFolder)
files = glob.glob(path)
textTest = processDataFiles(files)
textTest = textTest.split('\n')  # convert the raw text to a set of examples
# test_dataset_target = CharDataset(textTest, blockSize, chars, target=target)
test_dataset = CharDataset(textTest, 2 * blockSize, chars, numVars=numVars,
                           numYs=numYs, numPoints=numPoints, addVars=addVars)

# print a random sample
idx = np.random.randint(test_dataset.__len__())
inputs, outputs, points, variables = test_dataset.__getitem__(idx)
print(points.min(), points.max())
inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx, inputs, outputs, points, variables))

# create the model
pconf = PointNetConfig(embeddingSize=embeddingSize,
                       numberofPoints=numPoints,
                       numberofVars=numVars,
                       numberofYs=numYs,
                       method=method,
                       variableEmbedding=variableEmbedding)
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=embeddingSize,
                  padding_idx=train_dataset.paddingID)
model = GPT(mconf, pconf)

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=numEpochs, batch_size=batchSize,
                      learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * blockSize,
                      num_workers=0, ckpt_path=ckptPath)
trainer = Trainer(model, train_dataset, val_dataset, tconf, bestLoss, device=device, uniqueID=uniqueID)

try:
    trainer.train()
except KeyboardInterrupt:
    print('KeyboardInterrupt')

# load the best model
print('The following model {} has been loaded!'.format(ckptPath))
model.load_state_dict(torch.load(ckptPath))
model = model.eval().to(trainer.device)

## Test the model
# alright, let's sample some character-level symbolic GPT
loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    pin_memory=True,
    batch_size=1,
    num_workers=0)

from utils import *

resultDict = {}
try:
    with open(fName, 'w', encoding="utf-8") as o:
        resultDict[fName] = {'SymbolicGPT': []}
        difference_list = []
        # print(len(loader.dataset))
        binary_matrix_target = np.zeros((1000, 11))
        binary_matrix_predicted = np.zeros((1000, 11))
        num_invalid = 0
        num_valid = 0

        for i, batch in enumerate(loader):

            inputs, outputs, points, variables = batch

            print('Test Case {}.'.format(i))
            o.write('Test Case {}/{}.\n'.format(i, len(textTest) - 1))

            t = json.loads(textTest[i])

            inputs = inputs[:, 0:1].to(trainer.device)
            points = points.to(trainer.device)
            variables = variables.to(trainer.device)
            outputsHat = sample(model,
                                inputs,
                                blockSize,
                                points=points,
                                variables=variables,
                                temperature=1.0,
                                sample=True,
                                top_k=40)[0]

            # filter out predicted
            target = ''.join([train_dataset.itos[int(i)] for i in outputs[0]])
            predicted = ''.join([train_dataset.itos[int(i)] for i in outputsHat])

            if variableEmbedding == 'STR_VAR':
                target = target.split(':')[-1]
                predicted = predicted.split(':')[-1]

            target = target.strip(train_dataset.paddingToken).split('>')
            target = target[0]  # if len(target[0])>=1 else target[1]
            target = target.strip('<').strip(">")
            predicted = predicted.strip(train_dataset.paddingToken).split('>')
            predicted = predicted[0]  # if len(predicted[0])>=1 else predicted[1]
            predicted = predicted.strip('<').strip(">")

            target_split = target.split("+")
            print("target split \n", target_split)

            if "1.0" in target_split:
                binary_matrix_target[i, 0] = 1
            if "1.0*x1" in target_split:
                binary_matrix_target[i, 1] = 1
            if "1.0*x1**2" in target_split:
                binary_matrix_target[i, 2] = 1
            if "1.0*x1*x1" in target_split:
                binary_matrix_target[i, 2] = 1
            if "1.0*x1**3" in target_split:
                binary_matrix_target[i, 3] = 1
            if "1.0*x1*x1*x1" in target_split:
                binary_matrix_target[i, 3] = 1
            if "1.0*x1**4" in target_split:
                binary_matrix_target[i, 4] = 1
            if "1.0*x1*x1*x1*x1" in target_split:
                binary_matrix_target[i, 4] = 1
            if "1.0*x1**5" in target_split:
                binary_matrix_target[i, 5] = 1
            if "1.0*x1*x1*x1*x1*x1" in target_split:
                binary_matrix_target[i, 5] = 1
            if "1.0*x1**6" in target_split:
                binary_matrix_target[i, 6] = 1
            if "1.0*x1*x1*x1*x1*x1*x1" in target_split:
                binary_matrix_target[i, 6] = 1
            if "1.0*x1*x1*x1*x1*x1*x1*x1*x1" in target_split:
                binary_matrix_target[i, 8] = 1
            if "1.0*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1" in target_split:
                binary_matrix_target[i, 10] = 1

            print('Target:{}\nSkeleton:{}'.format(target, predicted))

            o.write('{}\n'.format(target))
            o.write('{}:\n'.format('SymbolicGPT'))
            o.write('{}\n'.format(predicted))

            # train a regressor to find the constants (too slow)
            c = [1.0 for i, x in enumerate(predicted) if x == 'C']  # initialize coefficients as 1
            # c[-1] = 0 # initialize the constant as zero
            b = [(-2, 2) for i, x in enumerate(predicted) if x == 'C']  # bounds on variables
            try:
                if len(c) != 0:
                    # This is the bottleneck in our algorithm
                    # for easier comparison, we are using minimize package

                    predicted = predicted.replace('C', '1.0')
            except ValueError:
                raise 'Err: Wrong Equation {}'.format(predicted)
            except Exception as e:
                raise 'Err: Wrong Equation {}, Err: {}'.format(predicted, e)

            # TODO: let's enjoy GPU
            predicted_split = predicted.split("+")
            print("predicted split \n", predicted_split)

            if "1.0" in predicted_split:
                binary_matrix_predicted[i, 0] = 1
            if "1.0*x1" in predicted_split:
                binary_matrix_predicted[i, 1] = 1
            if "1.0*x1**2" in predicted_split:
                binary_matrix_predicted[i, 2] = 1
            if "1.0*x1*x1" in predicted_split:
                binary_matrix_predicted[i, 2] = 1
            if "1.0*x1**3" in predicted_split:
                binary_matrix_predicted[i, 3] = 1
            if "1.0*x1*x1*x1" in predicted_split:
                binary_matrix_predicted[i, 3] = 1
            if "1.0*x1**4" in predicted_split:
                binary_matrix_predicted[i, 4] = 1
            if "1.0*x1*x1*x1*x1" in predicted_split:
                binary_matrix_predicted[i, 4] = 1
            if "1.0*x1**5" in predicted_split:
                binary_matrix_predicted[i, 5] = 1
            if "1.0*x1*x1*x1*x1*x1" in predicted_split:
                binary_matrix_predicted[i, 5] = 1
            if "1.0*x1**6" in predicted_split:
                binary_matrix_predicted[i, 6] = 1
            if "1.0*x1*x1*x1*x1*x1*x1" in predicted_split:
                binary_matrix_predicted[i, 6] = 1
            if "1.0*x1*x1*x1*x1*x1*x1*x1*x1" in predicted_split:
                binary_matrix_predicted[i, 8] = 1
            if "1.0*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1" in predicted_split:
                binary_matrix_predicted[i, 10] = 1

            print('Skeleton+LS:{}'.format(predicted))
            difference = len(target) - len(predicted)
            difference_list.append(difference)

            Ys = []  # t['YT']
            Yhats = []
            invalid_updated = False
            valid_updated = False
            for xs in t['X']:
                try:
                    eqTmp = target + ''  # copy eq
                    eqTmp = eqTmp.replace(' ', '')
                    eqTmp = eqTmp.replace('\n', '')
                    for i, x in enumerate(xs):
                        # replace xi with the value in the eq
                        eqTmp = eqTmp.replace('x{}'.format(i + 1), str(x))
                        if ',' in eqTmp:
                            assert 'There is a , in the equation!'
                    YEval = eval(eqTmp)

                    # YEval = 0 if np.isnan(YEval) else YEval
                    # YEval = 100 if np.isinf(YEval) else YEval
                except:
                    print('TA: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                    print(i)
                    raise
                    continue  # if there is any point in the target equation that has any problem, ignore it
                    YEval = 100  # TODO: Maybe I have to punish the model for each wrong template not for each point
                Ys.append(YEval)

                try:
                    eqTmp = predicted + ''  # copy eq
                    eqTmp = eqTmp.replace(' ', '')
                    eqTmp = eqTmp.replace('\n', '')
                    for i, x in enumerate(xs):
                        # replace xi with the value in the eq
                        eqTmp = eqTmp.replace('x{}'.format(i + 1), str(x))
                        if ',' in eqTmp:
                            assert 'There is a , in the equation!'
                    Yhat = eval(eqTmp)
                    # Yhat = 0 if np.isnan(Yhat) else Yhat
                    # Yhat = 100 if np.isinf(Yhat) else Yhat
                    if not valid_updated:
                        num_valid += 1
                        valid_updated = True

                except:
                    print('PR: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                    Yhat = 100
                    if not invalid_updated:
                        num_invalid += 1
                        invalid_updated = True

                Yhats.append(Yhat)

            err = relativeErr(Ys, Yhats, info=True)

            if type(err) is np.complex128 or np.complex:
                err = abs(err.real)

            resultDict[fName]['SymbolicGPT'].append(err)

            o.write('{}\n{}\n\n'.format(
                predicted,
                err
            ))

            print('Err:{}'.format(err))

            print('')  # just an empty line
    print('Avg Err:{}'.format(np.mean(resultDict[fName]['SymbolicGPT'])))

except KeyboardInterrupt:
    print('KeyboardInterrupt')

# plot the error frequency for model comparison
num_eqns = len(resultDict[fName]['SymbolicGPT'])
num_vars = pconf.numberofVars
title = titleTemplate.format(num_eqns, num_vars)

models = list(key for key in resultDict[fName].keys() if len(resultDict[fName][key]) == num_eqns)
lists_of_error_scores = [resultDict[fName][key] for key in models if len(resultDict[fName][key]) == num_eqns]
print(lists_of_error_scores)
linestyles = ["-", "dashdot", "dotted", "--"]

error_array = np.asarray(lists_of_error_scores)
np.save('withpadding/errors_xx'+str(uniqueID), error_array)

print(difference_list)
difference_array = np.asarray(difference_list)
np.save('withpadding/diff_xx'+str(uniqueID), difference_array)

print(binary_matrix_target)
np.save('withpadding/bm_target_xx'+str(uniqueID), binary_matrix_target)

print(binary_matrix_predicted)
np.save('withpadding/bm_predicted_xx'+str(uniqueID), binary_matrix_predicted)

print("Number of invalid equations is: ", num_invalid)
print("Number of valid equations is: ", num_valid)
print('Done')
