import sys
import pickle
import argparse
import re
import random

random.seed(1337)

import statistics

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score

from scipy.stats import ttest_rel

import numpy as np

#from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word

    

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.asarray(x)
    x = x.astype(np.float)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputA', type=str, default=None)
    parser.add_argument('inputB', type=str, default=None)
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/funcom/data/javastmt')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--obfuscate', action='store_true', default=False)
    parser.add_argument('--sbt', action='store_true', default=False)
    parser.add_argument('--not-diffonly', dest='diffonly', action='store_false', default=True)
    parser.add_argument('--shuffles', type=int, default=100)

    args = parser.parse_args()
    outdir = args.outdir
    dataprep = args.dataprep
    inputA_file = args.inputA
    inputB_file = args.inputB
    challenge = args.challenge
    obfuscate = args.obfuscate
    sbt = args.sbt
    diffonly = args.diffonly
    R = args.shuffles

    if challenge:
        dataprep = '../data/challengeset/output'

    if obfuscate:
        dataprep = '../data/obfuscation/output'

    if sbt:
        dataprep = '../data/sbt/output'

    if inputA_file is None:
        print('Please provide an input file to test with --input')
        exit()

    sys.path.append(dataprep)
    import tokenizer
    
    #prep('preparing predictions list A... ')
    predsA = dict()
    predictsA = open(inputA_file, 'r')
    for c, line in enumerate(predictsA):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        predsA[fid] = ' '.join(pred)
    predictsA.close()
    #drop()

    #prep('preparing predictions list B... ')
    predsB = dict()
    predictsB = open(inputB_file, 'r')
    for c, line in enumerate(predictsB):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        predsB[fid] = ' '.join(pred)
    predictsB.close()
    #drop()

    
    refs = dict()
    worddiff = dict()
    betterA = list()
    worseA = list()
    worseB = list()
    betterB = list()
    sameAB = list()
    fidbd = dict()
    smlnd = dict()
    d = 0
    targets = open('%s/output/coms.test' % (dataprep), 'r')
    for line in targets:
        (fid, com) = line.split('<SEP>')
        fid = int(fid)
        com = com.split()
        com = fil(com)
        
        if len(com) < 1:
            continue

        mA= corpus_bleu([[com]], [predsA[fid]])
        mB= corpus_bleu([[com]], [predsB[fid]])
        refs[fid] = com

        if mA > mB:
            betterA.append(predsA[fid])
            worseB.append(predsB[fid])
        elif mA < mB:
            betterB.append(predsB[fid])
            worseA.append(predsA[fid])
        else:
            sameAB.append(mA)
        
    betterAm = round(np.mean(betterA)*100,2)
    worseAm = round(np.mean(worseA)*100,2)
    betterBm = round(np.mean(betterB)*100,2)
    worseBm = round(np.mean(worseB)*100,2)
    sameABm = round(np.mean(sameAB)*100,2)


    print("Where {} does better:\n".format(inputA_file))
    print("number of methods:{}\n".format(len(betterA)))
    print("{} gets:{}\n".format(inputA_file,betterAm))
    print("{} gets:{}\n".format(inputB_file,worseBm))



    print("Where {} does better:\n".format(inputB_file))
    print("number of methods:{}\n".format(len(betterB)))
    print("{} gets:{}\n".format(inputB_file,betterBm))
    print("{} gets:{}\n".format(inputA_file,worseAm))

    print("Where both get simiar results")
    print("number of methods:{}\n".format(len(sameAB)))
    print("same score:{}\n".format(sameABm))

    



