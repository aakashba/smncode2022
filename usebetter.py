import sys
import pickle
import argparse
import re
import random

random.seed(1337)

import statistics

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score

from scipy.stats import ttest_rel

import numpy as np

#from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word

    


def use(reflist, predlist,predlist2):
        import tensorflow_hub as tfhub

        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        model = tfhub.load(module_url)
        #model2 = tfhub.load(module_url)
        #model3 = tfhub.load(module_url)
        refs = list()
        preds = list()
        preds2 = list()
        count = 0
        for ref, pred,pred2 in zip(reflist, predlist,predlist2):
                #print(ref)
            ref = ' '.join(ref[0]).strip()
            pred = ' '.join(pred).strip()
            pred2 = ' '.join(pred2).strip()
            #if pred == '':
             #   pred = ' <s> '
            refs.append(ref)
            preds.append(pred)
            preds2.append(pred2)

        #total_csd = np.zeros(count)
        scores = list()
        scores2 = list()
        #for i in range(0, len(refs)):
         #   if i % 100 == 0:
          #      print(i)
           # ref_emb = model([refs[i]])
           # pred_emb = model2([preds[i]])
           # pred2_emb = model3([preds2[i]])
           # scores.append(cosine_similarity(ref_emb,pred_emb))
           # scores2.append(cosine_similarity(ref_emb,pred2_emb))
        #conc = refs+preds+preds2
        #conc_emb = model(conc)
        ref_emb = model(refs)
        pred_emb = model(preds)
        pred2_emb = model(preds2)
        #l = len(refs)
        #ref_emb = conc_emb[0:l]
        #pred_emb = conc_emb[l:l+l]
        #pred2_emb = conc_emb[l+l:]
        
        csm = cosine_similarity_score(ref_emb, pred_emb)
        csm2 = cosine_similarity_score(ref_emb, pred2_emb)
        csd = csm.diagonal()
        csd2 = csm2.diagonal()
	  #  total_csd = csd #np.concatenate([total_csd, csd])
        scores = csd.tolist()
        scores2 = csd2.tolist()
        
        return scores,scores2

def cosine_similarity(x, y):
    #from numpy import dot
    #from numpy.linalg import norm
    #result = dot(x, y)/(norm(x)*norm(y))
    from scipy import spatial
    result = 1 - spatial.distance.cosine(x, y)
    return result


def cosine_similarity_score(x, y):
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarity_matrix = cosine_similarity(x, y)
    return cosine_similarity_matrix





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

def meteor_so_far_m_only(refs, preds):
    scores, m = corpus_meteor(refs, preds)
    m = round(m*100, 2)
    return m

def meteor_so_far(refs, preds):
    
    scores, m = corpus_meteor(refs, preds)
    m = round(m*100, 2)
    
    ret = ''
    ret += ('for %s functions\n' % (len(preds)))
    ret += ('M %s\n' % (m))
    return scores, m, ret


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
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--not-diffonly', dest='diffonly', action='store_false', default=True)
    parser.add_argument('--shuffles', type=int, default=100)

    args = parser.parse_args()
    outdir = args.outdir
    dataprep = args.dataprep
    inputA_file = args.inputA
    inputB_file = args.inputB
    challenge = args.challenge
    obfuscate = args.obfuscate
    gpu=args.gpu
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
        predsA[fid] = pred
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
        predsB[fid] = pred
    predictsB.close()
    #drop()

    refs = list()
    samesA=list()
    samesB=list()
    samesRefs=list()
    betterA = list()
    worseA = list()
    worseB = list()
    betterB = list()
    sameAB = list()
    predA=list()
    predB=list()
    d = 0
    targets = open('%s/output/coms.test' % (dataprep), 'r')
    for line in targets:
        (fid, com) = line.split('<SEP>')
        fid = int(fid)
        com = com.split()
        com = fil(com) 
        if len(com) < 1:
            continue
        
        if(predsA[fid] == predsB[fid]):
            samesA.append(predsA[fid])
            samesB.append(predsB[fid])
            samesRefs.append([com])
            continue


        refs.append([com])
        predA.append(predsA[fid])
        predB.append(predsB[fid])
    
    scoresamesA,scoresamesB= use(samesRefs,samesA, samesB)
    
#    noeq = 0
#    for n in range(len(scoresamesA)):
#        if scoresamesA[n] != scoresamesB[n]:
#            noeq += 1
#            print(samesA[n])
#            print(samesB[n])

#    print(noeq)

    scoreA,scoreB = use(refs,predA,predB)

    for n in range(len(scoreA)):
        
        mA=scoreA[n]
        mB=scoreB[n]
        
        if mA > mB:
            betterA.append(mA)
            worseB.append(mB)
        elif mA < mB:
            betterB.append(mB)
            worseA.append(mA)
        else:
            sameAB.append(mA)
    
    sameAB = sameAB+scoresamesA

    betterAm = round(np.average(betterA)*100,2)
    worseAm = round(np.average(worseA)*100,2)
    betterBm = round(np.average(betterB)*100,2)
    worseBm = round(np.average(worseB)*100,2)
    sameABm = round(np.average(sameAB)*100,2)


    print("Where {} does better:\n".format(inputA_file))
    print("number of methods:{}\n".format(len(betterA)))
    print("{} gets:{}\n".format(inputA_file,betterAm))
    print("{} gets:{}\n".format(inputB_file,worseBm))
    ttest = ttest_rel(betterA, worseB, alternative='greater')
    print(ttest)



    print("Where {} does better:\n".format(inputB_file))
    print("number of methods:{}\n".format(len(betterB)))
    print("{} gets:{}\n".format(inputB_file,betterBm))
    print("{} gets:{}\n".format(inputA_file,worseAm))
    ttest = ttest_rel(betterB, worseA, alternative='greater')
    print(ttest)


    print("Where both get simiar results")
    print("number of methods:{}\n".format(len(sameAB)))
    print("same score:{}\n".format(sameABm))

    



