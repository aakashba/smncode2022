import pickle
import sys
import os
import math
import traceback
import argparse
import signal
import atexit
import time
import h5py

import random
import tensorflow as tf
import numpy as np
from pathlib import Path

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#import tensorflow.keras as keras
#import tensorflow.keras.utils
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
#import tensorflow.keras.backend as K

#from model import create_model
##from myutils import prep, drop, batch_gen, seq2sent

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

import tokenizer

class HistoryCallback(Callback):
    
    def setCatchExit(self, outdir, modeltype, timestart, mdlconfig):
        self.outdir = outdir
        self.modeltype = modeltype
        self.history = {}
        self.timestart = timestart
        self.mdlconfig = mdlconfig
        
        atexit.register(self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)
    
    def handle_exit(self, *args):
        if len(self.history.keys()) > 0:
            try:
                Path(outdir+'/histories').mkdir(parents=True, exist_ok=True)
                fn = outdir+'/histories/'+self.modeltype+'_hist_'+str(self.timestart)+'.pkl'
                histoutfd = open(fn, 'wb')
                pickle.dump(self.history, histoutfd)
                print('saved history to: ' + fn)
                
                fn = outdir+'/histories/'+self.modeltype+'_conf_'+str(self.timestart)+'.pkl'
                confoutfd = open(fn, 'wb')
                pickle.dump(self.mdlconfig, confoutfd)
                print('saved config to: ' + fn)
            except Exception as ex:
                print(ex)
                traceback.print_exc(file=sys.stdout)
        sys.exit()
    
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        filepath = self.outdir+'/models/'+self.modeltype+f'_E{epoch:02d}_'+str(self.timestart)+'.h5' 
        keras.models.save_model(model, filepath, overwrite=True, include_optimizer=False)
        
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


if __name__ == '__main__':

    timestart = int(round(time.time()))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=200)
    parser.add_argument('--batchgen', dest='batchgen', type=str, default='regular')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--model-type', dest='modeltype', type=str, default='vanilla')
    parser.add_argument('--with-graph', dest='withgraph', action='store_true', default=False)
    parser.add_argument('--with-calls', dest='withcalls', action='store_true', default=False)
    parser.add_argument('--with-biodats', dest='withbiodats', type=str , default='vanilla')
    parser.add_argument('--with-simmat', dest='withsimmats', action='store_true', default=False)
    parser.add_argument('--with-codevec', dest='withcodevec', action='store_true', default=False)
    parser.add_argument('--simmat-file', dest='simmatfile', type=str, default='softmax_usec.pkl')
    parser.add_argument('--loss-type', dest='losstype', type=str, default='cce')
    parser.add_argument('--vmem-limit', dest='vmemlimit', type=int, default=0)
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/smn/data/javastmt/q90')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--hops', dest='hops', type=int, default= 5)
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--datfile', dest='datfile', type=str, default='dataset.pkl')
    parser.add_argument('--only-print-summary', dest='onlyprintsummary', action='store_true', default=False)
    parser.add_argument('--memory-network-input', dest='memory_network_input', type=str, default='positional-encoding',
                        help='the input module for memory networks. default: positional-encoding (from the '
                             '"End-To-End Memory Networks"). The other option: eos-embedding (from the "Ask Me Anything: '
                             'Dynamic Memory Networks for Natural Language Processing")')
    parser.add_argument('--max-sent-len', dest='maxsentlen', type=int, default=50,
                        help='for the memory networks, set the maximum length of the sentences/lines. default: 50')
    parser.add_argument('--max-sent-cnt', dest='maxsentcnt', type=int, default=50,
                        help='for the memory networks, set the maximum number of lines. default: 50')
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    hops = args.hops
    batch_size = args.batch_size
    batchgen = args.batchgen
    epochs = args.epochs
    modeltype = args.modeltype
    withgraph = args.withgraph
    withcalls = args.withcalls
    withcodevec = args.withcodevec
    withbiodats = False
    withsimmat = args.withsimmats
    simmatfile = args.simmatfile
    losstype = args.losstype
    vmemlimit = args.vmemlimit
    onlyprintsummary = args.onlyprintsummary
    memorynetwork_input = args.memory_network_input
    max_sentence_len = args.maxsentlen
    max_sentence_cnt = args.maxsentcnt

    if memorynetwork_input != "positional-encoding" and memorynetwork_input != "eos-embedding":
        print('memory-network-input: {} is not a valid option. use deafult: positional-encoding'.format(memorynetwork_input))
        memorynetwork_input = "positional-encoding"
    #datfile = args.datfile
    if args.withbiodats != 'vanilla':
        withbiodats = True
        biodatfile = args.withbiodats

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                if(vmemlimit > 0):
                    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vmemlimit)])
        except RuntimeError as e:
            print(e)

    #if(vmemlimit > 0):
    #    if gpus:
    #        try:
    #            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vmemlimit)])
    #        except RuntimeError as e:
    #            print(e)

    import tensorflow.keras as keras
    import tensorflow.keras.utils
    #from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
    import tensorflow.keras.backend as K

    from model import create_model

    K.set_floatx(args.dtype)

    if batchgen == 'qs':
        from qs_myutils import prep, drop, batch_gen, seq2sent
    else:
        from myutils import prep, drop, batch_gen, seq2sent
    print(dataprep)
    prep('loading sequences... ')
    sqlfile = '{}/rawdats.sqlite'.format(dataprep)
    extradata = pickle.load(open('%s/dataset_short.pkl' % (dataprep), 'rb'))
    seqdata = h5py.File('%s/dataset_seqs.h5' % (dataprep), 'r')
    drop()


    if withgraph:
        prep('loading graph data... ')
        graphdata = pickle.load(open('%s/dataset_graph.pkl' % (dataprep), 'rb'))
        for k, v in extradata.items():
            graphdata[k] = v
        extradata = graphdata
        drop()

    if withcalls:
        prep('loading call data... ')
        callnodes = pickle.load(open('%s/callsnodes.pkl' % (dataprep), 'rb'))
        calledges = pickle.load(open('%s/callsedges.pkl' % (dataprep), 'rb'))
        callnodesdata = pickle.load(open('%s/callsnodedata.pkl' % (dataprep), 'rb'))
        extradata['callnodes'] = callnodes
        extradata['calledges'] = calledges
        extradata['callnodedata'] = callnodesdata
        drop()

    if withbiodats:
        prep('loading biomodel results... ')
        biodats = pickle.load(open(biodatfile, 'rb'))
        extradata['biodats'] = biodats
        drop()
    
    if withcodevec:
        prep('loading codevec... ')
        codevecfile = h5py.File("%s/q90codebert.h5" % (dataprep), 'r')
        extradata['codevec'] = codevecfile
        drop()
    
    if withsimmat:
        prep('loading target comwords distribution... ')
        softmax_usemat = pickle.load(open('%s/%s' % (dataprep, simmatfile), 'rb'))
        extradata['target_dist'] = softmax_usemat
        drop()

    prep('loading tokenizers... ')
    comstok = extradata['comstok']
    tdatstok = extradata['tdatstok']
    sdatstok = tdatstok
    smlstok = extradata['smlstok']
    if withgraph:
        graphtok = extradata['graphtok']
    drop()

    if batchgen == 'qs':
        steps = int(np.array(seqdata.get('/ctrain').shape[0])/batch_size)*int(np.array(seqdata.get('/ctrain')).shape[1])
        valsteps = int(np.array(seqdata.get('/cval').shape[0])/batch_size)*int(np.array(seqdata.get('/ctrain')).shape[1])
    else:
        steps = int(np.array(seqdata.get('/ctrain').shape[0])/batch_size)
        valsteps = int(np.array(seqdata.get('/cval').shape[0])/batch_size)
    
    tdatvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smlstok.vocab_size

    print('tdatvocabsize %s' % (tdatvocabsize))
    print('comvocabsize %s' % (comvocabsize))
    print('smlvocabsize %s' % (smlvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps*batch_size))
    print('vaidation data size {}'.format(valsteps*batch_size))
    print('------------------------------------------')
    print('for memory networks:')
    print('input module type: %s' % (memorynetwork_input))
    print('------------------------------------------')

    config = dict()
    config['hops'] = hops
    config['tdatvocabsize'] = tdatvocabsize
    config['comvocabsize'] = comvocabsize
    config['smlvocabsize'] = smlvocabsize

    try:
        config['fidloc'] = extradata['fidloc']
        config['locfid'] = extradata['locfid']
        config['comstok'] = extradata['comstok']
        config['comlen'] = int(np.array(seqdata.get('/ctrain')).shape[1])
        config['tdatlen'] = int(np.array(seqdata.get('/dttrain')).shape[1])
        config['sdatlen'] = extradata['config']['sdatlen']
        config['smllen'] = int(np.array(seqdata.get('/strain')).shape[1])
        config['batchgen'] = batchgen
        config['target_dist'] = extradata['target_dist']
    except KeyError:
        pass # some configurations do not have all data, which is fine

    config['batch_size'] = batch_size
    config['memorynetwork_input'] = memorynetwork_input
    config['max_sentence_len'] = max_sentence_len
    config['max_sentence_cnt'] = max_sentence_cnt
    config['loss_type'] = losstype

    print(config.keys())

    prep('creating model... ')
    config, model = create_model(modeltype, config)
    drop()

    print(model.summary())
    
    if onlyprintsummary:
        sys.exit()

    gen = batch_gen(seqdata, extradata, 'train', config)
    Path(outdir+'/models').mkdir(parents=True, exist_ok=True)
    #checkpoint = ModelCheckpoint(outdir+'/'+modeltype+'_E{epoch:02d}_TA{acc:.2f}_VA{val_acc:.2f}_VB{val_bleu:}.h5', monitor='val_loss')
    #checkpoint = ModelCheckpoint(outdir+'/models/'+modeltype+'_E{epoch:02d}_'+str(timestart)+'.h5')
    savehist = HistoryCallback()
    savehist.setCatchExit(outdir, modeltype, timestart, config)
    
    valgen = batch_gen(seqdata, extradata, 'val', config)

    # If you want it to calculate BLEU Score after each epoch use callback_valgen and test_cb
    #####
    #callback_valgen = batch_gen_train_bleu(seqdata, comvocabsize, 'val', modeltype, batch_size=batch_size)
    #test_cb = mycallback(callback_valgen, steps)
    #####
    callbacks = [ savehist ]

    try:
        history = model.fit(x=gen, steps_per_epoch=steps, epochs=epochs, verbose=1, max_queue_size=8, workers=1, use_multiprocessing=False, callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)
        last_model_filepath = outdir + '/models/' + modeltype + '_last-epoch_' + str(timestart) + '.h5'
        keras.models.save_model(model, last_model_filepath, overwrite=True, include_optimizer=False)

    except Exception as ex:
        print(ex)
        traceback.print_exc(file=sys.stdout)
