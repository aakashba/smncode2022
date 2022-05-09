import tensorflow.keras as keras
import tensorflow as tf

from models.attendgru import AttentionGRUModel as attendgru
from models.codegnngru import CodeGNNGRUModel as codegnngru
from models.code2seq import Code2SeqModel as code2seq
from models.smn import StatementMemoryNetworks as smn
from models.smn_query import StatementMemoryNetworks_query as smn_query
from models.smn_h1 import StatementMemoryNetworks_h1 as smn_h1
from models.smn_h2 import StatementMemoryNetworks_h2 as smn_h2
from models.smn_h4 import StatementMemoryNetworks_h4 as smn_h4
from models.smn_h5 import StatementMemoryNetworks_h5 as smn_h5
from models.transformer_base import TransformerBase as xformer_base
from models.han_code import HANcode2 as hancode2

# from models.attendgru_bio2 import AttentionGRUBio2Model as attendgru_bio2

def create_model(modeltype, config):
    mdl = None

    elif modeltype == 'codegnngru':
        mdl = codegnngru(config)
    elif modeltype == 'code2seq':
        mdl = code2seq(config)
    elif modeltype == 'smn-h1':
        mdl = smn_h1(config)
    elif modeltype == 'smn-h2':
        mdl = smn_h2(config)
    elif modeltype == 'smn-h4':
        mdl = smn_h4(config)
    elif modeltype == 'smn-h5':
        mdl = smn_h5(config)
    elif modeltype == 'smn':
        mdl = smn(config)
    elif modeltype == 'hancode2':
        mdl = hancode2(config)
    elif modeltype == 'smn_query':
        mdl = smn_query(config)
    elif modeltype == 'transformer-base':
        mdl = xformer_base(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
