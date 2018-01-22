import gzip
import pickle

with gzip.open('lstm2v_duracloud_duraspace_dim50.pkl.gz','r') as featVector:
    print featVector
    t1,t2,t3,t4,t5,t6 = pickle.load(featVector)
    print t4
