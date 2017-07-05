import sys
import pickle
sys.path.insert(0, '//code/bhtsne')
import bhtsne

with open('//code/logs/cifarVGGfc8trainfeats.pkl','rb') as f:
    pca_feats = pickle.load(f)

print(pca_feats.shape)

embedding_array = bhtsne.run_bh_tsne(pca_feats, initial_dims=pca_feats.shape[1],verbose=True)

with open('//code/logs/tsned_cifar_train_fc8.pkl','wb') as f:
    pickle.dump(embedding_array,f)
