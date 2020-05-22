# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:26:59 2019

@author: Soubeiga Armel
"""

########################################################################################################
def split_balanced(data, target, test_size=0.2):

    classes = np.unique(target)
    # can give test_size as fraction of input data size of number of samples
    if test_size<1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
        
    n_train = max(0,len(target)-n_test)
    n_train_per_class = max(1,int(np.floor(n_train/len(classes))))
    n_test_per_class = max(1,int(np.floor(n_test/len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class+n_test_per_class) > np.sum(target==cl):
            splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target==cl)))
            ixs.append(np.r_[np.random.choice(np.nonzero(target==cl)[0][:splitix], n_train_per_class),
                np.random.choice(np.nonzero(target==cl)[0][splitix:], n_test_per_class)])
        else:
            ixs.append(np.random.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class,
                replace=False))

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])

    X_train = data[ix_train,:]
    X_test = data[ix_test,:]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test


xxtrain = xtrain.values

x_train, x_test, y_train, y_test = split_balanced(xxtrain, ytrain,test_size = 0.3)
x_train.shape
x_test.shape


classes, counts = np.unique(y_train, return_counts=True)
plt.bar(classes, counts)
plt.ylabel('number of instances')
plt.title('dataset constitution')
plt.xlabel('wine category')
plt.xticks(range(4), np.unique(ytrain),rotation='vertical')
plt.show()


def split_balanced_2 (target,data, trainSize=0.8, getTestIndexes=True, shuffle=False, seed=None):
    classes, counts = np.unique(target, return_counts=True)
    nPerClass = float(len(target))*float(trainSize)/float(len(classes))
    if nPerClass > np.min(counts):
        print("Le nombre de classes %s"%classes)
        print("Le nombres d'obs. par classe %s"%counts)
        ts = float(trainSize*np.min(counts)*len(classes)) / float(len(target))
        print("trainSize %s testSize %s"%(trainSize, ts))
        trainSize = ts
        nPerClass = float(len(target))*float(trainSize)/float(len(classes))
    # obtient le nombre de classe
    nPerClass = int(nPerClass)
    print("Nous avons  %i classes et la séparation donne %i par classe"%(len(classes),nPerClass ))
   
    # Les index
    trainIndexes = []
    for c in classes:
        if seed is not None:
            np.random.seed(seed)
        cIdxs = np.where(target==c)[0]
        cIdxs = np.random.choice(cIdxs, nPerClass, replace=False)
        trainIndexes.extend(cIdxs)
    # get test indexes
    testIndexes = None
    if getTestIndexes:
        testIndexes = list(set(range(len(target))) - set(trainIndexes))
        
    # return le train et le test
    if shuffle:
        trainIndexes = random.shuffle(trainIndexes)
        if testIndexes is not None:
            testIndexes = random.shuffle(testIndexes)
    
    # retourn moi les base separer
    
    X_train = data[trainIndexes,:]
    X_test = data[testIndexes,:]
    y_train = target[trainIndexes]
    y_test = target[testIndexes]
    
    return X_train, X_test, y_train, y_test

x_train, x_test, y_train, y_test= split_balanced_2(ytrain,xxtrain, trainSize=0.7, getTestIndexes=True, shuffle=False, seed=None)


classes, counts = np.unique(y_train, return_counts=True)
plt.bar(classes, counts)
plt.ylabel('number of instances')
plt.title('dataset constitution')
plt.xlabel('wine category')
plt.xticks(range(4), np.unique(ytrain),rotation='vertical')
plt.show()

#########################################################################################################
