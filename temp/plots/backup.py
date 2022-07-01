
# Prepare train data for all 5 folds
X1trains, X2trains, ytrains, trainweights = [], [], [], []

for i in range(len(ytests)):
    
    # Training data (combine all folds except test fold)
    X1train, X2train = np.zeros((1,437,21)), np.zeros((1,437,21))
    ytrain, trainweight = np.zeros(1), np.zeros(1)
                                                                           
    for k in range(len(ytests)):
        if k == i:
            continue
        X1train = np.append(X1train, X1tests[k], axis=0)
        X2train = np.append(X2train, X2tests[k], axis=0)
        ytrain = np.append(ytrain, ytests[k], axis=0)
        trainweight = np.append(trainweight, testweights[k], axis=0)
    X1trains.append(X1train[1:])
    X2trains.append(X2train[1:])
    ytrains.append(ytrain[1:])
    trainweights.append(trainweight[1:])
   

# Ensure array shapes match
for i in range(len(ytests)):
    assert X1trains[i].shape[0] == X2trains[i].shape[0] == ytrains[i].shape[0] == \
            trainweights[i].shape[0]
    assert X1tests[i].shape[0] == X2tests[i].shape[0] == ytests[i].shape[0] == \
            testweights[i].shape[0]
    print(f'Train fold {i} = {X1trains[i].shape[0]}')
    print(f'Test fold {i} = {X1tests[i].shape[0]}')
    print()
 

# Data of 3 choice PETases (for reference)
Xref = {}
for name in ['IsPETase_WT', 'TfCut2', 'LCC_WT']:
    loc = heads.index(name)
    Xref[name] = Xlabel[loc][None,:,:]
    
    
    
    
    

X1s = X1s[1:]  
X2s = X2s[1:]
weights = weights[1:]
yints = yints[1:]
pair_names = pair_names[1:]
dataset_names = dataset_names[1:]
assert len(X1s) == len(X2s) == len(weights) == len(yints) == len(pair_names) == \
        len(dataset_names)
