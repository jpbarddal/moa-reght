import os

maxInstances = 500000
windowWidth = 10000
sampleFreq = 10000
memCheck = 10000
ensembleSize = 100

base_classifiers = [
    'trees.HoeffdingTree -w',
    'trees.EFDT',
    'trees.HoeffdingTreeReg -C MCDIARMID -L 1.0 -w', # MCDIARMID TREE
    'meta.AdaptiveRandomForest -s 100 -j 50',  # ARF100
]

regularized_classifiers = []

# adding hoeffding trees
for lambda_val in range(0, 10, 0.5):
    line = 'trees.HoeffdingTreeReg -L {} -w'.format(lambda_val)
    regularized_classifiers.append(line)

# adding EFDT trees
for lambda_val in range(0, 10, 0.5):
    line = 'trees.EFDTReg -L {}'.format(lambda_val)
    regularized_classifiers.append(line)

# adding mcdiarmid trees
for lambda_val in range(0, 10, 0.5):
    line = 'trees.HoeffdingTreeReg -C MCDIARMID -L {} -w'.format(lambda_val)
    regularized_classifiers.append(line)

# adding ARF
for lambda_val in range(0, 10, 0.5):
    line = 'meta.AdaptiveRandomForest -l (ARFHoeffdingTreeRegularizer -L {} -w) -s 100 -j 50'.format(lambda_val)
    regularized_classifiers.append(line)

classifiers = base_classifiers + regularized_classifiers

streams_classif = [
    'generators.SEAGenerator',
    'generators.AgrawalGenerator',
    'generators.AssetNegotiationGenerator',
    'generators.RandomTreeGenerator'
]

# base_regressors = [
#     'trees.FIMTDD -s VarianceReductionSplitCriterion -w', # FIMTDD
# ]
# regularized_regressors = []
# regressors = base_regressors + regularized_regressors
# streams_reg = [
#     'generators.HyperplaneGeneratorReg -a 25 -k 0 -m Distance',
#     'generators.HyperplaneGeneratorReg -a 25 -k 0 -m SquareDistance',
#     'generators.HyperplaneGeneratorReg -a 25 -k 0 -m CubicDistance'
# ]

# CLASSIFICATION
for classifier in classifiers:
    for stream in streams_classif:
            print("\n" +  40 * "#")
            print("Training classifier: ", classifier)
            print("With stream: ", stream)
            print(40 * "#" + "\n")

            str_classifier = classifier.replace(' ', '_').replace(')', '').replace('(', '')
            str_stream = stream.replace(' ', '_').replace(')', '').replace('(', '')
            path_output = str_stream + '_' + str_classifier
            config = "EvaluatePrequential -l  ({}) ".format(classifier)
            config += "-s ({}) ".format(stream)
            config += "-e (WindowClassificationPerformanceEvaluator -w " + str(windowWidth) + " -o -p -r -f) "
            config += "-i " + str(maxInstances) + " -f " + str(sampleFreq) + " -q " + str(memCheck) + " "
            config += "-d ./{}.csv".format(path_output)
            command = "java -cp moa-pom.jar moa.DoTask \"{}\"".format(config)
            print(command)
            # os.system(command)

# REGRESSION
