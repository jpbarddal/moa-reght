package moa.classifiers.trees;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.core.Utils;

public class ARFHoeffdingTreeRegularizer extends ARFHoeffdingTree {
    public IntOption subspaceSizeOption = new IntOption("subspaceSizeSize", 'k',
            "Number of features per subset for each node split. Negative values = #features - k",
            2, Integer.MIN_VALUE, Integer.MAX_VALUE);

    @Override
    public String getPurposeString() {
        return "Adaptive Random Forest Hoeffding Tree for data streams. "
                + "Base learner for AdaptiveRandomForest.";
    }

    public static class RandomLearningNode extends ActiveLearningNode {

        private static final long serialVersionUID = 1L;

        protected int[] listAttributes;

        protected int numAttributes;

        public RandomLearningNode(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations);
            this.numAttributes = subspaceSize;
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());
            if (this.listAttributes == null) {
                this.listAttributes = new int[this.numAttributes];
                for (int j = 0; j < this.numAttributes; j++) {
                    boolean isUnique = false;
                    while (isUnique == false) {
                        this.listAttributes[j] = ht.classifierRandom.nextInt(inst.numAttributes() - 1);
                        isUnique = true;
                        for (int i = 0; i < j; i++) {
                            if (this.listAttributes[j] == this.listAttributes[i]) {
                                isUnique = false;
                                break;
                            }
                        }
                    }

                }
            }
            for (int j = 0; j < this.numAttributes - 1; j++) {
                int i = this.listAttributes[j];
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
            }
        }
    }

    public static class LearningNodeNB extends ARFHoeffdingTree.RandomLearningNode {

        private static final long serialVersionUID = 1L;

        public LearningNodeNB(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations, subspaceSize);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            if (getWeightSeen() >= ht.nbThresholdOption.getValue()) {
                return NaiveBayes.doNaiveBayesPrediction(inst,
                        this.observedClassDistribution,
                        this.attributeObservers);
            }
            return super.getClassVotes(inst, ht);
        }

        @Override
        public void disableAttribute(int attIndex) {
            // should not disable poor atts - they are used in NB calc
        }
    }

    public static class LearningNodeNBAdaptive extends ARFHoeffdingTree.LearningNodeNB {

        private static final long serialVersionUID = 1L;

        protected double mcCorrectWeight = 0.0;

        protected double nbCorrectWeight = 0.0;

        public LearningNodeNBAdaptive(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations, subspaceSize);
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            int trueClass = (int) inst.classValue();
            if (this.observedClassDistribution.maxIndex() == trueClass) {
                this.mcCorrectWeight += inst.weight();
            }
            if (Utils.maxIndex(NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers)) == trueClass) {
                this.nbCorrectWeight += inst.weight();
            }
            super.learnFromInstance(inst, ht);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            if (this.mcCorrectWeight > this.nbCorrectWeight) {
                return this.observedClassDistribution.getArrayCopy();
            }
            return NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers);
        }
    }

    public ARFHoeffdingTreeRegularizer() {
        this.removePoorAttsOption = null;
    }

    @Override
    protected LearningNode newLearningNode(double[] initialClassObservations) {
        LearningNode ret;
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        if (predictionOption == 0) { //MC
            ret = new ARFHoeffdingTree.RandomLearningNode(initialClassObservations, this.subspaceSizeOption.getValue());
        } else if (predictionOption == 1) { //NB
            ret = new ARFHoeffdingTree.LearningNodeNB(initialClassObservations, this.subspaceSizeOption.getValue());
        } else { //NBAdaptive
            ret = new ARFHoeffdingTree.LearningNodeNBAdaptive(initialClassObservations, this.subspaceSizeOption.getValue());
        }
        return ret;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}
