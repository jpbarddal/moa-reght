package moa.classifiers.trees;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.StringUtils;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;
import moa.core.SizeOf;
import moa.core.Utils;

import java.util.*;

public class ARFHoeffdingTreeRegularizer extends ARFHoeffdingTree {

    public FloatOption lambdaOption = new FloatOption("lambda", 'L',
            "Lambda parameter for regularization.", 0.5, 0.0, 1.0);

    public MultiChoiceOption regularizationOption = new MultiChoiceOption("regularization", '*',
            "Strategy for checking minimum merit for splits.", new String[]{"AVG", "MAX"},
            new String[]{"AVG", "MAX"}, 1);

    @Override
    public String getPurposeString() {
        return "Adaptive Random Forest Hoeffding Tree for data streams. "
                + "Base learner for AdaptiveRandomForest.";
    }

    public static class RandomLearningNode extends ARFHoeffdingTree.ActiveLearningNode {

        private static final long serialVersionUID = 1L;

        protected int[] listAttributes;

        protected int numAttributes;

        protected double lambda;

        public RandomLearningNode(double[] initialClassObservations, int subspaceSize, double lambda) {
            super(initialClassObservations);
            this.numAttributes = subspaceSize;
            this.lambda = lambda;
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


        public AttributeSplitSuggestion[] getBestSplitSuggestions(SplitCriterion criterion,
                                                                  ARFHoeffdingTreeRegularizer ht,
                                                                  int selectedFeatures[]) {
            List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
            double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
            if (!ht.noPrePruneOption.isSet()) {
                // add null split as an option
                bestSuggestions.add(new AttributeSplitSuggestion(null,
                        new double[0][], criterion.getMeritOfSplit(
                        preSplitDist,
                        new double[][]{preSplitDist})));
            }
            for (int i = 0; i < this.attributeObservers.size(); i++) {
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs != null) {
                    AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
                            preSplitDist, i, ht.binarySplitsOption.isSet());
                    if (bestSuggestion != null &&
                            bestSuggestion.splitTest != null && bestSuggestion.splitTest.getAttsTestDependsOn() != null) {
                        int selected = bestSuggestion.splitTest.getAttsTestDependsOn()[0];
                        if (!contains(selectedFeatures, selected)) {
                            bestSuggestion.merit *= lambda;
                        }/*else{
                        double gamma = 1.0 - (count(selected, selectedFeatures) / selectedFeatures.length);
                        bestSuggestion.merit *= gamma;
                    }*/
                    }
                    if (bestSuggestion != null) {
                        bestSuggestions.add(bestSuggestion);
                    }
                }
            }
            return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
        }

        private boolean contains(int a[], int v) {
            for (int i = 0; i < a.length; i++) {
                if (a[i] == v) return true;
            }
            return false;
        }
    }

    public static class RandomLearningNodeNB extends RandomLearningNode {

        private static final long serialVersionUID = 1L;

        public RandomLearningNodeNB(double[] initialClassObservations, int subspaceSize, double lambda) {
            super(initialClassObservations, subspaceSize, lambda);
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

    public static class RandomLearningNodeNBAdaptive extends RandomLearningNodeNB {

        private static final long serialVersionUID = 1L;

        protected double mcCorrectWeight = 0.0;

        protected double nbCorrectWeight = 0.0;

        public RandomLearningNodeNBAdaptive(double[] initialClassObservations, int subspaceSize, double lambda) {
            super(initialClassObservations, subspaceSize, lambda);
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


    protected RandomLearningNode newLearningNode() {
        return newLearningNode(new double[0]);
    }

    protected RandomLearningNode newLearningNode(double[] initialClassObservations) {
        RandomLearningNode ret;
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        double lambda = lambdaOption.getValue();
        if (predictionOption == 0) { //MC
            ret = new RandomLearningNode(initialClassObservations, subspaceSizeOption.getValue(), lambda);
        } else if (predictionOption == 1) { //NB
            ret = new RandomLearningNodeNB(initialClassObservations, subspaceSizeOption.getValue(), lambda);
        } else { //NBAdaptive
            ret = new RandomLearningNodeNBAdaptive(initialClassObservations, subspaceSizeOption.getValue(), lambda);
        }
        return ret;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    protected double computeMinValueForSplit(int selected, int[] indicesSelected, double[] meritsSelected) {
        double val = 0.0;
        if (regularizationOption.getChosenLabel().equalsIgnoreCase("AVG")) {
            int qtd = 0;
            for (int i = 0; i < indicesSelected.length; i++) {
                if (selected == indicesSelected[i]) {
                    val += meritsSelected[i];
                    qtd++;
                }
            }
            if (qtd > 0) val /= qtd;
        } else if (regularizationOption.getChosenLabel().equalsIgnoreCase("MAX")) {
            for (int i = 0; i < indicesSelected.length; i++) {
                if (selected == indicesSelected[i] && meritsSelected[i] > val) {
                    val = meritsSelected[i];
                }
            }
        }
        return val;
    }


    protected void attemptToSplit(ActiveLearningNode node,
                                  SplitNode parent,
                                  int parentIndex) {

        int indicesSelected[] = null;
        double meritsSelected[] = null;
        if (parent != null) {
            indicesSelected = new int[((SplitNodeReg) parent).indicesFeaturesSelected.length + 1];
            System.arraycopy(((SplitNodeReg) parent).indicesFeaturesSelected, 0, indicesSelected,
                    0, ((SplitNodeReg) parent).indicesFeaturesSelected.length);
            meritsSelected = new double[((SplitNodeReg) parent).meritsFeaturesSelected.length + 1];
            System.arraycopy(((SplitNodeReg) parent).meritsFeaturesSelected, 0, meritsSelected,
                    0, ((SplitNodeReg) parent).meritsFeaturesSelected.length);
        } else {
            indicesSelected = new int[1]; // room for a single value that will be selected later in the code
            meritsSelected = new double[1];
        }

        if (!node.observedClassDistributionIsPure()) {
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
            AttributeSplitSuggestion[] bestSplitSuggestions = ((RandomLearningNode) node).getBestSplitSuggestions(splitCriterion, this, indicesSelected);
            Arrays.sort(bestSplitSuggestions);
            boolean shouldSplit = false;
            if (bestSplitSuggestions.length < 2) {
                shouldSplit = bestSplitSuggestions.length > 0;
            } else {
                double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                        this.splitConfidenceOption.getValue(), node.getWeightSeen());
                AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];

                double minValueForSplit = computeMinValueForSplit(bestSuggestion.splitTest.getAttsTestDependsOn()[0], indicesSelected, meritsSelected);
                if (((bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)
                        || (hoeffdingBound < this.tieThresholdOption.getValue())) && bestSuggestion.merit > minValueForSplit) {
                    shouldSplit = true;
                }
                // }
                if ((this.removePoorAttsOption != null)
                        && this.removePoorAttsOption.isSet()) {
                    Set<Integer> poorAtts = new HashSet<Integer>();
                    // scan 1 - add any poor to set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit > hoeffdingBound) {
                                    poorAtts.add(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    // scan 2 - remove good ones from set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit < hoeffdingBound) {
                                    poorAtts.remove(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    for (int poorAtt : poorAtts) {
                        node.disableAttribute(poorAtt);
                    }
                }
            }

            if (shouldSplit) {
                AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                if (splitDecision.splitTest == null) {
                    // preprune - null wins
                    deactivateLearningNode(node, parent, parentIndex);
                } else {
                    indicesSelected[indicesSelected.length - 1] = splitDecision.splitTest.getAttsTestDependsOn()[0];
                    meritsSelected[meritsSelected.length - 1] = splitDecision.merit;
                    SplitNodeReg newSplit = newSplitNode(splitDecision.splitTest, indicesSelected, meritsSelected,
                            node.getObservedClassDistribution(), splitDecision.numSplits());
                    for (int i = 0; i < splitDecision.numSplits(); i++) {
                        Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));
                        newSplit.setChild(i, newChild);
                    }
                    this.activeLeafNodeCount--;
                    this.decisionNodeCount++;
                    this.activeLeafNodeCount += splitDecision.numSplits();
                    if (parent == null) {
                        this.treeRoot = newSplit;
                    } else {
                        parent.setChild(parentIndex, newSplit);
                    }
                }
                // manage memory
                enforceTrackerLimit();
            }
        }
    }

    public static class SplitNodeReg extends HoeffdingTree.SplitNode {

        private static final long serialVersionUID = 1L;

//        protected InstanceConditionalTest splitTest;
//
//        protected AutoExpandVector<LearningNode> children; // = new AutoExpandVector<Node>();

        protected int indicesFeaturesSelected[];

        protected double meritsFeaturesSelected[];

        @Override
        public int calcByteSize() {
            return super.calcByteSize()
                    + (int) (SizeOf.sizeOf(this.children) + SizeOf.fullSizeOf(this.splitTest));
        }

        @Override
        public int calcByteSizeIncludingSubtree() {
            int byteSize = calcByteSize();
            for (Node child : this.children) {
                if (child != null) {
                    byteSize += child.calcByteSizeIncludingSubtree();
                }
            }
            return byteSize;
        }


        public SplitNodeReg(InstanceConditionalTest splitTest, int indicesFeaturesSelected[],
                            double meritsFeaturesSelected[],
                            double[] classObservations, int size) {
            super(classObservations);
            this.splitTest = splitTest;
            this.indicesFeaturesSelected = indicesFeaturesSelected;
            this.meritsFeaturesSelected = meritsFeaturesSelected;
            this.children = new AutoExpandVector<>(size);
        }

        public SplitNodeReg(InstanceConditionalTest splitTest, int indicesFeaturesSelected[],
                            double meritsFeaturesSelected[],
                            double[] classObservations) {
            super(classObservations);
            this.splitTest = splitTest;
            this.indicesFeaturesSelected = indicesFeaturesSelected;
            this.meritsFeaturesSelected = meritsFeaturesSelected;
            this.children = new AutoExpandVector<>();
        }


        public int numChildren() {
            return this.children.size();
        }

        public void setChild(int index, LearningNode child) {
            if ((this.splitTest.maxBranches() >= 0)
                    && (index >= this.splitTest.maxBranches())) {
                throw new IndexOutOfBoundsException();
            }
            this.children.set(index, child);
        }

        public Node getChild(int index) {
            return this.children.get(index);
        }

        public int instanceChildIndex(Instance inst) {
            return this.splitTest.branchForInstance(inst);
        }

        @Override
        public boolean isLeaf() {
            return false;
        }

        public FoundNode filterInstanceToLeaf(Instance inst,
                                              SplitNodeReg parent,
                                              int parentBranch) {
            int childIndex = instanceChildIndex(inst);
            if (childIndex >= 0) {
                Node child = getChild(childIndex);
                if (child != null) {
                    return child.filterInstanceToLeaf(inst, this, childIndex, null);
                }
                return new FoundNode(null, this, childIndex);
            }
            return new FoundNode(this, parent, parentBranch);
        }

        public void describeSubtree(HoeffdingTreeReg ht, StringBuilder out,
                                    int indent) {
            for (int branch = 0; branch < numChildren(); branch++) {
                Node child = getChild(branch);
                if (child != null) {
                    StringUtils.appendIndented(out, indent, "if ");
                    out.append(this.splitTest.describeConditionForBranch(branch,
                            ht.getModelContext()));
                    out.append(": ");
                    StringUtils.appendNewline(out);
                    child.describeSubtree(ht, out, indent + 2);
                }
            }
        }

        @Override
        public int subtreeDepth() {
            int maxChildDepth = 0;
            for (Node child : this.children) {
                if (child != null) {
                    int depth = child.subtreeDepth();
                    if (depth > maxChildDepth) {
                        maxChildDepth = depth;
                    }
                }
            }
            return maxChildDepth + 1;
        }
    }

    //Procedure added for Hoeffding Adaptive Trees (ADWIN)
    protected SplitNodeReg newSplitNode(InstanceConditionalTest splitTest, int indicesFeaturesSelected[],
                                                         double meritsFeaturesSelected[],
                                                         double[] classObservations, int size) {
        return new SplitNodeReg(splitTest, indicesFeaturesSelected, meritsFeaturesSelected, classObservations, size);
    }

    protected SplitNodeReg newSplitNode(InstanceConditionalTest splitTest, int indicesFeaturesSelected[],
                                                         double meritsFeaturesSelected[],
                                                         double[] classObservations) {
        return new SplitNodeReg(splitTest, indicesFeaturesSelected, meritsFeaturesSelected, classObservations);
    }

}
