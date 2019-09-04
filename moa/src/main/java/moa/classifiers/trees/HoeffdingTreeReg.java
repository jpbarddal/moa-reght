package moa.classifiers.trees;

import com.github.javacliparser.*;
import com.github.javacliparser.StringUtils;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.DiscreteAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NullAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.*;
import moa.options.ClassOption;

import java.util.*;

public class HoeffdingTreeReg extends HoeffdingTree {


    public MultiChoiceOption confidenceMethodOption = new MultiChoiceOption("confidenceMethod", 'C',
            "Determines the confidence method for computing the bound.",
            new String[]{"HOEFFDING", "MCDIARMID"}, new String[]{"HOEFFDING", "MCDIARMID"}, 0);

    public FloatOption lambdaOption = new FloatOption("lambda", 'L',
            "Lambda parameter for regularization.", 0.5, 0.0, 1.0);

    public MultiChoiceOption regularizationOption = new MultiChoiceOption("regularization", '*',
            "Strategy for checking minimum merit for splits.", new String[]{"AVG", "MAX"},
            new String[]{"AVG", "MAX"}, 1);

    public static class FoundNodeReg {

        public NodeReg node;

        public SplitNodeReg parent;

        public int parentBranch;

        public FoundNodeReg(NodeReg node, SplitNodeReg parent, int parentBranch) {
            this.node = node;
            this.parent = parent;
            this.parentBranch = parentBranch;
        }
    }

    public static class NodeReg extends Node {

        private static final long serialVersionUID = 1L;

        protected DoubleVector observedClassDistribution;

        public NodeReg(double[] classObservations) {
            this.observedClassDistribution = new DoubleVector(classObservations);
        }

        public int calcByteSize() {
            return (int) (SizeOf.sizeOf(this) + SizeOf.fullSizeOf(this.observedClassDistribution));
        }

        public int calcByteSizeIncludingSubtree() {
            return calcByteSize();
        }

        public boolean isLeaf() {
            return true;
        }

        public FoundNodeReg filterInstanceToLeaf(Instance inst, SplitNodeReg parent,
                                              int parentBranch) {
            return new FoundNodeReg(this, parent, parentBranch);
        }

        public double[] getObservedClassDistribution() {
            return this.observedClassDistribution.getArrayCopy();
        }

        public double[] getClassVotes(Instance inst, HoeffdingTreeReg ht) {
            return this.observedClassDistribution.getArrayCopy();
        }

        public boolean observedClassDistributionIsPure() {
            return this.observedClassDistribution.numNonZeroEntries() < 2;
        }

        public void describeSubtree(HoeffdingTreeReg ht, StringBuilder out,
                                    int indent) {
            StringUtils.appendIndented(out, indent, "Leaf ");
            out.append(ht.getClassNameString());
            out.append(" = ");
            out.append(ht.getClassLabelString(this.observedClassDistribution.maxIndex()));
            out.append(" weights: ");
            this.observedClassDistribution.getSingleLineDescription(out,
                    ht.treeRoot.observedClassDistribution.numValues());
            StringUtils.appendNewline(out);
        }

        public int subtreeDepth() {
            return 0;
        }

        public double calculatePromise() {
            double totalSeen = this.observedClassDistribution.sumOfValues();
            return totalSeen > 0.0 ? (totalSeen - this.observedClassDistribution.getValue(this.observedClassDistribution.maxIndex()))
                    : 0.0;
        }

        public void getDescription(StringBuilder sb, int indent) {
            describeSubtree(null, sb, indent);
        }
    }

    public static class SplitNodeReg extends NodeReg {

        private static final long serialVersionUID = 1L;

        protected InstanceConditionalTest splitTest;

        protected AutoExpandVector<NodeReg> children; // = new AutoExpandVector<Node>();

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
            for (NodeReg child : this.children) {
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

        public SplitNodeReg(InstanceConditionalTest splitTest,
                            int indicesFeaturesSelected[],
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

        public void setChild(int index, NodeReg child) {
            if ((this.splitTest.maxBranches() >= 0)
                    && (index >= this.splitTest.maxBranches())) {
                throw new IndexOutOfBoundsException();
            }
            this.children.set(index, child);
        }

        public NodeReg getChild(int index) {
            return this.children.get(index);
        }

        public int instanceChildIndex(Instance inst) {
            return this.splitTest.branchForInstance(inst);
        }

        @Override
        public boolean isLeaf() {
            return false;
        }

        public FoundNodeReg filterInstanceToLeaf(Instance inst, SplitNodeReg parent,
                                              int parentBranch) {
            int childIndex = instanceChildIndex(inst);
            if (childIndex >= 0) {
                NodeReg child = getChild(childIndex);
                if (child != null) {
                    return child.filterInstanceToLeaf(inst, this, childIndex);
                }
                return new FoundNodeReg(null, this, childIndex);
            }
            return new FoundNodeReg(this, parent, parentBranch);
        }

        public void describeSubtree(HoeffdingTreeReg ht, StringBuilder out,
                                    int indent) {
            for (int branch = 0; branch < numChildren(); branch++) {
                NodeReg child = getChild(branch);
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
            for (NodeReg child : this.children) {
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

    public static abstract class LearningNodeReg extends NodeReg {

        private static final long serialVersionUID = 1L;

        public LearningNodeReg(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        public abstract void learnFromInstance(Instance inst, HoeffdingTreeReg ht);
    }

    public static class InactiveLearningNodeReg extends LearningNodeReg {

        private static final long serialVersionUID = 1L;

        public InactiveLearningNodeReg(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTreeReg ht) {
            this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());
        }
    }

    public static class ActiveLearningNodeReg extends LearningNodeReg {

        private static final long serialVersionUID = 1L;

        protected double weightSeenAtLastSplitEvaluation;

        protected AutoExpandVector<AttributeClassObserver> attributeObservers = new AutoExpandVector<AttributeClassObserver>();

        protected boolean isInitialized;

        protected double lambda;

        public ActiveLearningNodeReg(double[] initialClassObservations, double lambda) {
            super(initialClassObservations);
            this.weightSeenAtLastSplitEvaluation = getWeightSeen();
            this.isInitialized = false;
            this.lambda = lambda;
        }

        @Override
        public int calcByteSize() {
            return super.calcByteSize()
                    + (int) (SizeOf.fullSizeOf(this.attributeObservers));
        }

        public void learnFromInstance(Instance inst, HoeffdingTreeReg ht) {
            if (this.isInitialized == false) {
                this.attributeObservers = new AutoExpandVector<AttributeClassObserver>(inst.numAttributes());
                this.isInitialized = true;
            }
            this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());
            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
            }
        }

        public double getWeightSeen() {
            return this.observedClassDistribution.sumOfValues();
        }

        public double getWeightSeenAtLastSplitEvaluation() {
            return this.weightSeenAtLastSplitEvaluation;
        }

        public void setWeightSeenAtLastSplitEvaluation(double weight) {
            this.weightSeenAtLastSplitEvaluation = weight;
        }

        public AttributeSplitSuggestion[] getBestSplitSuggestions(SplitCriterion criterion,
                                                                  HoeffdingTreeReg ht,
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
                    if(bestSuggestion != null &&
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

        private double count(int selected, int[] selectedFeatures) {
            double qty = 0.0;
            for(int i = 0; i < selectedFeatures.length; i++) if(selected == selectedFeatures[i]) qty++;
            return qty;
        }

        public void disableAttribute(int attIndex) {
            this.attributeObservers.set(attIndex,
                    new NullAttributeClassObserver());
        }

        private boolean contains(int a[], int v){
            for(int i = 0; i < a.length; i++){
                if(a[i] == v) return true;
            }
            return false;
        }

    }

//    public Node treeRoot;

    protected int numClasses;

    protected int decisionNodeCount;

    protected int activeLeafNodeCount;

    protected int inactiveLeafNodeCount;

    protected double inactiveLeafByteSizeEstimate;

    protected double activeLeafByteSizeEstimate;

    protected double byteSizeEstimateOverheadFraction;

    protected boolean growthAllowed;

    public int calcByteSize() {
        int size = (int) SizeOf.sizeOf(this);
        if (this.treeRoot != null) {
            size += this.treeRoot.calcByteSizeIncludingSubtree();
        }
        return size;
    }

    @Override
    public int measureByteSize() {
        return calcByteSize();
    }

    @Override
    public void resetLearningImpl() {
        this.treeRoot = null;
        this.decisionNodeCount = 0;
        this.activeLeafNodeCount = 0;
        this.inactiveLeafNodeCount = 0;
        this.inactiveLeafByteSizeEstimate = 0.0;
        this.activeLeafByteSizeEstimate = 0.0;
        this.byteSizeEstimateOverheadFraction = 1.0;
        this.growthAllowed = true;
        if (this.leafpredictionOption.getChosenIndex()>0) {
            this.removePoorAttsOption = null;
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.treeRoot == null) {
            this.treeRoot = newLearningNodeReg();
            this.activeLeafNodeCount = 1;
        }
        FoundNodeReg foundNode = ((NodeReg) treeRoot).filterInstanceToLeaf(inst, null, -1);
        NodeReg leafNode = foundNode.node;
        if (leafNode == null) {
            leafNode = newLearningNodeReg();
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
            this.activeLeafNodeCount++;
        }
        if (leafNode instanceof LearningNodeReg) {
            LearningNodeReg learningNode = (LearningNodeReg) leafNode;
            learningNode.learnFromInstance(inst, this);
            if (this.growthAllowed
                    && (learningNode instanceof ActiveLearningNodeReg)) {
                ActiveLearningNodeReg activeLearningNode = (ActiveLearningNodeReg) learningNode;
                double weightSeen = activeLearningNode.getWeightSeen();
                if (weightSeen
                        - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= this.gracePeriodOption.getValue()) {
                    attemptToSplit(activeLearningNode, foundNode.parent,
                            foundNode.parentBranch);
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }
            }
        }
        if (this.trainingWeightSeenByModel
                % this.memoryEstimatePeriodOption.getValue() == 0) {
            estimateModelByteSizes();
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.treeRoot != null) {
            FoundNodeReg foundNode = ((NodeReg) treeRoot).filterInstanceToLeaf(inst,
                    null, -1);
            NodeReg leafNode = foundNode.node;
            if (leafNode == null) {
                leafNode = foundNode.parent;
            }
            return leafNode.getClassVotes(inst, this);
        } else {
            numClasses = inst.dataset().numClasses();
            return new double[numClasses];
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{
                new Measurement("tree size (nodes)", this.decisionNodeCount
                        + this.activeLeafNodeCount + this.inactiveLeafNodeCount),
                new Measurement("tree size (leaves)", this.activeLeafNodeCount
                        + this.inactiveLeafNodeCount),
                new Measurement("active learning leaves",
                        this.activeLeafNodeCount),
                new Measurement("tree depth", measureTreeDepth()),
                new Measurement("active leaf byte size estimate",
                        this.activeLeafByteSizeEstimate),
                new Measurement("inactive leaf byte size estimate",
                        this.inactiveLeafByteSizeEstimate),
                new Measurement("byte size estimate overhead",
                        this.byteSizeEstimateOverheadFraction)};
    }

    public int measureTreeDepth() {
        if (this.treeRoot != null) {
            return this.treeRoot.subtreeDepth();
        }
        return 0;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        this.treeRoot.describeSubtree(this, out, indent);
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    public static double computeHoeffdingBound(double range, double confidence,
                                               double n) {
        return Math.sqrt(((range * range) * Math.log(1.0 / confidence))
                / (2.0 * n));
    }

    public static double computeMcDiarmidInfoGain(int numClasses, double confidence, double n){
        double cGain = 6 * (numClasses * log2(n * Math.E) + log2(2 * n)) + 2 * log2(numClasses);
        double remainder = Math.sqrt(Math.log(1.0 / confidence) / 2 * n);
        double epsilon = cGain * remainder;
        return epsilon;
    }

    public static double computeMcDiarmidGini(double confidence, double n){
        double epsilon = 8 * Math.sqrt(Math.log(1 / confidence) / (1 * n));
        return epsilon;
    }

    public static double log2(double val){
        return Math.log10(val) / Math.log10(2.0);
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


    protected AttributeClassObserver newNominalClassObserver() {
        AttributeClassObserver nominalClassObserver = (AttributeClassObserver) getPreparedClassOption(this.nominalEstimatorOption);
        return (AttributeClassObserver) nominalClassObserver.copy();
    }

    protected AttributeClassObserver newNumericClassObserver() {
        AttributeClassObserver numericClassObserver = (AttributeClassObserver) getPreparedClassOption(this.numericEstimatorOption);
        return (AttributeClassObserver) numericClassObserver.copy();
    }

    protected void attemptToSplit(ActiveLearningNodeReg node, SplitNodeReg parent,
                                  int parentIndex) {

        int indicesSelected[] = null;
        double meritsSelected[] = null;
        if(parent != null) {
            indicesSelected = new int[parent.indicesFeaturesSelected.length + 1];
            System.arraycopy(parent.indicesFeaturesSelected, 0, indicesSelected, 0, parent.indicesFeaturesSelected.length);
            meritsSelected = new double[parent.meritsFeaturesSelected.length + 1];
            System.arraycopy(parent.meritsFeaturesSelected, 0, meritsSelected, 0, parent.meritsFeaturesSelected.length);
        }else{
            indicesSelected = new int[1]; // room for a single value that will be selected later in the code
            meritsSelected = new double[1];
        }

        if (!node.observedClassDistributionIsPure()) {
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this, indicesSelected);
            Arrays.sort(bestSplitSuggestions);
            boolean shouldSplit = false;
            if (bestSplitSuggestions.length < 2) {
                shouldSplit = bestSplitSuggestions.length > 0;
            } else {
                double bound = 0.0;
                if(confidenceMethodOption.getChosenLabel().equals("HOEFFDING")) {
                    bound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                            this.splitConfidenceOption.getValue(), node.getWeightSeen());
                }else if(confidenceMethodOption.getChosenLabel().equals("MCDIARMID")){
                    if(this.splitCriterionOption.toString().contains("InfoGain")){
                        bound = computeMcDiarmidInfoGain(numClasses, this.splitConfidenceOption.getValue(), node.getWeightSeen());
                    }else{
                        bound = computeMcDiarmidGini(this.splitConfidenceOption.getValue(), node.getWeightSeen());
                    }
                }
                AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];

                double minValueForSplit = computeMinValueForSplit(bestSuggestion.splitTest.getAttsTestDependsOn()[0], indicesSelected, meritsSelected);
                if (((bestSuggestion.merit - secondBestSuggestion.merit > bound)
                        || (bound < this.tieThresholdOption.getValue())) && bestSuggestion.merit > minValueForSplit) {
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
                                        - bestSplitSuggestions[i].merit > bound) {
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
                                        - bestSplitSuggestions[i].merit < bound) {
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
                    meritsSelected[meritsSelected.length   - 1] = splitDecision.merit;
                    SplitNodeReg newSplit = newSplitNode(splitDecision.splitTest, indicesSelected, meritsSelected,
                            node.getObservedClassDistribution(), splitDecision.numSplits());
                    for (int i = 0; i < splitDecision.numSplits(); i++) {
                        NodeReg newChild = newLearningNodeReg(splitDecision.resultingClassDistributionFromSplit(i));
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

    protected double computeMinValueForSplit(int selected, int[] indicesSelected, double[] meritsSelected) {
        double val = 0.0;
        if(regularizationOption.getChosenLabel().equalsIgnoreCase("AVG")){
            int qtd = 0;
            for(int i = 0; i < indicesSelected.length; i++){
                if(selected == indicesSelected[i]){
                    val += meritsSelected[i];
                    qtd++;
                }
            }
            if(qtd > 0) val /= qtd;
        }else if(regularizationOption.getChosenLabel().equalsIgnoreCase("MAX")){
            for(int i = 0; i < indicesSelected.length; i++) {
                if (selected == indicesSelected[i] && meritsSelected[i] > val) {
                    val = meritsSelected[i];
                }
            }
        }
        return val;
    }

    public void enforceTrackerLimit() {
        if ((this.inactiveLeafNodeCount > 0)
                || ((this.activeLeafNodeCount * this.activeLeafByteSizeEstimate + this.inactiveLeafNodeCount
                * this.inactiveLeafByteSizeEstimate)
                * this.byteSizeEstimateOverheadFraction > this.maxByteSizeOption.getValue())) {
            if (this.stopMemManagementOption.isSet()) {
                this.growthAllowed = false;
                return;
            }
            FoundNodeReg[] learningNodes = findLearningNodesReg();
            Arrays.sort(learningNodes, new Comparator<FoundNodeReg>() {

                @Override
                public int compare(FoundNodeReg fn1, FoundNodeReg fn2) {
                    return Double.compare(fn1.node.calculatePromise(), fn2.node.calculatePromise());
                }
            });
            int maxActive = 0;
            while (maxActive < learningNodes.length) {
                maxActive++;
                if ((maxActive * this.activeLeafByteSizeEstimate + (learningNodes.length - maxActive)
                        * this.inactiveLeafByteSizeEstimate)
                        * this.byteSizeEstimateOverheadFraction > this.maxByteSizeOption.getValue()) {
                    maxActive--;
                    break;
                }
            }
            int cutoff = learningNodes.length - maxActive;
            for (int i = 0; i < cutoff; i++) {
                if (learningNodes[i].node instanceof ActiveLearningNodeReg) {
                    deactivateLearningNode(
                            (ActiveLearningNodeReg) learningNodes[i].node,
                            learningNodes[i].parent,
                            learningNodes[i].parentBranch);
                }
            }
            for (int i = cutoff; i < learningNodes.length; i++) {
                if (learningNodes[i].node instanceof InactiveLearningNodeReg) {
                    activateLearningNode(
                            (InactiveLearningNodeReg) learningNodes[i].node,
                            learningNodes[i].parent,
                            learningNodes[i].parentBranch);
                }
            }
        }
    }

    public void estimateModelByteSizes() {
        FoundNodeReg[] learningNodes = findLearningNodesReg();
        long totalActiveSize = 0;
        long totalInactiveSize = 0;
        for (FoundNodeReg foundNode : learningNodes) {
            if (foundNode.node instanceof ActiveLearningNodeReg) {
                totalActiveSize += SizeOf.fullSizeOf(foundNode.node);
            } else {
                totalInactiveSize += SizeOf.fullSizeOf(foundNode.node);
            }
        }
        if (totalActiveSize > 0) {
            this.activeLeafByteSizeEstimate = (double) totalActiveSize
                    / this.activeLeafNodeCount;
        }
        if (totalInactiveSize > 0) {
            this.inactiveLeafByteSizeEstimate = (double) totalInactiveSize
                    / this.inactiveLeafNodeCount;
        }
        int actualModelSize = this.measureByteSize();
        double estimatedModelSize = (this.activeLeafNodeCount
                * this.activeLeafByteSizeEstimate + this.inactiveLeafNodeCount
                * this.inactiveLeafByteSizeEstimate);
        this.byteSizeEstimateOverheadFraction = actualModelSize
                / estimatedModelSize;
        if (actualModelSize > this.maxByteSizeOption.getValue()) {
            enforceTrackerLimit();
        }
    }

    public void deactivateAllLeaves() {
        FoundNodeReg[] learningNodes = findLearningNodesReg();
        for (int i = 0; i < learningNodes.length; i++) {
            if (learningNodes[i].node instanceof ActiveLearningNodeReg) {
                deactivateLearningNode(
                        (ActiveLearningNodeReg) learningNodes[i].node,
                        learningNodes[i].parent, learningNodes[i].parentBranch);
            }
        }
    }

    protected void deactivateLearningNode(ActiveLearningNodeReg toDeactivate,
                                          SplitNodeReg parent, int parentBranch) {
        NodeReg newLeaf = new InactiveLearningNodeReg(toDeactivate.getObservedClassDistribution());
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }
        this.activeLeafNodeCount--;
        this.inactiveLeafNodeCount++;
    }

    protected void activateLearningNode(InactiveLearningNodeReg toActivate,
                                        SplitNodeReg parent, int parentBranch) {
        NodeReg newLeaf = newLearningNodeReg(toActivate.getObservedClassDistribution());
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }
        this.activeLeafNodeCount++;
        this.inactiveLeafNodeCount--;
    }

    protected FoundNodeReg[] findLearningNodesReg() {
        List<FoundNodeReg> foundList = new LinkedList<FoundNodeReg>();
        findLearningNodesReg(((NodeReg) treeRoot), null, -1, foundList);
        return foundList.toArray(new FoundNodeReg[foundList.size()]);
    }

    protected void findLearningNodesReg(NodeReg node, SplitNodeReg parent,
                                     int parentBranch, List<FoundNodeReg> found) {
        if (node != null) {
            if (node instanceof LearningNodeReg) {
                found.add(new FoundNodeReg(node, parent, parentBranch));
            }
            if (node instanceof SplitNodeReg) {
                SplitNodeReg splitNode = (SplitNodeReg) node;
                for (int i = 0; i < splitNode.numChildren(); i++) {
                    findLearningNodesReg(splitNode.getChild(i), splitNode, i,
                            found);
                }
            }
        }
    }

    public static class LearningNodeNBReg extends ActiveLearningNodeReg {

        private static final long serialVersionUID = 1L;

        public LearningNodeNBReg(double[] initialClassObservations, double lambda) {
            super(initialClassObservations, lambda);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTreeReg ht) {
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

    public static class LearningNodeNBAdaptiveReg extends LearningNodeNBReg {

        private static final long serialVersionUID = 1L;

        protected double mcCorrectWeight = 0.0;

        protected double nbCorrectWeight = 0.0;

        public LearningNodeNBAdaptiveReg(double[] initialClassObservations, double lambda) {
            super(initialClassObservations, lambda);
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTreeReg ht) {
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
        public double[] getClassVotes(Instance inst, HoeffdingTreeReg ht) {
            if (this.mcCorrectWeight > this.nbCorrectWeight) {
                return this.observedClassDistribution.getArrayCopy();
            }
            return NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers);
        }
    }

    protected LearningNodeReg newLearningNodeReg() {
        return newLearningNodeReg(new double[0]);
    }

    protected LearningNodeReg newLearningNodeReg(double[] initialClassObservations) {
        LearningNodeReg ret;
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        double lambda = lambdaOption.getValue();
        if (predictionOption == 0) { //MC
            ret = new ActiveLearningNodeReg(initialClassObservations, lambda);
        } else if (predictionOption == 1) { //NB
            ret = new LearningNodeNBReg(initialClassObservations, lambda);
        } else { //NBAdaptive
            ret = new LearningNodeNBAdaptiveReg(initialClassObservations, lambda);
        }
        return ret;
    }
}
