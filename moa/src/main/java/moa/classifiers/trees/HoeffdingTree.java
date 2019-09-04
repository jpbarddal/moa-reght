/*
 *    HoeffdingTree.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.trees;

import com.github.javacliparser.FileOption;

import java.io.IOException;
import java.util.*;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import moa.AbstractMOAObject;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
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
import com.yahoo.labs.samoa.instances.Instance;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;

import moa.streams.filters.FeatureScore;

/**
 * Hoeffding Tree or VFDT.
 * <p>
 * A Hoeffding tree is an incremental, anytime decision tree induction algorithm
 * that is capable of learning from massive data streams, assuming that the
 * distribution generating examples does not change over time. Hoeffding trees
 * exploit the fact that a small sample can often be enough to choose an optimal
 * splitting attribute. This idea is supported mathematically by the Hoeffding
 * bound, which quantiﬁes the number of observations (in our case, examples)
 * needed to estimate some statistics within a prescribed precision (in our
 * case, the goodness of an attribute).</p> <p>A theoretically appealing feature
 * of Hoeffding Trees not shared by other incremental decision tree learners is
 * that it has sound guarantees of performance. Using the Hoeffding bound one
 * can show that its output is asymptotically nearly identical to that of a
 * non-incremental learner using inﬁnitely many examples. See for details:</p>
 * <p>
 * <p>G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
 * In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.</p>
 * <p>
 * <p>Parameters:</p> <ul> <li> -m : Maximum memory consumed by the tree</li>
 * <li> -n : Numeric estimator to use : <ul> <li>Gaussian approximation
 * evaluating 10 splitpoints</li> <li>Gaussian approximation evaluating 100
 * splitpoints</li> <li>Greenwald-Khanna quantile summary with 10 tuples</li>
 * <li>Greenwald-Khanna quantile summary with 100 tuples</li>
 * <li>Greenwald-Khanna quantile summary with 1000 tuples</li> <li>VFML method
 * with 10 bins</li> <li>VFML method with 100 bins</li> <li>VFML method with
 * 1000 bins</li> <li>Exhaustive binary tree</li> </ul> </li> <li> -e : How many
 * instances between memory consumption checks</li> <li> -g : The number of
 * instances a leaf should observe between split attempts</li> <li> -s : Split
 * criterion to use. Example : InfoGainSplitCriterion</li> <li> -c : The
 * allowable error in split decision, values closer to 0 will take longer to
 * decide</li> <li> -t : Threshold below which a split will be forced to break
 * ties</li> <li> -b : Only allow binary splits</li> <li> -z : Stop growing as
 * soon as memory limit is hit</li> <li> -r : Disable poor attributes</li> <li>
 * -p : Disable pre-pruning</li>
 * <li> -l : Leaf prediction to use: MajorityClass (MC), Naive Bayes (NB) or NaiveBayes
 * adaptive (NBAdaptive).</li>
 * <li> -q : The number of instances a leaf should observe before
 * permitting Naive Bayes</li>
 * </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class HoeffdingTree extends AbstractClassifier
        implements MultiClassClassifier, FeatureScore {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Hoeffding Tree or VFDT.";
    }

    public IntOption maxByteSizeOption = new IntOption("maxByteSize", 'm',
            "Maximum memory consumed by the tree.", 33554432, 0,
            Integer.MAX_VALUE);

    /*
     * public MultiChoiceOption numericEstimatorOption = new MultiChoiceOption(
     * "numericEstimator", 'n', "Numeric estimator to use.", new String[]{
     * "GAUSS10", "GAUSS100", "GK10", "GK100", "GK1000", "VFML10", "VFML100",
     * "VFML1000", "BINTREE"}, new String[]{ "Gaussian approximation evaluating
     * 10 splitpoints", "Gaussian approximation evaluating 100 splitpoints",
     * "Greenwald-Khanna quantile summary with 10 tuples", "Greenwald-Khanna
     * quantile summary with 100 tuples", "Greenwald-Khanna quantile summary
     * with 1000 tuples", "VFML method with 10 bins", "VFML method with 100
     * bins", "VFML method with 1000 bins", "Exhaustive binary tree"}, 0);
     */
    public ClassOption numericEstimatorOption = new ClassOption("numericEstimator",
            'n', "Numeric estimator to use.", NumericAttributeClassObserver.class,
            "GaussianNumericAttributeClassObserver");

    public ClassOption nominalEstimatorOption = new ClassOption("nominalEstimator",
            'd', "Nominal estimator to use.", DiscreteAttributeClassObserver.class,
            "NominalAttributeClassObserver");

    public IntOption memoryEstimatePeriodOption = new IntOption(
            "memoryEstimatePeriod", 'e',
            "How many instances between memory consumption checks.", 1000000,
            0, Integer.MAX_VALUE);

    public IntOption gracePeriodOption = new IntOption(
            "gracePeriod",
            'g',
            "The number of instances a leaf should observe between split attempts.",
            200, 0, Integer.MAX_VALUE);

    public ClassOption splitCriterionOption = new ClassOption("splitCriterion",
            's', "Split criterion to use.", SplitCriterion.class,
            "InfoGainSplitCriterion");

    public FloatOption splitConfidenceOption = new FloatOption(
            "splitConfidence",
            'c',
            "The allowable error in split decision, values closer to 0 will take longer to decide.",
            0.0000001, 0.0, 1.0);

    public FloatOption tieThresholdOption = new FloatOption("tieThreshold",
            't', "Threshold below which a split will be forced to break ties.",
            0.05, 0.0, 1.0);

    public FlagOption binarySplitsOption = new FlagOption("binarySplits", 'b',
            "Only allow binary splits.");

    public FlagOption stopMemManagementOption = new FlagOption(
            "stopMemManagement", 'z',
            "Stop growing as soon as memory limit is hit.");

    public FlagOption removePoorAttsOption = new FlagOption("removePoorAtts",
            'r', "Disable poor attributes.");

    public FlagOption noPrePruneOption = new FlagOption("noPrePrune", 'p',
            "Disable pre-pruning.");

    public FlagOption useMinMeritForSplitOption = new FlagOption("useMinMeritForSplit", 'w', "");
    public FloatOption minMeritForSplitOption = new FloatOption("minMeritForSplit", 'M',
            "Threshold for minimum merit.", 1e-10, 0.0, 1.0);

    public FlagOption outputTreeOption = new FlagOption("outputTree", 'T', "Determine whether trees should be outputted.");

    public IntOption debugOutputTrainingWeightSeenOption = new IntOption("debugOutputTrainingWeightSeen", 'j',
            "The training weight seen before outputing a new tree to a new file.", 10000, 1, Integer.MAX_VALUE);

    public FileOption debugDotFilePrefixOption = new FileOption("debugDotFilePrefix", 'v',
            "File prefix to create the output dot tree files.", "tree", "dot", true);

    public FlagOption doNotUseParentStatisticsOption = new FlagOption("doNotUseParentStatistics",
            'x', "Do not take into account the statistics from the parent of a node during split.");

    public static class FoundNode {

        public Node node;

        public SplitNode parent;

        public int parentBranch;

        public FoundNode(Node node, SplitNode parent, int parentBranch) {
            this.node = node;
            this.parent = parent;
            this.parentBranch = parentBranch;
        }
    }


    // For feature score/importance calculation.
    protected double[] featureScores;
    protected int currentNodeCount = 0;
    protected boolean shouldTrainFeatureScore = true;

    // For a evaluation metric
    double totalInstancesSeen = 0.0;
    double weightedHits = 0.0;

    /***
     * Update the tree model using an instance and updates featureImportance. 
     * Update the featureImportance array only if a splits happens
     *  There will be no changes in the scores only by updating the leaves. 
     *  This calculation is based on the Mean Decrease Impurity (MDI), more
     *  details about how it is calculated can be found in here: 
     *      Louppe, G., Wehenkel, L., Sutera, A., & Geurts, P. (2013). 
     *      Understanding variable importances in forests of randomized trees. 
     *      In Advances in neural information processing systems (pp. 431-439).
     * @param instance
     */
    @Override
    public boolean updateFeaturesScore(Instance instance) {
        // Instantiate the featureImportance array. 
        //  instance.numAttributes() - 1 is used to not allocate an array space to the class variable.
        if (this.featureScores == null)
            this.featureScores = new double[instance.numAttributes() - 1];

        // Update the tree model
        //  When used from within a random forest, then it will not be updated through here.
        if (this.shouldTrainFeatureScore)
            this.trainOnInstance(instance);

        return this.getNodeCount() > this.currentNodeCount;
    }

    public int getNodeCount() {
        return this.decisionNodeCount + this.activeLeafNodeCount + this.inactiveLeafNodeCount;
    }

    public void setShouldTrainFeatureScore(boolean value) {
        this.shouldTrainFeatureScore = value;
    }

    public boolean getShouldTrainFeatureScore() {
        return this.shouldTrainFeatureScore;
    }

    private void calcMeanDecreaseImpurity(Node node) {
        if (node instanceof SplitNode) {
            SplitNode splitNode = (SplitNode) node;
            int attributeIndex = splitNode.getSplitTest().getAttsTestDependsOn()[0];

            if (this.featureScores.length <= attributeIndex) {
                System.out.println("Going to explode!!!");
            }

            this.featureScores[attributeIndex] += calcNodeDecreaseImpurity(splitNode);

            for (Node childNode : splitNode.children) {
                if (childNode != null)
                    calcMeanDecreaseImpurity(childNode);
            }
        }
    }

    public double calcNodeDecreaseImpurity(SplitNode splitNode) {
        double nodeImpurity = splitNode.getImpurity();
        //(int) Utils.sum(splitNode.getObservedClassDistribution());
        double sumChildrenImpurityDecrease = 0;

        // Sum the children class label distribution, whatever instance
        //  that arrive at a child node, will have passed by the current node. 
        int childrenNumInstances = 0;
        for (Node childNode : splitNode.children) {
            if (childNode != null) {
                double childImpurity = childNode.getImpurity();
                int childNumInstances = (int) Utils.sum(childNode.getObservedClassDistribution());
                childrenNumInstances += childNumInstances;

                sumChildrenImpurityDecrease += childImpurity * childNumInstances;
            }
        }

        return nodeImpurity * childrenNumInstances - sumChildrenImpurityDecrease;
    }

    @Override
    public double[] getFeaturesScore(boolean normalized) {
        if (this.treeRoot != null) {
            // Check if there were changes to the tree topology (new nodes)
            //  If not, it is not necessary to recalculated featureScores, 
            //  just return the current values. 
            if (this.getNodeCount() > this.currentNodeCount) {
                this.featureScores = new double[this.featureScores.length];
                currentNodeCount = this.getNodeCount();

                // If there was a split, then recalculate scores. 
                // TODO: This can be more efficiently, if intermediary computations
                //  are kept in memory. For example, store each node absolute contribution
                //  to MDI and then just add the new one when a split occurs. 
                this.calcMeanDecreaseImpurity(this.treeRoot);

                // normalize the featureScores
                if (normalized) {
                    double sumFeatureScores = Utils.sum(this.featureScores);
                    for (int i = 0; i < this.featureScores.length; ++i) {
                        this.featureScores[i] /= sumFeatureScores;
                    }
                }
            }
        }

        return this.featureScores;
    }

    @Override
    public int[] getTopKFeatures(int k, boolean normalize) {
        // It is important to use the method to access featureScores, if
        //  in the future they are calculate as they are accessed, this 
        //  method won't require changes.

        if (this.getFeaturesScore(normalize) == null)
            return null;
        if (k > this.getFeaturesScore(normalize).length)
            k = this.getFeaturesScore(normalize).length;

        int[] topK = new int[k];
        double[] currentFeatureScores = new double[this.getFeaturesScore(normalize).length];
        for (int i = 0; i < currentFeatureScores.length; ++i)
            currentFeatureScores[i] = this.getFeaturesScore(normalize)[i];

        for (int i = 0; i < k; ++i) {
            int currentTop = Utils.maxIndex(currentFeatureScores);
            topK[i] = currentTop;
            currentFeatureScores[currentTop] = -1;
        }

        return topK;
    }

    public static class Node extends AbstractMOAObject {

        private static final long serialVersionUID = 1L;

        protected DoubleVector observedClassDistribution;

        protected AutoExpandVector<AttributeClassObserver> attributeObservers = new AutoExpandVector<AttributeClassObserver>();

        protected static int NODE_GEN;
        protected final int nodeID;

        protected double impurity = 0.0;

        public double getImpurity() {
            return impurity;
        }

        public Node() {
            nodeID = 0;
        }

        public Node(double[] classObservations) {
            this.observedClassDistribution = new DoubleVector(classObservations);
            this.nodeID = Node.NODE_GEN++;
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

        public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent,
                                              int parentBranch, TreeSet<Integer> usedFeatures) {
            return new FoundNode(this, parent, parentBranch);
        }

        public double[] getObservedClassDistribution() {
            return this.observedClassDistribution.getArrayCopy();
        }

        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            return this.observedClassDistribution.getArrayCopy();
        }

        public boolean observedClassDistributionIsPure() {
            return this.observedClassDistribution.numNonZeroEntries() < 2;
        }

        public void describeSubtree(HoeffdingTree ht, StringBuilder out,
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

        public void getNodeDot(HoeffdingTree ht, StringBuilder output, Instance instance) {
            output.append(this.nodeID);
            output.append(" [label=\"Node");
            output.append("\"]\n");
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {
            describeSubtree(null, sb, indent);
        }
    }

    public static class SplitNode extends Node {

        private static final long serialVersionUID = 1L;

        protected InstanceConditionalTest splitTest;

        protected AutoExpandVector<Node> children; // = new AutoExpandVector<Node>();

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

        public InstanceConditionalTest getSplitTest() {
            return splitTest;
        }

        public SplitNode(double[] classObservations) {
            super(classObservations);
        }

        public SplitNode(InstanceConditionalTest splitTest,
                         double[] classObservations, double impurity, int size) {
            super(classObservations);
            this.splitTest = splitTest;
            this.children = new AutoExpandVector<Node>(size);
            this.impurity = impurity;
        }

        public SplitNode(InstanceConditionalTest splitTest,
                         double[] classObservations, double impurity) {
            super(classObservations);
            this.splitTest = splitTest;
            this.children = new AutoExpandVector<Node>();
            this.impurity = impurity;
        }


        public int numChildren() {
            return this.children.size();
        }

        public void setChild(int index, Node child) {
            InstanceConditionalTest splitTest = this.splitTest;
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

        @Override
        public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent,
                                              int parentBranch, TreeSet<Integer> usedFeatures) {
            if(usedFeatures != null) usedFeatures.add(this.splitTest.getAttsTestDependsOn()[0]);
            int childIndex = instanceChildIndex(inst);
            if (childIndex >= 0) {
                Node child = getChild(childIndex);
                if (child != null) {
                    return child.filterInstanceToLeaf(inst, this, childIndex, usedFeatures);
                }
                return new FoundNode(null, this, childIndex);
            }
            return new FoundNode(this, parent, parentBranch);
        }

        @Override
        public void describeSubtree(HoeffdingTree ht, StringBuilder out,
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

        @Override
        public void getNodeDot(HoeffdingTree ht, StringBuilder output, Instance instance) {
            output.append(this.nodeID);
            output.append(" [label=\"attribute = ");
            output.append(instance.attribute(this.splitTest.getAttsTestDependsOn()[0]).name());
            output.append("(index = ");
            output.append(this.splitTest.getAttsTestDependsOn()[0]);
            output.append(")");


            StringBuilder childrenOutput = new StringBuilder();
            int childrenNumInstances = 0;

            for (int branch = 0; branch < numChildren(); branch++) {
                Node child = getChild(branch);

                if (child != null) {
//                    output.append(this.splitTest.describeConditionForBranch(branch,
//                            ht.getModelContext()));
                    int childNumInstances = (int) Utils.sum(child.getObservedClassDistribution());
                    childrenNumInstances += childNumInstances;
//                    child.describeSubtree(ht, out, indent + 2);
//                    output.append();
                    child.getNodeDot(ht, childrenOutput, instance);
                    childrenOutput.append(this.nodeID);
                    childrenOutput.append(" -> ");
                    childrenOutput.append(child.nodeID);
                    childrenOutput.append(" [label = \"");
                    childrenOutput.append(this.splitTest.describeBranchValue(branch, ht.getModelContext()));
                    childrenOutput.append("\"]");
                    childrenOutput.append(";\n");
                }
            }

            output.append("\\nimpurity_value = ");
            output.append(this.getImpurity());
            output.append("\\nsamples_child_sum = ");
            output.append(childrenNumInstances);
            output.append("\\nsamples_at_split = ");
            output.append((int) Utils.sum(this.getObservedClassDistribution()));
            output.append("\\nclass_distribution = {");
            for (int i = 0; i < this.observedClassDistribution.numValues(); ++i) {
                output.append(instance.classAttribute().value(i));
                output.append(" = ");
                output.append((int) this.observedClassDistribution.getValue(i));
                if (i != this.observedClassDistribution.numValues() - 1)
                    output.append(", ");
            }
            output.append('}');

            output.append("\\nMDI = ");
            output.append(ht.calcNodeDecreaseImpurity(this));

            output.append("\"];\n");
            output.append(childrenOutput);
//            return output.toString();
        }
    }

    public static abstract class LearningNode extends Node {

        private static final long serialVersionUID = 1L;

        public LearningNode() {
        }

        public LearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        public abstract void learnFromInstance(Instance inst, HoeffdingTree ht);

        @Override
        public void getNodeDot(HoeffdingTree ht, StringBuilder output, Instance instance) {
            output.append("\t");
            output.append(this.nodeID);
            output.append(" [label=\"");
//            output.append("leaf_type = ");
//            String className = this.getClass().getName();
//            output.append(className.substring(className.indexOf("$")+1));
            output.append("\\nimpurity_value = ");
            output.append(this.getImpurity());
            output.append("\\nsamples = ");
            output.append((int) Utils.sum(this.getObservedClassDistribution()));
            output.append("\\nclass_distribution = {");
            for (int i = 0; i < this.observedClassDistribution.numValues(); ++i) {
                output.append(instance.classAttribute().value(i));
                output.append(" = ");
                output.append((int) this.observedClassDistribution.getValue(i));
                if (i != this.observedClassDistribution.numValues() - 1)
                    output.append(", ");
            }
            output.append('}');
            output.append("\"];\n");
        }
    }

    public static class InactiveLearningNode extends LearningNode {

        private static final long serialVersionUID = 1L;

        public InactiveLearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());
        }
    }

    public static class ActiveLearningNode extends LearningNode {

        private static final long serialVersionUID = 1L;

        protected double weightSeenAtLastSplitEvaluation;

        protected boolean isInitialized;

        public ActiveLearningNode() {
        }

        public ActiveLearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
            this.weightSeenAtLastSplitEvaluation = getWeightSeen();
            this.isInitialized = false;
        }

        @Override
        public int calcByteSize() {
            return super.calcByteSize()
                    + (int) (SizeOf.fullSizeOf(this.attributeObservers));
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
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

        public AttributeSplitSuggestion[] getBestSplitSuggestions(
                SplitCriterion criterion, HoeffdingTree ht) {
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
//                    System.out.println("[" + i +"] \t pre = " + Arrays.toString(preSplitDist));
                    AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
                            preSplitDist, i, ht.binarySplitsOption.isSet());
                    if (bestSuggestion != null) {
                        bestSuggestions.add(bestSuggestion);
                    }
                }
            }
            return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
        }

        public void disableAttribute(int attIndex) {
            this.attributeObservers.set(attIndex,
                    new NullAttributeClassObserver());
        }
    }

    public Node treeRoot;

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
        if (this.leafpredictionOption.getChosenIndex() > 0) {
            this.removePoorAttsOption = null;
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        // if(this.trainingWeightSeenByModel % 500 == 0) System.out.println(inst);
        if (this.treeRoot == null) {
            this.treeRoot = newLearningNode();
            this.activeLeafNodeCount = 1;
        }

//        int gtRel[] = inst.dataset().getIndicesRelevants();
//        int gtIrrel[] = inst.dataset().getIndicesIrrelevants();

//        TreeSet<Integer> usedFeatures = new TreeSet<>();
        FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1, null);
        Node leafNode = foundNode.node;

//        int used[] = new int[usedFeatures.size()];
//        int ix = 0;
//        for(int v : usedFeatures){
//            used[ix] = v;
//            ix++;
//        }
//        Integer used[] = (Integer[]) usedFeatures.toArray();
//        double nrfc = FeatureSelectionUtils.intersection(used, gtRel);
//        double nifc = FeatureSelectionUtils.intersection(used, gtIrrel);
//        double nrf = gtRel.length;
//        double nif = gtIrrel.length;
//
//        double w = (nrfc / nrf) - (nifc / nif);

        if (leafNode == null) {
            leafNode = newLearningNode();
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
            this.activeLeafNodeCount++;
        }
//        if(Utils.maxIndex(leafNode.getClassVotes(inst, this)) == inst.classValue()){
//            weightedHits += w;
//        }
        totalInstancesSeen++;

        if (leafNode instanceof LearningNode) {
            LearningNode learningNode = (LearningNode) leafNode;
            learningNode.learnFromInstance(inst, this);
            if (this.growthAllowed
                    && (learningNode instanceof ActiveLearningNode)) {
                ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
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
        try {
            if (outputTreeOption.isSet())
                if (this.trainingWeightSeenByModel % this.debugOutputTrainingWeightSeenOption.getValue() == 0)
                    outputTreeDotFile(this, inst);
        }catch (Exception e){

        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.treeRoot != null) {
            FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst,
                    null, -1, null);
            Node leafNode = foundNode.node;
            if (leafNode == null) {
                leafNode = foundNode.parent;
            }
            return leafNode.getClassVotes(inst, this);
        } else {
            int numClasses = inst.dataset().numClasses();
            return new double[numClasses];
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement m[] =  new Measurement[]{
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
                        this.byteSizeEstimateOverheadFraction),
                new Measurement("weightedAcc",
                        this.weightedHits / this.totalInstancesSeen)};
        this.weightedHits = this.totalInstancesSeen = 0.0;
        return m;
    }

    public void outputTreeDotFile(HoeffdingTree ht, Instance instance) throws IOException {
        FileOutputStream debugStream = null;
        SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
        String classSplitName = splitCriterion.getClass().getName().replace("moa.classifiers.core.splitcriteria.", "");

        File dumpFile = new File(this.debugDotFilePrefixOption.getValue() + "_" + classSplitName + "_" +
                (int) this.trainingWeightSeenByModel + "." + this.debugDotFilePrefixOption.getDefaultFileExtension());

        try {
            debugStream = new FileOutputStream(dumpFile, false);
        } catch (Exception ex) {
            throw new RuntimeException("Unable to open immediate result file: " + dumpFile, ex);
        }

        StringBuilder output = new StringBuilder();
        this.treeRoot.getNodeDot(ht, output, instance);
        String treeDot = output.toString();
        debugStream.write("digraph Tree {\nnode [shape=box] ;\n".getBytes());
        debugStream.write(("// Split criterion = " + classSplitName + "\n").getBytes());
        debugStream.write(treeDot.getBytes());
        debugStream.write("}".getBytes());
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

    //Procedure added for Hoeffding Adaptive Trees (ADWIN)
    // "impurity" added to store for each splitNode the impurity at the time of split
    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations, double impurity, int size) {
        return new SplitNode(splitTest, classObservations, impurity, size);
    }

    // "impurity" added to store for each splitNode the impurity at the time of split
    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations, double impurity) {
        return new SplitNode(splitTest, classObservations, impurity);
    }

    protected AttributeClassObserver newNominalClassObserver() {
        AttributeClassObserver nominalClassObserver = (AttributeClassObserver) getPreparedClassOption(this.nominalEstimatorOption);
        return (AttributeClassObserver) nominalClassObserver.copy();
    }

    protected AttributeClassObserver newNumericClassObserver() {
        AttributeClassObserver numericClassObserver = (AttributeClassObserver) getPreparedClassOption(this.numericEstimatorOption);
        return (AttributeClassObserver) numericClassObserver.copy();
    }

    protected void attemptToSplit(ActiveLearningNode node, SplitNode parent,
                                  int parentIndex) {
        if (!node.observedClassDistributionIsPure()) {
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
            Arrays.sort(bestSplitSuggestions);
//            System.out.println("----------------");
//            for(int i = 0; i < bestSplitSuggestions.length; i++){
//                if(bestSplitSuggestions[i].splitTest != null) {
//                    System.out.println(bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0] + "\t" +
//                            bestSplitSuggestions[i].merit);
//                }
//            }
            boolean shouldSplit = false;
            if (bestSplitSuggestions.length < 2) {
                shouldSplit = bestSplitSuggestions.length > 0;
            } else {
                double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                        this.splitConfidenceOption.getValue(), node.getWeightSeen());
                AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
                if (useMinMeritForSplitOption.isSet() &&
                        bestSuggestion.merit > minMeritForSplitOption.getValue() &&
                        (bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)){
                    shouldSplit = true;
                }else if (!useMinMeritForSplitOption.isSet()
                        && ((bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)
                                || (hoeffdingBound < this.tieThresholdOption.getValue()))) {
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
                    // System.out.println("@ " + this.trainingWeightSeenByModel + "\tCHOSEN = " + splitDecision.splitTest.getAttsTestDependsOn()[0] + "\t merit = " + splitDecision.merit);
                    SplitNode newSplit = newSplitNode(splitDecision.splitTest,
                            node.getObservedClassDistribution(), splitDecision.merit);
                    for (int i = 0; i < splitDecision.numSplits(); i++) {
                        Node newChild = null;
                        if (doNotUseParentStatisticsOption.isSet()) {
                            newChild = newLearningNode();
                        } else {
                            newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));
                        }
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

    public void enforceTrackerLimit() {
        if ((this.inactiveLeafNodeCount > 0)
                || ((this.activeLeafNodeCount * this.activeLeafByteSizeEstimate + this.inactiveLeafNodeCount
                * this.inactiveLeafByteSizeEstimate)
                * this.byteSizeEstimateOverheadFraction > this.maxByteSizeOption.getValue())) {
            if (this.stopMemManagementOption.isSet()) {
                this.growthAllowed = false;
                return;
            }
            FoundNode[] learningNodes = findLearningNodes();
            Arrays.sort(learningNodes, new Comparator<FoundNode>() {

                @Override
                public int compare(FoundNode fn1, FoundNode fn2) {
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
                if (learningNodes[i].node instanceof ActiveLearningNode) {
                    deactivateLearningNode(
                            (ActiveLearningNode) learningNodes[i].node,
                            learningNodes[i].parent,
                            learningNodes[i].parentBranch);
                }
            }
            for (int i = cutoff; i < learningNodes.length; i++) {
                if (learningNodes[i].node instanceof InactiveLearningNode) {
                    activateLearningNode(
                            (InactiveLearningNode) learningNodes[i].node,
                            learningNodes[i].parent,
                            learningNodes[i].parentBranch);
                }
            }
        }
    }

    public void estimateModelByteSizes() {
        FoundNode[] learningNodes = findLearningNodes();
        long totalActiveSize = 0;
        long totalInactiveSize = 0;
        for (FoundNode foundNode : learningNodes) {
            if (foundNode.node instanceof ActiveLearningNode) {
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
        FoundNode[] learningNodes = findLearningNodes();
        for (int i = 0; i < learningNodes.length; i++) {
            if (learningNodes[i].node instanceof ActiveLearningNode) {
                deactivateLearningNode(
                        (ActiveLearningNode) learningNodes[i].node,
                        learningNodes[i].parent, learningNodes[i].parentBranch);
            }
        }
    }

    protected void deactivateLearningNode(ActiveLearningNode toDeactivate,
                                          SplitNode parent,
                                          int parentBranch) {
        Node newLeaf = new InactiveLearningNode(toDeactivate.getObservedClassDistribution());
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }
        this.activeLeafNodeCount--;
        this.inactiveLeafNodeCount++;
    }

    protected void activateLearningNode(InactiveLearningNode toActivate,
                                        SplitNode parent, int parentBranch) {
        Node newLeaf = newLearningNode(toActivate.getObservedClassDistribution());
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }
        this.activeLeafNodeCount++;
        this.inactiveLeafNodeCount--;
    }

    protected FoundNode[] findLearningNodes() {
        List<FoundNode> foundList = new LinkedList<FoundNode>();
        findLearningNodes(this.treeRoot, null, -1, foundList);
        return foundList.toArray(new FoundNode[foundList.size()]);
    }

    protected void findLearningNodes(Node node, SplitNode parent,
                                     int parentBranch, List<FoundNode> found) {
        if (node != null) {
            if (node instanceof LearningNode) {
                found.add(new FoundNode(node, parent, parentBranch));
            }
            if (node instanceof SplitNode) {
                SplitNode splitNode = (SplitNode) node;
                for (int i = 0; i < splitNode.numChildren(); i++) {
                    findLearningNodes(splitNode.getChild(i), splitNode, i,
                            found);
                }
            }
        }
    }

    public MultiChoiceOption leafpredictionOption = new MultiChoiceOption(
            "leafprediction", 'l', "Leaf prediction to use.", new String[]{
            "MC", "NB", "NBAdaptive"}, new String[]{
            "Majority class",
            "Naive Bayes",
            "Naive Bayes Adaptive"}, 2);

    public IntOption nbThresholdOption = new IntOption(
            "nbThreshold",
            'q',
            "The number of instances a leaf should observe before permitting Naive Bayes.",
            0, 0, Integer.MAX_VALUE);

    public static class LearningNodeNB extends ActiveLearningNode {

        private static final long serialVersionUID = 1L;

        public LearningNodeNB() {
        }

        public LearningNodeNB(double[] initialClassObservations) {
            super(initialClassObservations);
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

    public static class LearningNodeNBAdaptive extends LearningNodeNB {

        private static final long serialVersionUID = 1L;

        protected double mcCorrectWeight = 0.0;

        protected double nbCorrectWeight = 0.0;

        public LearningNodeNBAdaptive() {
        }

        public LearningNodeNBAdaptive(double[] initialClassObservations) {
            super(initialClassObservations);
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

    protected LearningNode newLearningNode() {
        return newLearningNode(new double[0]);
    }

    protected LearningNode newLearningNode(double[] initialClassObservations) {
        LearningNode ret;
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        if (predictionOption == 0) { //MC
            ret = new ActiveLearningNode(initialClassObservations);
        } else if (predictionOption == 1) { //NB
            ret = new LearningNodeNB(initialClassObservations);
        } else { //NBAdaptive
            ret = new LearningNodeNBAdaptive(initialClassObservations);
        }
        return ret;
    }


}
