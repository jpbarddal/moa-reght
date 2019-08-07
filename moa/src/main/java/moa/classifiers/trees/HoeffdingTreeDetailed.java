package moa.classifiers.trees;

import moa.classifiers.MultiClassClassifier;
import moa.options.ClassOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import java.util.ArrayList;
import java.util.TreeSet;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
// import moa.classifiers.featureselection.FeatureSelection;
import moa.classifiers.trees.HoeffdingTree.Node;
import moa.classifiers.trees.HoeffdingTree.SplitNode;
import moa.core.Measurement;

/**
 *
 * @author Jean Paul Barddal
 */
public class HoeffdingTreeDetailed extends AbstractClassifier implements MultiClassClassifier{ //, FeatureSelection {

    public ClassOption treeLearnerOption
            = new ClassOption("treeLearner", 't', "",
                    Classifier.class, "trees.HoeffdingTree");

    Classifier tree;

    Instances header;

    @Override
    public double[] getVotesForInstance(Instance instnc) {
        if (header == null) {
            header = instnc.dataset();
        }
        return tree.getVotesForInstance(instnc);
    }

    @Override
    public void resetLearningImpl() {
        this.header = null;
        this.tree = (Classifier) getPreparedClassOption(this.treeLearnerOption);
        this.tree.resetLearning();
    }

    @Override
    public void trainOnInstanceImpl(Instance instnc) {
        if (header == null) {
            header = instnc.dataset();
        }
        this.tree.trainOnInstance(instnc);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        ArrayList<Measurement> measures = new ArrayList<>();
        for (Measurement m : tree.getModelMeasurements()) {
            measures.add(m);
        }
        Measurement measuresSelection[] = new Measurement[]{
            new Measurement("# features selected",
            getSelectedFeatures().size()),
            new Measurement("pct of features selected",
            ((float) getSelectedFeatures().size()) / (header.numAttributes() - 1))};
        for (Measurement m : measuresSelection) {
            measures.add(m);
        }
        Measurement measuresFinal[] = new Measurement[measures.size()];
        for (int i = 0; i < measures.size(); i++) {
            measuresFinal[i] = measures.get(i);
        }
        return measuresFinal;
    }

    @Override
    public void getModelDescription(StringBuilder sb, int i) {

    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

//    @Override
    public TreeSet<String> getSelectedFeatures() {
        ArrayList<Node> nodes = new ArrayList<Node>();
        Node node = ((HoeffdingTree) tree).treeRoot;
        nodes.addAll(branch(node));

        // for each of the nodes, let's check
        // the feature used and let's append 
        // its name to the final list
        TreeSet<String> selected = new TreeSet<>();
        for (Node n : nodes) {
            int indexAtt = ((SplitNode) n).splitTest.getAttsTestDependsOn()[0];
            if(header == null){
                int debug = 1;
            }
            selected.add(header.attribute(indexAtt).name());
        }
        return selected;
    }

    // Performs a DFS to get all the nodes we have in the tree
    private ArrayList<Node> branch(Node node) {
        ArrayList<Node> ret = new ArrayList<Node>();
        if (node != null && !node.isLeaf()) {
            ret.add(node);
            if (node instanceof HoeffdingTree.SplitNode) {
                for (Node child : ((SplitNode) node).children) {
                    ret.addAll(branch(child));
                }
            }
        }
        return ret;
    }

}
