/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.streams.filters;

import com.yahoo.labs.samoa.instances.Instance;

/**
 *
 * @author Heitor Murilo Gomes
 */
public interface FeatureScore {
    
    /**
     * Update the feature scoring method.
     * 
     * @param instance 
     */
    abstract boolean updateFeaturesScore(Instance instance);
    
    /**
     * Obtain the current scores for each feature. 
     * 
     * @return array containing the score for each feature
     */
    abstract public double[] getFeaturesScore(boolean normalize);
    
    /**
     * The output is a double array where values indicates the 
     * original feature index and the order of the array its 
     * ranking. The size of this array is expected to be less than 
     * the complete set of features. 
     * @param k
     * @param normalize
     * @return the k features with the highest scores. 
     */
    abstract public int[] getTopKFeatures(int k, boolean normalize);
}
