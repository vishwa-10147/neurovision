/** Algorithm metadata and pure helpers for the Classical ML Lab (unit-tested). */
import type { Algorithm, ClassicDataset, ClassicRequest } from '../../api/classical'

export type Param = 'c' | 'gamma' | 'neighbors' | 'max_depth' | 'estimators' | 'learning_rate' | 'clusters' | 'epochs'

export interface AlgorithmInfo {
  label: string
  family: 'Linear' | 'Kernel' | 'Instance-based' | 'Trees' | 'Unsupervised'
  params: Param[]
  datasets: ClassicDataset[]
  /** What the animation steps through. */
  animates: string | null
  about: string
}

const TWO_D: ClassicDataset[] = ['moons', 'circles', 'spiral', 'blobs']

export const ALGORITHM_INFO: Record<Algorithm, AlgorithmInfo> = {
  logistic_regression: {
    label: 'Logistic regression',
    family: 'Linear',
    params: ['c', 'epochs'],
    datasets: [...TWO_D, 'digits'],
    animates: 'training epochs',
    about:
      'A weighted sum of the features passed through a sigmoid (softmax for more classes). Its boundaries are straight lines, so it cannot separate moons or circles perfectly. Shading shows the predicted probability.',
  },
  linear_svm: {
    label: 'Linear SVM',
    family: 'Linear',
    params: ['c', 'epochs'],
    datasets: TWO_D,
    animates: 'training epochs',
    about:
      'Finds the straight boundary with the widest margin between classes (hinge loss). Larger C fits the training points more tightly.',
  },
  rbf_svm: {
    label: 'SVM, RBF kernel',
    family: 'Kernel',
    params: ['c', 'gamma'],
    datasets: TWO_D,
    animates: null,
    about:
      'A support vector machine with a Gaussian kernel: curved boundaries built from the support vectors (ringed). Larger gamma makes each support vector’s influence more local; larger C fits more tightly.',
  },
  knn: {
    label: 'k-nearest neighbors',
    family: 'Instance-based',
    params: ['neighbors'],
    datasets: [...TWO_D, 'digits'],
    animates: 'k from 1 upward',
    about:
      'No training: a point takes the majority class of its k closest training points. Small k follows every point (jagged, overfits); large k smooths the boundary.',
  },
  decision_tree: {
    label: 'Decision tree',
    family: 'Trees',
    params: ['max_depth'],
    datasets: TWO_D,
    animates: 'tree depth',
    about:
      'Splits the plane with one axis-aligned threshold at a time, choosing the split that most reduces impurity (Gini). Each extra level adds rectangles. The tree is drawn below.',
  },
  random_forest: {
    label: 'Random forest',
    family: 'Trees',
    params: ['estimators', 'max_depth'],
    datasets: [...TWO_D, 'digits'],
    animates: 'number of trees',
    about:
      'Many decision trees, each trained on a bootstrap sample with random feature choices, averaged. Watch the jagged single-tree boundary smooth out as trees are added.',
  },
  gradient_boosting: {
    label: 'Gradient boosting',
    family: 'Trees',
    params: ['estimators', 'learning_rate', 'max_depth'],
    datasets: TWO_D,
    animates: 'boosting stages',
    about:
      'Adds small trees one after another, each correcting the errors of the ensemble so far, scaled by the learning rate.',
  },
  kmeans: {
    label: 'k-Means clustering',
    family: 'Unsupervised',
    params: ['clusters', 'epochs'],
    datasets: TWO_D,
    animates: 'iterations',
    about:
      'Unsupervised: no labels. Alternates assigning each point to its nearest centroid (✕) and moving each centroid to the mean of its points, until nothing changes. Regions show which centroid is nearest.',
  },
  pca: {
    label: 'PCA',
    family: 'Unsupervised',
    params: [],
    datasets: [...TWO_D, 'digits'],
    animates: null,
    about:
      'Principal component analysis finds the directions of greatest variance. On 2D data the arrows show them; on the 64-pixel digits it compresses every image to 2 numbers, and similar digits land together.',
  },
}

export const PARAM_INFO: Record<Param, { label: string; min: number; max: number; step: number }> = {
  c: { label: 'C (inverse regularization)', min: 0.01, max: 100, step: 0.01 },
  gamma: { label: 'Gamma (kernel width)', min: 0.01, max: 50, step: 0.01 },
  neighbors: { label: 'Neighbors k', min: 1, max: 50, step: 1 },
  max_depth: { label: 'Max depth', min: 1, max: 12, step: 1 },
  estimators: { label: 'Trees', min: 1, max: 200, step: 1 },
  learning_rate: { label: 'Learning rate', min: 0.01, max: 1, step: 0.01 },
  clusters: { label: 'Clusters k', min: 1, max: 10, step: 1 },
  epochs: { label: 'Epochs / iterations', min: 1, max: 200, step: 1 },
}

export const DEFAULT_REQUEST: ClassicRequest = {
  algorithm: 'decision_tree',
  dataset: 'moons',
  samples: 400,
  noise: 0.2,
  seed: 0,
  c: 1,
  gamma: 1,
  neighbors: 15,
  max_depth: 5,
  estimators: 50,
  learning_rate: 0.1,
  clusters: 3,
  epochs: 30,
}
