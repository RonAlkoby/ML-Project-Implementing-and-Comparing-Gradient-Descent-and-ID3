import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from graphviz import Digraph


def entropy_calculation(data):
    # Calculates the entropy of the dataset based on the labels (last column).
    # Entropy measures the impurity or uncertainty of the data.
    labels = data[:, -1]
    unique, counts = np.unique(labels, return_counts=True)
    P = counts / len(labels)
    return -0.5 * np.sum(P * np.log2(P + 1e-9))  # Add small value to avoid log(0)


def information_gain(data, feature_index):
    # Calculates the information gain of splitting the dataset on a given feature.
    # Information gain measures how much a feature reduces the entropy of the dataset.
    total_entropy = entropy_calculation(data)
    unique_values = np.unique(data[:, feature_index])
    best_gain = -1
    best_threshold = None

    for threshold in unique_values:
        # Split the data into left and right parts based on the threshold.
        left_split = data[data[:, feature_index] <= threshold]
        right_split = data[data[:, feature_index] > threshold]
        # Calculate the weighted entropy of the split.
        weighted_entropy = (
                (len(left_split) / len(data)) * entropy_calculation(left_split) +
                (len(right_split) / len(data)) * entropy_calculation(right_split)
        )
        # Calculate the information gain for this split.
        gain = total_entropy - weighted_entropy

        # Update the best gain and threshold if this split is better.
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    return best_gain, best_threshold


def build_tree(data, features):
    # Recursively builds a decision tree using the data and available features.
    labels = data[:, -1]
    if len(np.unique(labels)) == 1:
        # If all labels are the same, return the label (leaf node).
        return labels[0]

    if len(features) == 0:
        # If there are no features left, return the majority label.
        return np.bincount(labels.astype(int)).argmax()

    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature in features:
        # Find the best feature and threshold to split the data.
        gain, threshold = information_gain(data, feature)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_threshold = threshold

    if best_gain == -1:
        # If no good split is found, return the majority label.
        return np.bincount(labels.astype(int)).argmax()

    # Create a dictionary to represent the tree node.
    tree = {best_feature: {"threshold": best_threshold, "left": None, "right": None}}
    # Split the data based on the best feature and threshold.
    left_split = data[data[:, best_feature] <= best_threshold]
    right_split = data[data[:, best_feature] > best_threshold]

    # Remove the used feature from the list of features.
    remaining_features = features.copy()
    remaining_features.remove(best_feature)

    # Recursively build the left and right subtrees.
    tree[best_feature]["left"] = build_tree(left_split, remaining_features)
    tree[best_feature]["right"] = build_tree(right_split, remaining_features)

    return tree


def visualize_tree(tree, graph=None, parent_name=None, edge_label="", accuracy=None):
    # Visualizes the decision tree using the Graphviz library.
    if graph is None:
        graph = Digraph(format="png")
        graph.attr(dpi='300')

        if accuracy is not None:
            # Add a title to the graph showing the accuracy.
            graph.attr(label=f"Decision Tree\nAccuracy: {accuracy:.2f}", labelloc="t", fontsize="20")

    if not isinstance(tree, dict):  
        # If the tree is a leaf node, add a leaf node to the graph.
        node_name = f"Leaf_{id(tree)}"
        graph.node(node_name, label=f"Class: {int(tree)}", shape="ellipse", style="filled", color="lightgrey")
        if parent_name is not None:
            graph.edge(parent_name, node_name, label=edge_label)
        return graph

    # Add a decision node to the graph.
    feature = next(iter(tree))
    threshold = tree[feature]["threshold"]
    node_name = f"Node_{feature}_{id(tree)}"
    graph.node(node_name, label=f"Feature {feature}\nThreshold: {threshold:.2f}", shape="box")

    if parent_name is not None:
        # Connect the current node to its parent.
        graph.edge(parent_name, node_name, label=edge_label)

    # Recursively visualize the left and right branches.
    visualize_tree(tree[feature]["left"], graph, node_name, edge_label="<= Threshold")
    visualize_tree(tree[feature]["right"], graph, node_name, edge_label="> Threshold")

    return graph


def decision_path(tree, sample):
    # Predicts the label of a single sample by following the decision tree path.
    if not isinstance(tree, dict):
        # If the tree is a leaf node, return its value (class label).
        return tree
    feature = next(iter(tree))
    threshold = tree[feature]["threshold"]
    # Decide which branch to follow based on the sample's feature value.
    if sample[feature] <= threshold:
        return decision_path(tree[feature]["left"], sample)
    else:
        return decision_path(tree[feature]["right"], sample)


# Load the breast cancer dataset.
dataset = load_breast_cancer()
data = np.column_stack((dataset.data, dataset.target))

# Split the data into training and testing sets.
train_data, test_data = train_test_split(data, test_size=0.2, random_state=20)

# Create a list of features to use for building the tree.
features = list(range(train_data.shape[1] - 1))

# Build the decision tree using the training data.
tree = build_tree(train_data, features)

# Make predictions on the test data using the tree.
predictions = [decision_path(tree, row) for row in test_data]

# Calculate the accuracy of the model.
accuracy = accuracy_score(test_data[:, -1], predictions)

# Visualize the decision tree and save it as an image.
graph = visualize_tree(tree, accuracy=accuracy)
graph.render("decision_tree", view=True)

# Print the accuracy of the model.
print(f"Accuracy: {accuracy:.2f}")
