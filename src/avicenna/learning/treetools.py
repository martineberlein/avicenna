"""This file bundles several functions which are helpful when working with decision trees."""
import pandas
import numpy
import matplotlib.patches as mpp


from ..data import OracleResult


def all_path(clf, node=0):
    """Iterate over all path in a decision tree. Path will be represented as
    a list of integers, each integer is the index of a node in the clf.tree_ structure.
    """
    left = clf.tree_.children_left[node]
    right = clf.tree_.children_right[node]

    if left == right:
        yield [node]
    else:
        for path in all_path(clf, left):
            yield [node] + path
        for path in all_path(clf, right):
            yield [node] + path


def path_samples(clf, path):
    """Returns the number of samples for this path."""
    return clf.tree_.n_node_samples[path[-1]]


def generic_feature_names(clf):
    """Gives a list of feature names of the form f1, f2, ..."""
    return ["f{}".format(f) for f in range(0, clf.tree_.n_features)]


def box(clf, path, data=None, feature_names=None):
    """For a decision tree classifier clf and a path (as returned, e.g. by all_path),
    this method gives a pandas DataFrame with the min and max of each feature value on the given path.
    """

    if feature_names is None:
        feature_names = generic_feature_names(clf)
    check_for_duplicates(feature_names)
    if data is None:
        bounds = (
            pandas.DataFrame(
                [
                    {"feature": c, "min": -numpy.inf, "max": numpy.inf}
                    for c in feature_names
                ],
                columns=["feature", "min", "max"],
            )
            .set_index(["feature"])
            .transpose()
        )
    else:
        bounds = (
            pandas.DataFrame(
                [
                    {"feature": c, "min": data[c].min(), "max": data[c].max()}
                    for c in feature_names
                ],
                columns=["feature", "min", "max"],
            )
            .set_index(["feature"])
            .transpose()
        )

    for pos in range(0, len(path) - 1):
        node = path[pos]
        child = path[pos + 1]
        feature = feature_names[clf.tree_.feature[node]]
        threshold = clf.tree_.threshold[node]

        if child == clf.tree_.children_left[node]:
            bounds.at["max", feature] = threshold
        else:
            bounds.at["min", feature] = threshold
    return bounds


def rectangles(clf, colormap, data, feature_names=None):
    """yields matplotlib.patches rectangle objects. Each object represents a leaf of the tree."""
    if feature_names is None:
        feature_names = ["in_x", "in_y"]
    if 2 != len(feature_names):
        raise AssertionError(
            "Rectangles can only be generated if there are at most 2 features."
        )

    x_feature = feature_names[0]
    y_feature = feature_names[1]

    for path in all_path(clf):
        b = box(clf, path, data=data, feature_names=feature_names)
        p = prediction_for_path(clf, path)
        c = colormap[p]
        rect = mpp.Rectangle(
            (b[x_feature]["min"], b[y_feature]["min"]),  # coordinates
            b[x_feature]["max"] - b[x_feature]["min"],  # width
            b[y_feature]["max"] - b[y_feature]["min"],  # height
            alpha=0.2,
            facecolor=c,
            edgecolor="k",
        )
        yield rect


def prediction_for_path(clf, path, classes=None) -> OracleResult:
    if classes is None:
        classes = [True, False]
    last_value = clf.tree_.value[path[-1]][0]
    p_class = numpy.argmax(last_value)
    cls = clf.classes_[p_class]
    if isinstance(cls, float):
        cls = classes[p_class]
    if cls == False:
        return OracleResult.PASSING
    else:
        return OracleResult.FAILING
    # return OracleResult(cls)


def rule(clf, path, feature_names, class_names=None):
    """Creates a rule from one path in the decision tree."""
    bounds = box(clf, path, feature_names=feature_names)
    prediction = prediction_for_path(clf, path)
    if class_names is not None:
        prediction = class_names[prediction]

    feature_rules = []
    for fname in feature_names:
        min_ = bounds[fname]["min"]
        max_ = bounds[fname]["max"]

        if numpy.isinf(min_) and numpy.isinf(max_):
            pass  # no rule if both are unbound
        elif numpy.isinf(min_):
            feature_rules.append("{} <= {:.4f}".format(fname, max_))
        elif numpy.isinf(max_):
            feature_rules.append("{} > {:.4f}".format(fname, min_))
        else:
            feature_rules.append("{} in {:.4f} to {:.4f}".format(fname, min_, max_))

    return (
        " AND ".join(feature_rules),
        prediction,
        clf.tree_.impurity[path[-1]],
        clf.tree_.n_node_samples[path[-1]],
    )


def rules(clf, class_names=None, feature_names=None):
    """Formats Decision trees in a rule-like representation."""

    if feature_names is None:
        feature_names = generic_feature_names(clf)

    samples = clf.tree_.n_node_samples[0]
    return "\n".join(
        [
            "IF {2} THEN PREDICT '{3}' ({0}: {4:.4f}, support: {5} / {1})".format(
                clf.criterion,
                samples,
                *rule(clf, path, feature_names, class_names=class_names),
            )
            for path in all_path(clf)
        ]
    )


def grouped_rules(clf, class_names=None, feature_names=None):
    """Formats decision trees in a rule-like representation, grouped by class."""

    if feature_names is None:
        feature_names = generic_feature_names(clf)

    rules = {}
    for path in all_path(clf):
        rulestr, clz, impurity, support = rule(
            clf, path, class_names=class_names, feature_names=feature_names
        )
        if clz not in rules:
            rules[clz] = []
        rules[clz].append((rulestr, impurity, support))

    res = ""
    samples = clf.tree_.n_node_samples[0]
    for clz in rules:
        rulelist = rules[clz]
        res = res + "\n{}:\n\t".format(clz)
        rl = [
            "{} ({}: {:.4f}, support: {}/{})".format(
                r, clf.criterion, impurity, support, samples
            )
            for r, impurity, support in rulelist
        ]
        res = res + "\n\tor ".join(rl)
    return res.lstrip()


def check_for_duplicates(names):
    seen = set()
    for name in names:
        if name in seen:
            raise AssertionError("Duplicate name: {}".format(name))
        seen.add(name)


def is_leaf(clf, node: int) -> bool:
    """returns true if the given node is a leaf."""
    return clf.tree_.children_left[node] == clf.tree_.children_right[node]


def leaf_label(clf, node: int) -> int:
    """returns the index of the class at this node. The node must be a leaf."""
    assert is_leaf(clf, node)
    occs = clf.tree_.value[node][0]
    idx = 0
    maxi = occs[idx]
    for i, o in zip(range(0, len(occs)), occs):
        if maxi < o:
            maxi = o
            idx = i
    return idx


# def remove_infeasible(clf, features: List[Feature]):
#     for node in range(0, clf.tree_.node_count):
#         if not is_leaf(clf, node):
#             feature = features[clf.tree_.feature[node]]
#             threshold = clf.tree_.threshold[node]
#             if not feature.is_feasible(threshold):
#                 clf.tree_.feature[node] = find_existence_index(features, feature)
#                 clf.tree_.threshold[node] = 0.5
#     return clf


def iterate_nodes(clf):
    stack = [0]
    while 0 != len(stack):
        node = stack.pop()
        yield node
        if not is_leaf(clf, node):
            stack.append(clf.tree_.children_left[node])
            stack.append(clf.tree_.children_right[node])


def count_nodes(clf):
    return len(list(iterate_nodes(clf)))


def count_leaves(clf):
    return len([n for n in iterate_nodes(clf) if is_leaf(clf, n)])


def list_features(clf):
    return [clf.tree_.feature[node] for node in iterate_nodes(clf)]


def remove_unequal_decisions(clf):
    """
    This method rewrites a decision tree classifier to remove nodes where the same
    decision is taken on both sides.

    :param clf: a decision tree classifier
    :return: the same classifier, rewritten
    """
    changed = True
    while changed:
        changed = False
        for node in range(0, clf.tree_.node_count):
            if not is_leaf(clf, node) and (
                is_leaf(clf, clf.tree_.children_left[node])
                and is_leaf(clf, clf.tree_.children_right[node])
            ):
                # both children of this node are leaves
                left_label = leaf_label(clf, clf.tree_.children_left[node])
                right_label = leaf_label(clf, clf.tree_.children_right[node])
                if left_label == right_label:
                    clf.tree_.children_left[node] = -1
                    clf.tree_.children_right[node] = -1
                    clf.tree_.feature[node] = -2
                    changed = True
                    assert left_label == leaf_label(clf, node)
    return clf
