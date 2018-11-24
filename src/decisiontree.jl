__precompile__()

using Ensemblearn

export
    DecisionTree,
    fit!,
    predict,
    predict_probs,
    print_tree

# Based on https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/decision_tree.py

mutable struct DecisionNode
    feature_i::Union{Int64, Nothing}          # Index for the feature that is tested
    threshold::Union{Float64, Nothing}          # Threshold value for feature
    value#::Union{Float64, Nothing}                  # Value if the node is a leaf in the tree
    true_branch::Union{DecisionNode,Nothing}      # 'Left' subtree
    false_branch::Union{DecisionNode, Nothing}    # 'Right' subtree

    function DecisionNode(;feature_i=nothing::Union{Int64, Nothing}, threshold=nothing::Union{Float64, Nothing},
                          value=nothing, true_branch=nothing::Union{DecisionNode,Nothing}, false_branch=nothing::Union{DecisionNode, Nothing})
        new(feature_i, threshold, value, true_branch, false_branch)
    end
end

# Super class of ClassificationTree
mutable struct DecisionTree
    # Minimum n of samples to justify split
    min_samples_split::Int64
    # The minimum impurity to justify split
    min_impurity::Float64
    # The maximum depth to grow the tree to
    max_depth::Int64
    # Impurity method to use on information gain calculation
    impurity::String
    # Function to determine prediction of y at leaf
    #_leaf_value_calculation = None
    root  # Root node in dec. tree
    classes

    function DecisionTree(min_samples_split=2::Int64, min_impurity=1e-7::Float64, max_depth=10::Int64,
                          impurity="gini"::String, classes=nothing)
        # TODO: here assert impurity ?
        new(min_samples_split, min_impurity, max_depth, impurity, nothing, classes)
    end
end

impurity_methods = Dict("entropy" => entropy, "gini" => gini_index, "cerr" => classification_error)

function calculate_information_gain(y::Array{T,2}, y1::Array{T2,1}, y2::Array{T2,1}, impurity="gini"::String) where{T <: Real, T2 <: Real}  # TODO: y::Array{T,1}

    @assert haskey(impurity_methods, impurity)
    i = impurity_methods[impurity]

    # Calculate information gain
    p = length(y1) / length(y)
    info_gain = i(y) - p * i(y1) - (1 - p) * i(y2)

    return info_gain
end

function fill_dict!(d, ks, v=0)
    for k in ks
        if !haskey(d, k)
            d[k] = v;
        end
    end
end

# TODO: partial fit, by fingerprinting samples to add counts (it is necessary to fingerprint?)

function fit!(tree::DecisionTree, x::Array{T,2}, y::Array{T2,1}; current_depth=0::Int64) where {T <: Real, T2 <: Real}
    """ Recursive method which builds out the decision tree and splits x and respective y
    on the feature of x which (based on impurity) best separates the data"""

    largest_impurity = 0
    best_criteria = nothing    # Feature index and threshold
    best_sets = nothing        # Subsets of the data

    # Check if expansion of y is needed
    # TODO: is this necesary?
    if ndims(y) == 1
        y = reshape(y, length(y), 1)
    end

    # Add y as last column of x
    xy = hcat(x, y)

    if tree.classes == nothing
        tree.classes = unique(y)
    end

    n_samples, n_features = size(x)

    if (n_samples >= tree.min_samples_split) && (current_depth <= tree.max_depth)
        # Calculate the impurity for each feature
        for feature_i in 1:n_features
            # All values of feature_i
            feature_values = x[:, feature_i]
            unique_values = unique(feature_values)

            # Iterate through all unique values of feature column i and
            # calculate the impurity
            for threshold in unique_values
                # Divide X and y depending on if the feature value of X at index feature_i
                # meets the threshold

                # TODO: just get indexes to avoid mixing x and y types in a single array
                xy1, xy2 = divide_on_feature(xy, feature_i, threshold)

                if (size(xy1,1) > 0) && (size(xy2,1) > 0)
                    # Select the y-values of the two sets
                    y1 = xy1[:, end]
                    y2 = xy2[:, end]

                    # Calculate information gain
                    # TODO: impurity (general method) instead of information gain (particular)
                    impurity = calculate_information_gain(y, y1, y2, tree.impurity)

                    # If this threshold resulted in a higher information gain than previously
                    # recorded save the threshold value and the feature
                    # index
                    if impurity > largest_impurity
                        largest_impurity = impurity
                        best_criteria = Dict("feature_i"=> feature_i, "threshold"=> threshold)
                        best_sets = Dict(
                            "leftx"=> xy1[:, 1:end-1],   # X of left subtree
                            "lefty"=> y1,   # y of left subtree
                            "rightx"=> xy2[:, 1:end-1],  # X of right subtree
                            "righty"=> y2   # y of right subtree
                        )
                    end
                end
            end
        end
    end

    #tree.one_dim = size(y, 1) == 1
    #tree.loss = nothing

    if largest_impurity > tree.min_impurity
        # Build subtrees for the right and left branches
        true_branch = DecisionTree(tree.min_samples_split, tree.min_impurity, tree.max_depth, tree.impurity, tree.classes)
        false_branch = DecisionTree(tree.min_samples_split, tree.min_impurity, tree.max_depth, tree.impurity, tree.classes)

        fit!(true_branch, best_sets["leftx"], best_sets["lefty"], current_depth=current_depth + 1)
        fit!(false_branch, best_sets["rightx"], best_sets["righty"], current_depth=current_depth + 1)

        tree.root = DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"],
                                 true_branch=true_branch.root, false_branch=false_branch.root)
    else
        # We're at leaf => determine value
        leaf_value = count_distinct(y)
        fill_dict!(leaf_value, tree.classes)

        tree.root = DecisionNode(value=leaf_value)
    end
end

function get_ratios_dict(d)
    n = sum(values(d))
    return Dict([(x[1], x[2]/n) for x in d])
end


function likelihood(n, x, delta=1e-4)
    # TODO: test this!
    # From https://www.datascience.com/blog/introduction-to-bayesian-inference-learn-data-science-tutorials
    """
    likelihood function for a binomial distribution

    n: [int] the number of experiments
    x: [int] the number of successes
    theta: [float] the proposed probability of success
    """
    lk = theta ->  (factorial(n) / (factorial(x) * factorial(n - x))) * (theta ^ x) * ((1 - theta) ^ (n - x))

    probs = 0:delta:1

    likelihoods = map(lk, probs)

    return probs[argmax(likelihoods)]
end


function predict_value(model::DecisionTree, x::Array{T,1}, node=nothing::Union{DecisionNode, Nothing};
                       get_probs=false::Bool) where T <: Real
    """ Do a recursive search down the tree and make a prediction of the data sample by the
        value of the leaf that we end up at """

    if node == nothing
        node = model.root
    end

    # If we have a value (i.e we're at a leaf) => return value as the prediction
    if node.value != nothing
        if get_probs == true
            v = get_ratios_dict(node.value)
        else
            v = findmax(node.value)[2]
        end

        return v
    end

    # Choose the feature that we will test)
    feature_value = x[node.feature_i]

    # Determine if we will follow left or right branch
    branch = node.false_branch
    if feature_value isa Number
        if feature_value >= node.threshold
            branch = node.true_branch
        end
    elseif feature_value == node.threshold
        branch = node.true_branch
    end

    # Test subtree
    return predict_value(model, x, branch, get_probs=get_probs)
end

function predict(model::DecisionTree, x::Array{T,2}) where T <: Real
    """ Classify samples one by one and return the set of labels """
    pred = []

    for i in 1:size(x, 1)
        y  = predict_value(model, x[i,:], get_probs=false)
        push!(pred, y)
    end

    pred
end

function predict_probs(model::DecisionTree, x::Array{T,2}) where T <: Real
    """ Classify samples one by one and return the set of labels """
    pred = Dict{Any,Array{Float64,1}}()

    for i in 1:size(x, 1)
        y  = predict_value(model, x[i,:], get_probs=true)

        for kv in y
            if !haskey(pred, kv[1])
                pred[kv[1]] = []
            end
            push!(pred[kv[1]], kv[2])
        end
    end

    pred
end



function print_tree(tree::DecisionTree, node=nothing::Union{DecisionNode,Nothing}; indent=" "::String)
    """ Recursively print the decision tree """
    if node == nothing
        node = tree.root
    end

    # If we're at leaf => print the label
    if node.value != nothing
        println(node.value)
    # Go deeper down the tree
    else
        # Print test
        # TODO: print column name when trees are fitted with DFs
        println("Feature_$(node.feature_i):$(node.threshold)?")
        # Print the true scenario
        println("$(indent)T->")
        print_tree(tree, node.true_branch, indent=indent*indent)
        # Print the false scenario
        println("$(indent)F->")
        print_tree(tree, node.false_branch, indent=indent*indent)
    end
end
