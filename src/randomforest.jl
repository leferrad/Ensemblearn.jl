__precompile__()

using Ensemblearn

# Based on https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/random_forest
# - https://machinelearningmastery.com/implement-random-forest-scratch-python/
# - https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249

export
    RandomForest,
    fit!,
    predict,
    predict_probs

# TODO: type?
mutable struct RandomForest
    n_trees::Int64
    max_depth::Int64  # TODO: any of None or Int
    max_features::Union{Int64, Nothing}
    min_samples_split::Int64
    min_gain::Float64
    #bootstrap::Bool
    #oob_score::Bool
    seed::Int64
    trees::Union{Array{DecisionTree, 1}, Nothing}
    feature_indices::Union{Array{Array{Int64,1},1}, Nothing}  # TODO: Array{Int64, 2} ?

    function RandomForest(n_trees=10::Int64, max_depth=3::Int64, max_features=nothing::Union{Int64, Nothing},
                          min_samples_split=2::Int64, min_gain=0.0::Float64, seed=123::Int64)
        trees = []

        for i=1:n_trees
            push!(trees, DecisionTree(min_samples_split, 1e-7, max_depth))
        end

        new(n_trees, max_depth, max_features, min_samples_split, min_gain, seed, trees, nothing)
    end

end

function fit!(model::RandomForest, x::Array{T,2}, y::Array{T2,1}) where {T <: Real, T2 <: Real}
    n_samples, n_features = size(x)

    # If max_features have not been defined => select it as
    # sqrt(n_features)
    if model.max_features == nothing
        model.max_features = round(Integer, sqrt(n_features))
    end

    if model.feature_indices == nothing
        model.feature_indices = []
    end

    # TODO assert dims of x and y matching
    x_y = hcat(x, y)

    # TODO: parallel -> https://docs.julialang.org/en/v1/manual/parallel-computing/
    for i in 1:model.n_trees
        # Choose one random subset of the data for each tree
        subset = sample(x_y, n_samples, seed=model.seed)  # TODO: replacement?
        x_subset, y_subset = subset[:, 1:end-1], subset[:, end]

        # Feature bagging (select random subsets of the features)
        idx = rand(1:n_features, model.max_features)

        # Save the indices of the features for prediction
        push!(model.feature_indices, idx)

        # Choose the features corresponding to the indices
        x_subset = x_subset[:, idx]

        # Fit the tree to the data
        fit!(model.trees[i], x_subset, y_subset)
    end
end

function predict(model::RandomForest, x::Array{T,2}) where T <: Real
    n_samples = size(x,1)
    y_preds = Array{Float64, 2}(undef, n_samples, model.n_trees)
    # Let each tree make a prediction on the data
    for i_tree in enumerate(model.trees)
        i, tree = i_tree
        # Indices of the features that the tree has trained on
        idx = model.feature_indices[i]

        # Make a prediction based on those features
        prediction = predict(tree, x[:, idx])

        y_preds[:, i] = prediction
    end

    y_pred = []
    # For each sample
    for i in 1:n_samples
        y = modes(y_preds[i, :])[1]
        push!(y_pred, y)
    end

    y_pred
end


function predict_probs(model::RandomForest, x::Array{T,2}; get_probs=false::Bool, index_class=true::Bool) where T <: Real
    n_samples = size(x,1)

    y_preds = Dict{Any, Array{Float64,1}}()

    # Let each tree make a prediction on the data
    for i_tree in enumerate(model.trees)
        i, tree = i_tree
        # Indices of the features that the tree has trained on
        idx = model.feature_indices[i]

        # Make a prediction based on those features
        prediction = predict_probs(tree, x[:, idx])

        #for kv in prediction:
        #    if !haskey(y_preds, )
        #end

        y_preds[:, i] = prediction
    end

    y_pred = []
    # For each sample
    for i in 1:n_samples
        if get_probs == true
            y = nothing
        else
            y = modes(y_preds[i, :])[1]
        end

        push!(y_pred, y)
    end

    y_pred
end
