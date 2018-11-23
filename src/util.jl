using Random

# TODO: use Macros and Symbols

function divide_on_feature(x, idx, threshold)
    """
    Divide dataset based on if sample value on feature index is larger than
    the given threshold
    """

    x1, x2 = [], []

    for i in 1:size(x, 1)
        if x[i, idx] >= threshold
            push!(x1, x[i,:]')
        else
            push!(x2, x[i,:]')
        end
    end

    return vcat(x1...), vcat(x2...)

end

function sample(data::Array{T,2}, n::Int64; seed=123::Int64) where T <: Any
    # TODO: accept seed!
    n_samples = size(data, 1)

    # Shuffle data
    idx = shuffle(MersenneTwister(seed), 1:n_samples)
    data = data[idx, :]

    data[1:n, :]
end

function split_data(data, fraction)
    n_samples = size(data, 1)

    # Suffle data
    shuffle!(data)

    s = round(Integer, n_samples * fraction)

    return data[1:s, :], data[s:end, :]
end

function entropy(y)
    """ Calculate the entropy of label array y """
    unique_labels = unique(y)
    e = 0
    n = length(y)

    for label in unique_labels
        c = length(filter(x-> x==label, y))
        p = c / n
        e += -p * log2(p)
    end

    return e
end

function modes(values)
    # From https://rosettacode.org/wiki/Averages/Mode#Julia
    dict = Dict() # Values => Number of repetitions
    modesArray = typeof(values[1])[] # Array of the modes so far
    max = 0 # Max of repetitions so far

    for v in values
        # Add one to the dict[v] entry (create one if none)
        if v in keys(dict)
            dict[v] += 1
        else
            dict[v] = 1
        end

        # Update modesArray if the number of repetitions
        # of v reaches or surpasses the max value
        if dict[v] >= max
            if dict[v] > max
                empty!(modesArray)
                max += 1
            end
            append!(modesArray, [v])
        end
    end

    return modesArray
end

function count_distinct(seq)
    dict = Dict{Any, Int64}();
    for item in seq
        if !haskey(dict, item)
            dict[item] = 0;
        end
        dict[item] += 1
    end

    dict
end
