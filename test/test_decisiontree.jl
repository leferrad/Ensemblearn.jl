using Ensemblearn
using Test

include("util.jl")


function test_create_tree(min_score=0.8::AbstractFloat)
    # Get iris data
    x, y = get_iris_data()
    n = size(y, 1)

    # Create a DecisionTree
    dt = DecisionTree()

    # Fit DecisionTree
    fit!(dt, x, y)

    # Get predictions over x
    p = predict(dt, x)

    # Obtain tuples of (pred, actual)
    p_y = collect(zip(p, y))

    # Score as rate of hits (accuracy)
    score = sum(map(t -> t[1] == t[2], p_y)) / Float64(n)

    # Assert a minimum score on predictions
    @test score >= min_score

    # Print tree obtained
    println("Resulting tree:")
    println("---------------")
    print_tree(dt)

end

@testset "RunTestDecisionTree" begin
    # Normal case
    test_create_tree()
end
