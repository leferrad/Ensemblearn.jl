using Test

@testset "TestEnsemblearn" begin
    include("test_decisiontree.jl")
    include("test_randomforest.jl")
end
