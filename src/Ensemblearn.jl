__precompile__()

module Ensemblearn

#export 
#    run_tesseract, 
#    run_and_get_output

include("decisiontree.jl")
include("randomforest.jl")
include("util.jl")

Ensemblearn

end # module
