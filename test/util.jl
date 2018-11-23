using Ensemblearn
using CSV
using DataFrames

export 
    get_iris_data

pkg_path = abspath(joinpath(dirname(pathof(Ensemblearn)), ".."))
path_to_iris_data = "$(pkg_path)/test/data/iris.csv"

dict_mapping = Dict("Iris-setosa"=> 0, "Iris-versicolor"=> 1, "Iris-virginica"=> 2)

function get_iris_data()
    # Read test data
    df = CSV.File(path_to_iris_data) |> DataFrame

    # Filter data to have only 2 classes
    df = filter(r -> r.Name != "Iris-virginica", df)

    # Get labels as integers
    y = map(r -> dict_mapping[r], df.Name)
    x = hcat(df.PetalLength, df.PetalWidth, df.SepalLength, df.SepalWidth)

    x, y
end