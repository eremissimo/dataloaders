include(".\\dataloaders.jl")
using Base.Threads
using .DataLoaders

function transform(x)
    sleep(rand())
    println("uploaded batch $x by worker: $(threadid())")
    return x
end

function test_loader()
    dataset = 1:100
    loader = DataLoader(dataset, 5; transform=transform)
    @show loader
    for (i, batch) in enumerate(loader)
        sleep(rand())
        println("$i ~ processed batch $batch by worker: $(threadid())");
    end
end
