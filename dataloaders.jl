module DataLoaders
    using Base.Threads: @threads, nthreads, threadid
    using LearnBase, MLDataPattern

    # TODO: PinnedMemoryDataloader

    export DataLoader, AbstractDataLoader

    abstract type AbstractDataLoader end

    struct DataLoader{TData} <: AbstractDataLoader
        data::TData
        batch_size::Int
        workers::Int
        transform
        len::Int
        function DataLoader{TData}(dataset::TData, batch_size,
                                            workers, transform) where TData
            len = nobs(dataset) ÷ batch_size
            return new{TData}(dataset, batch_size, workers, transform, len)
        end
    end

    """
        DataLoader(dataset, batch_size; [workers=nthreads-1, transform=id])

    Standard pytorch-like dataloader based on awesome Julia multithreading and Channel """
    function DataLoader(dataset::TData, batch_size::Int;
        workers=Threads.nthreads()-1, transform=identity) where TData
        return DataLoader{TData}(dataset, batch_size, workers, transform)
    end

    Base.length(loader::AbstractDataLoader) = loader.len

    function Base.show(io::IO, loader::DataLoader{TData}) where TData
        println(io, "DataLoader{$TData}[len=$(loader.len), batch_size=$(loader.batch_size)]")
    end

    function Base.iterate(dataloader::DataLoader)
        bv = BatchView(shuffleobs(dataloader.data), dataloader.batch_size)
        chan = Channel(dataloader.workers) do c
            @threads for i in eachindex(bv)
                obs = dataloader.transform(getobs(bv[i]))
                put!(c, obs)
            end
        end
        return take!(chan), (chan, 1)
    end

    function Base.iterate(dataloader::DataLoader, state::Tuple{Channel, Int})
        chan, iter = state
        val = (iter >= dataloader.len) ? nothing : (take!(chan), (chan, iter + 1))
    end

end  # module
