using SafeTestsets

function recursive_readdir!(cur_dir, path, list_of_scripts)
    fullpath = joinpath(cur_dir, path)
    if isdir(fullpath)
        for entry in readdir(fullpath)
            recursive_readdir!(fullpath, entry, list_of_scripts)
        end
    end
    endswith(path, ".jl") && push!(list_of_scripts, fullpath)
    return
end

scripts = []
recursive_readdir!(@__DIR__, "", scripts)

for script in scripts
    @info "Running $script"
    @time @eval @safetestset $script begin include($script) end
end
