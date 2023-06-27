include("./src/bistable_mean_field.jl")
using NPZ

#store output in equivalent format ready for python plot scripts
function main()
    hs=[0.1,0.4,0.7]
    names="activity_".*["low","intermediate","high"]
    data = Dict{String, Any}()
    data["h_selection"] = hs
    seed=1000
    for (h,name) in zip(hs,names)
        println(h, " ", name)
        times, rhos = simulate(h, seed=seed)
        data[name] = rhos
        data["bins"] = times
        seed+=1
    end
    npzwrite("../manuscript/data/theory_activity.npz", data)
end

main()
