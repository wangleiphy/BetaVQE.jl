module VAN

using StatsBase
import ChainRulesCore: @non_differentiable, NoTangent, Tangent, rrule
using Optimisers: update!
import Zygote

export AutoRegressiveModel, get_logp, model_dispatch!, gen_samples, model_parameters
export exact_free_energy, free_energy
export AbstractSampler
export PSAModel, get_logp, gen_samples, model_parameters, model_dispatch!

include("sampler.jl")
include("arm.jl")
include("psa.jl")
include("exact.jl")
include("loss.jl")
include("utils.jl")
include("zygote_patch.jl")

end # module
