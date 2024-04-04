# What is memory optimization?

## A simple example

Actuarial models involve projecting financial quantities across timesteps. It is common for the value at a timestep to depend on values from the previous timestep. For a simple example

* `f(t)` is used by `g(t)`
* `g(t)` is used by `g(t+1)`, `f(t+1)`, and `h(t)`
* `h(t)` is not used by any other functions

## Why clear the cache

The dependency structure of actuarial models requires caching to run recursive function calls efficiently. Suppose we have `F` functions each run for `T` timesteps, and each function returns an array of size `P`. At the end of the calculation the memory consumption of the caches will be `F * T * P`.

In practice, it is common for a timestep to be a month. Projecting results 20 years into the future means that `T = 240`. But because we know that functions only depend on the previous timestep, we know that after calculating results for `T = 2` that we can clear all caches where `T == 1`, they are no longer needed.

Clearing the cache in this manner, the number of cached values would no longer depend on the number of timesteps, and we reduce the cache size to less than 1% of its original size.

!!! example

    16 GB can store 2 billion 64 bit numbers. Consider a model with 10 formulas, 240 timesteps. `F * T * P` = `10 * 240 * P` = 2 billion and `P = 833,333` which is pretty good. Optimizing away the dependency on timesteps, `F * P` =  `10 * P` = 2 billion and P = 200,000,000.

!!! note

    In general `f(t)` could depend on `g(t-12)`, so it is unwise to hard code cache deletions for an entire timestep without analyzing the dependency graph.

## Implementation details

We collect the dependency graph of the functions, this is done by running the model with a small sample of the data, `P = 1` for example.

![dependency graph for first three timesteps, light mode](./_static/dependency-graph-light.svg#only-light)
![dependency graph for first three timesteps, dark mode](./_static/dependency-graph-dark.svg#only-dark)

Internally, we track the last function to use a cached value. For an execution order of
`["f(0)", "g(0)", "h(0)", "f(1)", "g(1)", "h(1)", ... ]` the arrow between `g(0)` and `g(1)` in the diagram below represents that `g(1)` is the last function to require the cached value of `g(0)`.

![](./_static/memopt-last-needed-dark.svg#only-dark)
![](./_static/memopt-last-needed-light.svg#only-light)

Since `g(1)` is the last function to require the cached value of `g(0)` we make a new data structure, the arrow between `g(1)` and `g(0)` below now represents that `g(1)` clears the cache of `g(0)`. Also, functions with no dependencies can clear their cache after being calculated.

![](./_static/memopt-can-clear-dark.svg#only-dark)
![](./_static/memopt-can-clear-light.svg#only-light)





