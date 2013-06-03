
# Method of computation
# Assumes it works over 2D arrays

function count_inputs_per_class(Y, C)
  # Don't know how to specify type of C
  # Anyhow since it's not known at compile time,
  # doesn't make a performance difference I suppose
  #
  # Maybe find a more fancy comprehension way
  # to do it one line
  N_c = Dict{Any, Int64}()
  for y in Y
    N_c[y] = haskey(N_c, y) ? N_c[y] + 1 : 1
  end
  N_c
end

# X - training set inputs
# Y - training set results
# c - class for which tokens qualify
# t - token for which count needs to be taken
function count_tokens_T_in_class(X, Y, c, t)
  count = 0
  for i = 1:size(X, 1)
    if !(Y[i] == c)
      continue
    end
    for x in X[i,:]
      if x == t
        count += 1
      end
    end
  end #Loop over training set
  count
end

function count_tokens_in_class(X, Y, c)
  count = 0
  for i = 1:size(X, 1)
    if (Y[i] == c)
      count += length(X[i,:])
    end
  end #Loop over training set
  count
end

# Currently supports only Multinomial Naive Bayes
# TODO: Add optional options hash as 3rd parameter
#
# X - Array containing training set inputs.
# Y - Vector containing training set results. (TODO: Change to vector parameter)
# returns - learned classification function gamma
#
function naive_bayes_fit(X::Array, Y::Array)
  V = unique(X)
  C = unique(Y)

  N = size(X, 1)
  N_c = count_inputs_per_class(Y, C)

  prior = Dict{Any, Float64}()
  #P_T_c = Array(Float64, (length(V), length(C)))
  # Might slow down due to the absence of type
  P_T_c = Dict()
  for c in C
    prior[c] = N_c[c] / N
    sum_T_ct = count_tokens_in_class(X, Y, c) + length(V)
    for t in V
      T_ct = count_tokens_T_in_class(X, Y, c, t) + 1
      P_T_c[(t, c)] = T_ct / sum_T_ct
    end
  end

  gamma(d) = naive_bayes_predict(prior, P_T_c, C, d)
end

function naive_bayes_predict(prior, P_T_c, C, d)
  n_classes = length(C)
  scores = Array(Float64, n_classes)
  for i = 1:n_classes; c = C[i]
    scores[i] = prior[c]
    for t in d
      if (haskey(P_T_c, (t, c)))
        scores[i] += log(P_T_c[(t, c)])
      end
    end
  end
  C[indmax(scores)]
end

