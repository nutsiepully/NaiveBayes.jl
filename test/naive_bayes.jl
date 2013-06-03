
X = reshape(rand(0:5, 600), 6, 100)
Y = [1:6]

gamma = naive_bayes_fit(X, Y)
print(gamma(X[1]))
