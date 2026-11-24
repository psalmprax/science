"""The version of ``propose`` every run of the loop starts from, as text.

Status: experimental

Deliberately plain: one fixed shape of each variable (the variable, its square,
its reciprocal, its root, its logarithm, a decay), added up with constants
fitted by least squares, and the four combinations that fit best. It contains
no constant inside a function, no product and no ratio, so most laws are out
of its reach; what a model adds to it is then the model's.

It is text and not a module on purpose. A candidate is never imported: it is
checked by ``restricted`` and run by ``jail``, and that holds for this one too.
"""
from __future__ import annotations

SEED_GENERATOR = '''\
def fit(columns, y):
    """Least squares of y on the columns and a constant: the coefficients and the error left."""
    A = np.column_stack(columns + [np.ones(len(y))])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    rest = y - A.dot(coef)
    return coef, float(np.sqrt(np.mean(rest ** 2)))


def written(coef, terms):
    """The fitted law as text: coefficient times term, added up, and the constant."""
    parts = [f"{coef[i]}*{terms[i]}" for i in range(len(terms))]
    return " + ".join(parts + [f"{coef[len(terms)]}"])


def shapes(name, x):
    """A few fixed shapes of one variable: the text and the values."""
    return [(name, x), (f"{name}**2", x ** 2), (f"1/{name}", 1.0 / x), (f"sqrt({name})", np.sqrt(np.abs(x))),
            (f"log({name})", np.log(np.abs(x) + 1e-12)), (f"exp(-{name})", np.exp(-x))]


def propose(X, y, names):
    found = []
    combos = [[]]
    for i in range(X.shape[1]):
        combos = [combo + [shape] for combo in combos for shape in shapes(names[i], X[:, i])]
    for combo in combos:
        coef, err = fit([values for text, values in combo], y)
        if np.all(np.isfinite(coef)):
            found.append((err, written(coef, [text for text, values in combo])))
    found.sort()
    return [text for err, text in found[:4]]
'''
