"""An adversary for the referee: worlds built so that a wrong law looks right.

Status: experimental

The referee's measured record (which laws it reported, how many were wrong) comes
from benchmarks nobody designed to defeat it. This package designs them:

- ``counterfeit``: laws of a different form that fit the same rows almost as well
  as the true one, made by exchanging one building block of a law for another
  and refitting.
- ``worlds``: a world is a law, the candidates a search is said to have returned
  (counterfeits, with or without the law), how the rows are sampled, and what
  noise they carry. Rows and noise are drawn from the world's own seed.
- ``attack``: draws worlds inside an arena (which of those settings may vary) and
  changes them step by step towards the outcome a referee must not produce: a
  reported law that is wrong.

Nothing here decides anything. The decision under attack is passed in as a
function, and what counts as wrong is the scoring the benchmarks already use.
"""
