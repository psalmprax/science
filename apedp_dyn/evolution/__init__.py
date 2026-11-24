"""A search that a language model may rewrite, under a judge it cannot touch.

Status: experimental

The second step after ``proposers.agentic_variation``, and the closer one to
Agentic Variation Operators (Chen et al., 2026, arXiv:2603.24517): there the
agent varies candidate laws for one problem; here it varies a piece of the
search's own code, and each version is scored on many problems.

- ``restricted``: the subset of Python a candidate may be written in, checked
  from its syntax tree before anything runs.
- ``jail``: runs a checked candidate in a separate process that has no
  environment, cannot write a file, and is killed when it runs over.
- ``operator``: the loop. A model is shown the contract, the lineage of earlier
  versions with their scores and failures, and writes the next version; what
  the check or the run rejects goes back to it for repair.
- ``evolved_proposer``: a checked candidate as a proposer, for the path.

What holds throughout: the candidate returns text only, and the judge parses
that text with the same whitelist the language-model proposers use; the judge
is ordinary code of this repository that no candidate can reach; nothing the
loop produces is installed by the loop. A version becomes part of the system
only by a reviewed commit, after one preregistered test on problems it was
never scored on.
"""
