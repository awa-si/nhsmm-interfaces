## Summary

Describe the change and why it is needed.

## Contract / compatibility

Describe any effect on adapter contracts, feature ordering, timestamps, runtime lifecycle, walk-forward ordering, public exports, core API compatibility, or framework integration. Write `none` if there is no contract change.

## Verification

List the exact commands and tests actually executed, including relevant results. If a framework dependency was unavailable, state that explicitly.

## Notes

Call out unresolved assumptions, follow-up work, or validation that remains outside this pull request.

## Checklist

- [ ] Change is scoped to the intended interface/integration behavior.
- [ ] Tests were added or updated when behavior changed.
- [ ] Public adapter/evaluator changes include compatibility review.
- [ ] No private NHSMM internals were imported or mirrored.
- [ ] Framework-specific behavior remains in the appropriate adapter package.
- [ ] Documentation reflects implemented behavior without unsupported claims.
- [ ] Unavailable checks or remaining risks are stated explicitly.
