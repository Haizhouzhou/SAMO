# Method

SAMO means residualized predictability-sensitive reader-profile modeling. The public task is CopCo dyslexia-labelled vs typical/control reader-level classification.

The pipeline validates word-level observations, adds or validates predictability-style columns, fits a residualizer inside each reader-disjoint fold, aggregates residualized gaze behavior into one row per reader, and evaluates a reader-level classifier with LOPO splits.

Residualization is fold-local. In each fold, the residualizer is fit only on training-reader word rows and then applied to the held-out reader. Direct reader IDs, labels, target columns, direct text IDs, and speech IDs are not allowed as predictors.

The public synthetic pipeline is a functional smoke test. It demonstrates code paths and contracts, not scientific evidence.
