# Data Access

This release candidate does not include CopCo data, raw gaze files, participant-level prediction rows, real word-level feature tables, or model artifacts. Users must obtain CopCo separately through the appropriate access route and then provide local file paths in a copied configuration file.

Expected local inputs are a word-level feature table and a reader-label table. The label is reader-level; word rows are repeated observations for profile construction and must not be treated as independent target samples.

The example configuration uses placeholders only and intentionally does not point to any private or bundled data location.
