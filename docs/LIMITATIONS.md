# Limitations

- Real mDeBERTa probe/fine-tuning, XLM-R, and final held-out test evaluations remain **PENDING**
  on a separate GPU laptop. Current test runs use tiny random models for software smoke checks.
- The earlier project's recorded probe was weak (accuracy 0.4167 versus recorded majority
  accuracy 0.4060), with zero Neutral recall. These historical values are not reproduced here.
- This rebuild's test split has 610 Negative, 306 Neutral, and 584 Positive examples; its
  train-majority-label test accuracy is 0.389333. These differ from the earlier project split
  and must not be compared as if they were the same examples.
- The benchmark contains English, French, and Chinese only. Arabic and Italian UI samples show
  extensibility but are explicitly not evaluated.
- Attention weights are diagnostics, not faithful causal explanations.
- QLoRA is a separate five-example causal-LM experiment. It does not measure or improve the
  three-class classifier and is excluded from all benchmark reports.
- The demo is a research tool, not a production service; it does not provide calibration,
  moderation, privacy review, or reliability guarantees.
