# Classroom poll images

`model-size-choices-portrait.png` stacks three conceptual alternatives vertically for Mentimeter. The separate `model-size-choice-a.png`, `-b.png`, and `-c.png` files contain the same options. These sketches are not measured results.

**Poll question:** We train neural networks on the same 200 clothing images and evaluate them on the same held-out validation images. We increase the width of one hidden layer, keeping the optimizer and number of training epochs fixed. Each network starts from scratch. Which sketch best predicts what we will observe?

Use single-choice options **A / B / C**, and collect responses before showing the notebook's measured model-size plot. The horizontal axis is model size, not training time. Blue is training error; dashed orange is validation error.

For a different discussion, ask **“Which of these patterns could occur in practice with a fixed training budget?”** and allow multiple selections. All three can occur under different data and optimization conditions; only asking students to predict this particular experiment makes B the closest answer. A wider model's representational capacity does not guarantee that a finite training run finds equally good parameters.

`model-size-poll.png` is the measured notebook plot. `mixup-label-noise-study.png` shows the paired noisy-label experiment; it should be read with the notebook's checkpoint-selection comparison, rather than as a claim that mixup always improves accuracy.
