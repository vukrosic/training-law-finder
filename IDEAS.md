# Ideas to work on

Pick one, or bring your own. Before you start, search for the closest published paper and say how yours differs. Open questions that are not ready for a problem folder go in an issue with the `question` label.

## Ready to start

The two open problems in [problems/](problems/README.md) have data, a scoring command and a leaderboard.

The forgetting problem also has a list of open directions, from no training to one GPU:
[drafts/forgetting-rules/NEXT-STEPS.md](drafts/forgetting-rules/NEXT-STEPS.md).

## Waiting for a folder

Problems that the current data can already start on, waiting for a folder: fact memorization
(bits stored per repeat count, `bits_by_k`), best learning rate per width (one number per sweep, not per run: see [GUIDE.md](GUIDE.md#1-pick-one-target)), schedule shape at a fixed
learning-rate sum, mixture-of-experts load balance. Pull requests welcome.

<details>
<summary>60 more topics a formula could answer</summary>

| Topic | The formula predicts |
|---|---|
| **Loss and scaling** | |
| Final loss | validation loss at the end of training |
| Loss curve | validation loss at every step of training |
| Early stopping guess | final loss, read from the first 10% of the loss curve |
| Compute-optimal size | the model size with the lowest loss for a fixed compute budget |
| Depth vs width | the best ratio of layers to width for a given parameter count |
| Repeated data | loss when training on the same data several times |
| Diminishing returns | the point where more training tokens stop lowering loss |
| Small-to-large transfer | a large model's loss, predicted from small models' runs |
| **Learning rate and schedule** | |
| Best learning rate | the peak learning rate that gives the lowest loss |
| Learning-rate schedule | final loss under any schedule shape |
| Warmup length | the shortest warmup that still trains stably |
| Cooldown length | the best share of training spent decaying the learning rate |
| Schedule switch | loss after the learning rate changes mid-run |
| Learning-rate sensitivity | how much loss rises when the learning rate is off by 2× |
| Restart cost | loss lost when training resumes with a new schedule |
| **Optimizer** | |
| Muon weight decay | the best weight decay for the Muon optimizer |
| Muon vs Adam gap | how much lower Muon's loss is than Adam's at the same compute |
| Momentum | the best momentum value |
| Gradient clipping | the best clipping threshold |
| Update size | how large each weight update is at every step |
| Newton-Schulz steps | how many orthogonalization steps Muon needs |
| Embedding learning rate | the best learning rate for embeddings, relative to hidden layers |
| **Batch size and data** | |
| Critical batch size | the largest batch that still trains efficiently |
| Best batch size | the batch size with the lowest loss for a fixed token budget |
| Batch-size ramp | loss when the batch size grows during training |
| Data mixture | loss on each domain for a given mix of training data |
| Data order | how much the order of training data changes final loss |
| Sequence length | the best training context length for a given budget |
| Noisy data | how much loss rises as a share of the data is corrupted |
| **Memory and facts** | |
| Fact memorization | the share of facts a model can recall |
| Forgetting | how much a model forgets after it stops seeing a fact |
| Memory capacity | how many facts a model can store per parameter |
| Repetitions needed | how many times a fact must be seen before it is recalled |
| Timing of facts | how recall depends on when in training a fact was seen |
| Interference | how much learning new facts erases old ones |
| **Stability** | |
| Loss spikes | the learning rate at which training breaks |
| Divergence time | the step at which an unstable run blows up |
| Attention logit growth | how large attention logits grow during training |
| Weight norm growth | how large the weights grow during training |
| Low-precision safety | whether a run stays stable in 8-bit or 16-bit training |
| **Architecture** | |
| Attention scale | the best attention temperature |
| RoPE base | the best rotary-position base for a given context length |
| Number of heads | the best number of attention heads for a given width |
| MLP width | the best hidden size of the feed-forward layer |
| Residual scale | the best scale for each layer's residual branch |
| Init scale | the best initial weight size per layer type |
| **Mixture of experts** | |
| Number of experts | the best number of experts for a given compute budget |
| Expert load balance | how unevenly tokens spread across experts during training |
| Top-k routing | the best number of experts per token |
| Dense vs MoE gap | how much lower an MoE model's loss is than a dense model's at the same compute |
| **Fine-tuning and RL** | |
| Fine-tuning forgetting | how much pretraining skill is lost during fine-tuning |
| LoRA rank | the smallest LoRA rank that matches full fine-tuning |
| RL reward curve | reward at every step of reinforcement learning |
| KL penalty | the best KL coefficient in RL fine-tuning |
| Distillation gap | how close a small student gets to its teacher |
| **Compression and inference** | |
| Quantization loss | how much loss rises after 4-bit or 8-bit quantization |
| Pruning loss | how much loss rises when a share of weights is removed |
| KV-cache eviction | accuracy when the cache keeps only some tokens |
| Speculative decoding | how many draft tokens get accepted |

</details>
