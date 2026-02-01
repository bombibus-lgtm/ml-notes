# ml-notes

# Unrolling RNN

When the RNN has a fixed horizon t <= T : it is equivalent to a Feed Forward Network with T hidden layers
-> sharing of params between layers (time-invariance)
-> inputs and outputs are processed incrementally

The RNN has to remember the relevant info in the latent state sequence z^t

The hidden states represent a noisy memory -> this is more powerful than memorizing fixed-length context

## Exploding and banishing gradients

The spectral norm of a matrix is defined as being the largest singular value.
if the spectral norm of the weights is < 1 -> vanishing gradients
if the spectral norm of the weights is > 1 -> exploding gradients

-> Solution is to control when memory is kept and when it is overwritten
-> Learning what new info is relevant and when to trade of with stored information

## LSTM

<img width="847" height="463" alt="image" src="https://github.com/user-attachments/assets/4a9665cc-4529-4ce7-a464-0ca2ea92f6c5" />


- Forget Gate
- Prepare new input info to be added
- Combine stored and new information
- Compute output selectively

## GRU : simpler LSTMs

<img width="601" height="214" alt="image" src="https://github.com/user-attachments/assets/a1c7aa68-d4e2-4556-a5c4-8cba2ff60708" />


- Combines forget and storage gate
- Simplify recurrence of input augmentation
- The new storage remains complex

# Linear Recurrent Unit
<img width="751" height="276" alt="image" src="https://github.com/user-attachments/assets/7099443e-3e70-47d0-bf25-aa3063b15e78" />

Advantages of LRU:

- clean understanding of long range vs short dependencies
- no requirement for mixing of channels
- Parallelization during learning

Representational power of LRUs: move modeling power into **ouput map** to compensate for simpler hidden state recurrence

<img width="677" height="153" alt="image" src="https://github.com/user-attachments/assets/a7220785-b482-4c9b-992d-8e2f8e3e6c75" />

## Teacher forcing

The problem during training is that the predicted sequence **diverge** from target one. **THE FIX** : feedback target sequence $y^*$ during training -> only learns one step predictions

## Professor teaching

we still train with teacher forcing but add **adversarial objective so the hidden-state dynamics are similar** in:
1. teacher-forced mode -> training with ground truth inputs
2. free-running mode

The idea is to reduce the difference between inference and training

## Types of attention
1. Self attention in encoder
2. Masked self attention in decoder
3. Mixed attention : queries come from decoder, values from encoder, residual recombination with decoder features

The attention blocks are stacked


Recurrent model has intelligent forgetting -> compress the context into evolving state
Attention model : store and index intelligently
