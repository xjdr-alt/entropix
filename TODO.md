#TODO

## Repo
 - Code and Docs cleanup (this is very hacky right now)
 - Concept explanation and simple impelmentation examples

## Vanilla Sampler
 - Repition penalties (DRY, Frequency, etc)
 - min_p

## Entropy Sampler
 - Base sampler with dynamic thresholds and no beam / best of N

## Model
 - TPU Splash, TPU Paged and GPU Flash attention for jax
 - Flex attention for Torch
 - Parallel CoT Attenion Masks

## Generation
 - Genration loop does not properly handle batching of different sized inputs, fix
 - Batched Best of N based on sampler output
 - Parallel CoT (Batched) Generation
 - Captain Planet entropy from the base model when we hit entropy collapse

## Tests
 - port over test suite and setup with ref models
 - write sampler test

## Server
 - OpenAI compat server (use sglang impl?)
 - continious batching

## Evals
 - Set up eval suite
   - Eluther eval harness
   - OAI simple evals
   - EQ Bench?
   - Berkley function bench?
   - swe-bench?
   - aider?