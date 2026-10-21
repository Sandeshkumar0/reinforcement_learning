# Verification run

The human and bomber each completed 1,024 PPO training steps with rollout size 128. The bomber used the saved human policy; the human checkpoint was unchanged by bomber training. This is a short integration check, not a trained gameplay benchmark.

The comparison used seeds 100–109 for both controllers, a 240×240 map, one human, two zombies, one bomber, no obstacles, a safe zone, and a 160-step limit. Explosion damage was 60 HP. Full settings are in `config.json`; per-episode results are in `comparison.csv` and aggregate metrics in `report.json`.

| Mean per episode | Short-trained pair | Random pair |
| --- | ---: | ---: |
| Human escape rate | 30% | 10% |
| Human death rate | 60% | 80% |
| Zombie HP removed | 28 | 28 |
| Zombie kills | 0.1 | 0.1 |
| Useful explosions | 0.4 | 0.5 |
| Invalid/wasted explosion attempts | 15.7 | 10.3 |
| Human reward | -99.349 | -162.422 |
| Bomber reward | -45.629 | -58.767 |

The mixed results, particularly unchanged zombie damage and more invalid attempts, do not establish a reliable improvement. Run the full curriculum and longer bomber training, then compare on more held-out seeds.

Reproduce training and comparison with:

```bash
python -m rl.verify_training --timesteps 1024 --episodes 10
```

The accompanying regression suite checks combat rules and rewards, Gymnasium compatibility, real PPO updates, frozen human weights, checkpoint save/load/resume, logged training losses, and seeded evaluation reproducibility. Pygame combat rendering and RGB-array output were also checked with SDL's dummy display/audio drivers.
