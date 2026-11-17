# Contributing

1. **Branch from `main`.** Keep PRs focused, and describe *why* the change is needed.
2. **Run the checks locally:**
   ```bash
   (cd platform && python -m pytest -q && bash pentest/run_all.sh)
   (cd sim-core && python -m pytest -q)
   (cd apps/simulator && node sensors2.test.js)
   ```
3. **Platform rules** (enforced by tests):
   - SQL is literal strings with bound parameters only.
   - The tenant comes only from verified identity.
   - New endpoints need RBAC entries, schema validation and tests, including an IDOR test.
4. **Sensor models:** cite a source for every new constant in `docs/research/SENSOR_MODELS.md`, or mark it *assumed*.
5. **Data:** every new dataset needs a license entry in `sim-core/training/datasets.yaml`. Don't commit footage of real people or real site layouts.
