# Vendored Model Sources

This directory contains upstream model code and model-specific assets.

- Keep method integration code in:
  - `api/methods/`
  - `api/prediction_engines/`
- Keep third-party model source code and weights here under:
  - `models/<MethodName>/`

This separation keeps platform integration logic independent from vendored model code.
