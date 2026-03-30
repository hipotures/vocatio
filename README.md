# Vocatio

This repository is the new home for the event workflow system that is currently still being migrated out of `scriptoza`.

Current state:

- `scripts/pipeline/` contains a safe copied snapshot of the working event pipeline scripts
- the original scripts still remain in `scriptoza`
- no behavior was removed from the current production workflow yet

Planned direction:

- keep the existing scripts runnable during migration
- move the workflow toward a proper application structure
- later replace script-to-script orchestration with shared services, GUI actions, and web delivery
