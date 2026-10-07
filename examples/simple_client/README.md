# Simple Client

A minimal client that sends observations to the server and prints the inference rate.

You can specify which runtime environment to use using the `--env` flag. You can see the available options by running:

```bash
uv run examples/simple_client/main.py --help
```

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/simple_client/compose.yml up --build
```

## Without Docker

The server only serves LIBERO policies (`--env LIBERO` is the only server-side
mode). The client's `--env` flag selects a random-observation generator
(`DROID`, `ALOHA_SIM`, `LIBERO`) used to exercise the server without a real
robot or simulator.

Terminal window 1:

```bash
uv run examples/simple_client/main.py --env LIBERO
```

Terminal window 2:

```bash
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config=<config_name> \
  --policy.dir=checkpoints/<config_name>/<exp_name>/<step>
```
