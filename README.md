# Argos Python SDK Usage Example

This is a simple visualization that subscribes to data from the the Argos system and displays it. To use it the SonaSoftSDK demo needs to be running to process and send pre-recorded data (or an actual installation connected to hardware).

The only currently support display backend is [rerun.io](https://rerun.io/) - but I tried to set it up in such a that it could be used to test out and prototype other systems and see an example of how to get the proto data etc.

It's set up to show:
- a point cloud of the bottom, accumulated each ping
- a grid (1 meter but configurable - only sent every 10 pings - configurable)
- a surface mesh made from the 1m grid (sent even less as it's slow)
- a realtime surface mesh from the live bottoms (each ping)

This current version is pretty slow - it can only keep up in short runs as sending the whole set of grids / mesh gets more expensive. There's really no edge case handling, minimal  error handling etc, it's purely meant as a prototyping example to help get going.

## To run it

Use whatever package manager you like but I recommend [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run main.py # optional --log-level DEBUG|INFO|WARNING
```
It should open re-run, and if the SonaSoftSDK demo is running, start displaying data!
