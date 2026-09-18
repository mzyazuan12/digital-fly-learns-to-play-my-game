# Embodied fly

The fly is a persistent organism. Default view is the **living room**.

```
python -m sim3d.serve
# http://127.0.0.1:8765/          living room
# http://127.0.0.1:8765/desk      monitor experiment (does not type)
```

Shiritori is a website on a monitor. The fly does **not** receive the
dictionary, the prefix, the correct word, or whose turn it is. Opening a
match does not make it type. Typing during the opponent's turn was a
lexicon autoplay bug, not the animal.

Manual override on the desk is you pressing keys. That is logged as a
developer action, not fly behavior.
