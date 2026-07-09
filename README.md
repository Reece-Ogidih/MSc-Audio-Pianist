# MSc Audio Pianist

This project builds a one-hand RoboPianist simulation benchmark for comparing two audio-to-control pipelines:

- indirect: audio to symbolic music to action;
- direct: audio to action.

The current first benchmark is deliberately small:

- one right Shadow Hand;
- local key range initially;
- no sustain pedal;
- RoboPianist's default one-hand forearm translation controls;
- monophonic MIDI clips first.

RoboPianist is kept as an external dependency under `third_party/`, with the checked-out commit recorded in `third_party/robopianist_commit.txt`.
